# train2.5.py --- initial draft code 
##################################################################
# Refer to the latest version of this file for the most up-to-date code. (File name : train_with_stockfish_V2.py)
##################################################################

import os
import csv
import time
import random
import chess
import chess.engine
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.mcts import mcts_search, MCTSNode, ucb_score
from src.config import Config
from src.model import ChessNet
from src.utils import enforce_single_king
from src.phase2_helpers import evaluate_position, board_to_tensor, move_to_index

class PPOTrainer:
    def __init__(self):
        self.writer = SummaryWriter(log_dir="runs/ppo_chess_" + time.strftime("%Y%m%d-%H%M%S"))
        self.global_step = 0
        self.elo_prog = []
        self.win_rates = []
        
        self.model = ChessNet().to(Config.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.PHASE1_5["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        self.current_elo = self._load_checkpoint()
        
        self.engine = self._init_engine()
        
        # Validation opponents
        self.validation_levels = [200, 800, 1200, 1500, 1800, 2000]
        
    def _load_checkpoint(self):
        try:
            ckpt = torch.load("/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/Kaggle/better.pth", map_location=Config.DEVICE)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            return ckpt.get("elo", Config.PHASE1_5["start_elo"])
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return Config.PHASE1_5["start_elo"]
    
    def _init_engine(self):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)
            return engine
        except Exception as e:
            print(f"Failed to initialize Stockfish engine: {e}")
            raise

    def get_game_result(self, board, is_model_white):
        """Enhanced reward shaping with material and positional considerations"""
        if board.is_checkmate():
            return 1.0 if board.turn != (chess.WHITE if is_model_white else chess.BLACK) else -1.0
        
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return -0.1
        
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.5,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        score = 0
        for square, piece in board.piece_map().items():
            val = piece_values[piece.piece_type]
            if piece.color == (chess.WHITE if is_model_white else chess.BLACK):
                score += val
            else:
                score -= val
        
        positional_bonus = 0
        if self.current_elo > 1000: 
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            for sq in center_squares:
                piece = board.piece_at(sq)
                if piece and piece.color == (chess.WHITE if is_model_white else chess.BLACK):
                    positional_bonus += 0.1
        
        normalized_score = (score + positional_bonus) / 40.0  
        return np.tanh(normalized_score * 3)  

    def simplify_position(self, board, current_elo):
        """Simplify position for lower ELO training"""
        if current_elo < 500 and random.random() < 0.4:
            pieces_to_remove = random.randint(1, 3)
            non_king_squares = [
                sq for sq in chess.SQUARES 
                if board.piece_at(sq) and 
                board.piece_at(sq).piece_type != chess.KING and
                sq not in (chess.E1, chess.E8)
            ]
            for sq in random.sample(non_king_squares, min(pieces_to_remove, len(non_king_squares))):
                board.remove_piece_at(sq)

    def play_game_ppo(self, board, level_config, is_model_white, tau):
        """Play game using PPO with MCTS enhancements"""
        game_history, value_history, log_prob_history = [], [], []
        self.simplify_position(board, self.current_elo)
        
        while not board.is_game_over() and len(board.move_stack) < Config.MAX_MOVES:
            if board.turn == (chess.WHITE if is_model_white else chess.BLACK):
                
                if self.current_elo > 1000:
                    move_probs = self.run_mcts(board, Config.PHASE1_5["mcts_sims"])
                    legal_moves = list(board.legal_moves)
                    probs = np.array([move_probs[move_to_index(m)] for m in legal_moves])
                else:
                   
                    with torch.no_grad():
                        t = board_to_tensor(board).unsqueeze(0).to(Config.DEVICE)
                        policy_logits, value = self.model(t)
                        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    legal_moves = list(board.legal_moves)
                    legal_idx = [move_to_index(m) for m in legal_moves]
                    probs = policy[legal_idx]
                
                exploration = max(0.1, 1.0 - self.current_elo / 2000.0)
                probs = probs * (1 - exploration) + exploration / len(probs)
                probs = np.clip(probs, 1e-8, None)
                probs = probs ** (1.0 / tau)
                probs /= probs.sum()
                
                if np.any(np.isnan(probs)) or probs.sum() <= 0:
                    probs = np.ones_like(probs) / len(probs)
                
                move_idx = np.random.choice(len(legal_moves), p=probs)
                move = legal_moves[move_idx]
                logp = np.log(probs[move_idx] + 1e-10)
                
                game_history.append({
                    'board': board.copy(),
                    'action': move,
                    'log_prob': logp,
                    'value': value.item(),
                    'legal_moves': legal_moves,
                    'legal_indices': legal_idx
                })
                value_history.append(value.item())
                log_prob_history.append(logp)
            else:
                if self.current_elo < 600:
                    move = random.choice(list(board.legal_moves))
                else:
                    try:
                        res = self.engine.play(
                            board,
                            chess.engine.Limit(
                                time=level_config['time'],
                                depth=level_config['depth']
                            )
                        )
                        move = res.move
                    except Exception as e:
                        print(f"Engine error: {e}, using random move")
                        move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        result = self.get_game_result(board, is_model_white)
        returns = self.calculate_returns(result, value_history)
        
        for i, d in enumerate(game_history):
            d['return'], d['advantage'] = returns[i], returns[i] - d['value']
        
        return result, game_history

    def calculate_returns(self, result, value_history):
        """Calculate returns with GAE (Generalized Advantage Estimation)"""
        returns = []
        advantages = []
        last_value = result
        last_gae = 0
        gamma = Config.PHASE1_5["gamma"]
        lam = 0.95  
        
        for v in reversed(value_history):
            delta = last_value - v
            last_gae = delta + gamma * lam * last_gae
            returns.insert(0, last_gae + v)
            last_value = v
        
        return returns

    def run_mcts(self, board, simulations):
        """Run MCTS and return move probabilities"""
        root = MCTSNode(board.copy())
        
        for _ in range(simulations):
            node = root
            while node.children:
                node = max(node.children, key=lambda x: ucb_score(x))
            
            if not node.board.is_game_over():
                legal_moves = list(node.board.legal_moves)
                
                board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
                with torch.no_grad():
                    policy_logits, _ = self.model(board_tensor)
                policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                
                for move in legal_moves:
                    child_board = node.board.copy()
                    child_board.push(move)
                    child = MCTSNode(child_board, node)
                    child.policy = policy[move_to_index(move)]
                    node.children.append(child)
            
            value = evaluate_position(node.board, self.model)
            
            while node:
                node.visits += 1
                node.value += value
                node = node.parent
    
        move_probs = np.zeros(4672)  
        for child in root.children:
            move = child.board.peek()
            move_probs[move_to_index(move)] = child.visits
        
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()  
        
        return move_probs

    def update_model_ppo(self, game_data):
        """Update model with proper gradient accumulation"""
        if not game_data:
            return
            
        self.model.train()
        
        boards = torch.stack([board_to_tensor(d['board']) for d in game_data]).to(Config.DEVICE)
        old_logp = torch.tensor([d['log_prob'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        returns = torch.tensor([d['return'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        advantages = torch.tensor([d['advantage'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        legal_moves_list = [d['legal_moves'] for d in game_data]
        legal_idx_list = [d['legal_indices'] for d in game_data]
        actions = [d['action'] for d in game_data]
        for _ in range(Config.PHASE1_5.get('ppo_epochs', 4)):
            perm = torch.randperm(len(game_data))
            for i in range(0, len(game_data), Config.PHASE1_5.get('batch_size', 256)):
                idx = perm[i:i+Config.PHASE1_5.get('batch_size', 256)]
                b = boards[idx]
                olp = old_logp[idx]
                rtn = returns[idx]
                adv = advantages[idx]
                
                policy_logits, value_preds = self.model(b)
                new_probs = F.softmax(policy_logits, dim=1)
                
                new_logp, ent = [], []
                for j, orig_idx in enumerate(idx):
                    li = legal_idx_list[orig_idx]
                    lprobs = new_probs[j, li]
                    lprobs = lprobs / (lprobs.sum() + 1e-10)
                    act = actions[orig_idx]
                    aidx = legal_moves_list[orig_idx].index(act)
                    new_logp.append(torch.log(lprobs[aidx] + 1e-10))
                    ent.append(-torch.sum(lprobs * torch.log(lprobs + 1e-10)))
                
                new_logp = torch.stack(new_logp)
                ent = torch.stack(ent)
                ratio = torch.exp(new_logp - olp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1-0.2, 1+0.2) * adv
                policy_loss = -torch.min(s1, s2).mean()
                
                value_preds = value_preds.squeeze()
                value_clipped = value_preds + (value_preds - value_preds.detach()).clamp(-0.5, 0.5)
                v_loss = torch.max(F.mse_loss(value_preds, rtn),
                                 F.mse_loss(value_clipped, rtn))
                ent_loss = -ent.mean()
                
                total = policy_loss + 0.5 * v_loss + Config.PHASE1_5.get('entropy_coeff', 0.01) * ent_loss
                
                self.optimizer.zero_grad()
                total.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                self.writer.add_scalar("Loss/policy", policy_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/value", v_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/entropy", ent_loss.item(), self.global_step)
                self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], self.global_step)
                self.global_step += 1

    def validate_model(self):
        """Validate against multiple Stockfish levels"""
        validation_results = {}
        for level in self.validation_levels:
            if level > self.current_elo + 300: 
                continue
                
            wins, losses, draws = 0, 0, 0
            for _ in range(Config.EVAL_GAMES // 2): 
                board = chess.Board()
                result, _ = self.play_game_ppo(
                    board,
                    Config.STOCKFISH_LEVELS.get(level, {"time":0.1,"depth":1}),
                    is_model_white=True,
                    tau=1.0
                )
                if result > 0.5: wins += 1
                elif result < -0.5: losses += 1
                else: draws += 1
                
                board = chess.Board()
                result, _ = self.play_game_ppo(
                    board,
                    Config.STOCKFISH_LEVELS.get(level, {"time":0.1,"depth":1}),
                    is_model_white=False,
                    tau=1.0
                )
                if result > 0.5: wins += 1
                elif result < -0.5: losses += 1
                else: draws += 1
            
            win_rate = wins / (wins + losses + draws)
            self.writer.add_scalar("Progress/WinRate", win_rate, self.global_step)
            self.writer.add_scalar("Progress/ELO", self.current_elo, self.global_step)
            validation_results[level] = win_rate
            self.writer.add_scalar(f"Validation/Level_{level}", win_rate, self.global_step)
        
        return validation_results

    def save_checkpoint(self, is_final=False):
        """Save model checkpoint with comprehensive info"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "elo": self.current_elo,
            "win_rates": self.win_rates,
            "global_step": self.global_step
        }
        
        filename = "final.pth" if is_final else f"checkpoint_elo_{self.current_elo}.pth"
        torch.save(checkpoint, f"models/ELO1.5/{filename}")
        
        torch.save(checkpoint, "models/ELO1.5/latest.pth")

    def train(self):
        """Main training loop"""
        os.makedirs("models/ELO1.5", exist_ok=True)
        os.makedirs("Graphs/stockfish", exist_ok=True)
        
        epoch = 0
        while self.current_elo <= Config.PHASE1_5["max_elo"]:
            level = Config.STOCKFISH_LEVELS.get(self.current_elo, {"time":0.1,"depth":1})
            tau = max(0.5, 2.0 - (self.current_elo / Config.PHASE1_5["max_elo"]) * 1.5)
            game_data, results = [], []
            for game_idx in tqdm(range(Config.PHASE1_5["games_per_level"]), desc="Games"):
                board = chess.Board()
                is_white = (game_idx % 2 == 0)
                res, hist = self.play_game_ppo(board, level, is_white, tau)
                game_data.extend(hist)
                results.append(res)
                freq = max(5, Config.PHASE1_5["update_freq"] - self.current_elo // 200)
                if (game_idx+1) % freq == 0 and game_data:
                    self.update_model_ppo(game_data)
                    game_data = []
            
            validation_results = self.validate_model()
            
            wins = sum(1 for r in results if r > 0.5)
            draws = sum(1 for r in results if abs(r) <= 0.5)
            losses = len(results) - wins - draws
            win_rate = wins / len(results)
            
            if win_rate > Config.PHASE1_5["win_threshold"]:
                self.current_elo += Config.PHASE1_5["elo_step"]
            elif win_rate < Config.PHASE1_5["win_threshold"] / 2:
                self.current_elo = max(Config.PHASE1_5["start_elo"], self.current_elo - Config.PHASE1_5["elo_step"] // 2)
            
            self.scheduler.step(win_rate)
            
            self.elo_prog.append(self.current_elo)
            self.win_rates.append(win_rate)
            
            print("="*100)
            print(f"Epoch {epoch}: ELO {self.current_elo}")
            print(f"Results: {wins}W {losses}L {draws}D | Win Rate: {win_rate:.1%}")
            print("Validation Results:")
            for level, wr in validation_results.items():
                print(f"  Level {level}: {wr:.1%}")
            print("="*100)
            
            self.save_checkpoint()
            self.save_progress()
            
            if win_rate < 0.1 and self.current_elo > Config.PHASE1_5["start_elo"] + 200:
                print("Performance dropped too much, stopping training")
                break
                
            epoch += 1
    
        self.save_checkpoint(is_final=True)
        self.engine.quit()
        self.writer.close()

    def save_progress(self):
        """Save training progress to CSV and plot"""

        with open("Graphs/stockfish/progress.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ELO", "WinRate", "Timestamp"])
            for e, wr in zip(self.elo_prog, self.win_rates):
                writer.writerow([e, wr, time.strftime("%Y-%m-%d %H:%M:%S")])
        

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(self.elo_prog)
        plt.title("ELO")
        plt.xlabel("Epoch")
        
        plt.subplot(1,2,2)
        plt.plot(self.win_rates)
        plt.title("Win Rate")
        plt.xlabel("Epoch")
        
        plt.savefig("Graphs/stockfish/progress.png")
        plt.close()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()