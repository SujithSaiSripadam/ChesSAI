# train2.5.py 
# diff: Cosine LR decay, mid-game position injection, TrueSkill Elo, adversarial self-play, and opening curriculum. --- Done....
import os, csv, time, random, chess, numpy as np, torch, torch.optim as optim, torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trueskill import Rating, rate_1vs1
from src.mcts import mcts_search, MCTSNode, ucb_score
from src.config import Config
from src.model import ChessNet
from src.utils import enforce_single_king
from src.phase2_helpers import evaluate_position, board_to_tensor, move_to_index
from src.openings import load_opening_fens  #opening FENs for mid-game injection -- Done
import chess.engine
import psutil
from torch.nn.utils.rnn import pad_sequence

# ------------------------------------------------------------------------
# Constants & Configs
# ------------------------------------------------------------------------
MAX_KL = 3
EARLY_STOP_PATIENCE = 50
INITIAL_ENTROPY = Config.PHASE1_5.get("entropy_coeff", 0.01)
FEN_INJECTION_PROB = Config.PHASE1_5.get("fen_injection_prob", 0.2)
OPENING_FENS = load_opening_fens()
# ------------------------------------------------------------------------
# PPO Trainer
# ------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self):
        self.writer = SummaryWriter(log_dir="runs/ppo_chess_" + time.strftime("%Y%m%d-%H%M%S"))
        self.global_step = 0
        self.no_improve_epochs = 0
        self.max_kl = 6

        #Skill tracking (TrueSkill)
        self.agent_rating = Rating()
        self.stock_rating = Rating()
        self.validation_levels = [1, 5, 10, 12, 15, 20]  
        self.win_rates = []


        #Model, optimizer, scheduler
        self.model = ChessNet().to(Config.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.PHASE1_5["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=Config.PHASE1_5.get("scheduler_T0", 10),
            T_mult=Config.PHASE1_5.get("scheduler_Tmult", 1),
        )
        self.global_step = 0

        #Load checkpoint
        self.current_elo = self._load_checkpoint()
        self.elo_prog = [self.current_elo]        #List to store ELO scores
        self.win_rates = [] 

        #Stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)

    def _load_checkpoint(self):
        try:
            ckpt = torch.load("/Users/sujithsaisripadam/Desktop/Cool_project/chessai/ckpt/best_30.pth", map_location=Config.DEVICE)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            return Config.PHASE1_5["start_elo"]
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return Config.PHASE1_5["start_elo"]
        
    def get_game_result(self, board, is_model_white):
        """Enhanced reward shaping with material and positional considerations"""
        if board.is_checkmate():
            return 1.0 if board.turn != (chess.WHITE if is_model_white else chess.BLACK) else -1.0
        
        #Draw conditions with small negative reward (discourage draws)
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return -0.1
        
        #Material advantage
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
        
        #Positional bonuses
        positional_bonus = 0
        if self.current_elo > 1000: 
            #Encourage central control
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            for sq in center_squares:
                piece = board.piece_at(sq)
                if piece and piece.color == (chess.WHITE if is_model_white else chess.BLACK):
                    positional_bonus += 0.1
        
        #Normalize and squash with tanh
        normalized_score = (score + positional_bonus) / 40.0  #Max possible ~39
        return np.tanh(normalized_score * 3)  #More sensitive to small advantages

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
                
    def inject_opening(self, board):
        if random.random() < FEN_INJECTION_PROB and OPENING_FENS:
            fen = random.choice(OPENING_FENS)
            board.set_fen(fen)
        else:
            board.reset()


    def play_game_ppo(self, board, level_config, is_model_white, tau):
        """Play game using PPO with MCTS enhancements"""
        game_history, value_history, log_prob_history = [], [], []
        self.simplify_position(board, self.current_elo)
        
        while not board.is_game_over() and len(board.move_stack) < Config.MAX_MOVES:
            if board.turn == (chess.WHITE if is_model_white else chess.BLACK):
                #Use MCTS for policy improvement at higher ELOs
                if self.current_elo > 1000:
                    move_probs = self.run_mcts(board, Config.PHASE1_5["mcts_sims"])
                    legal_moves = list(board.legal_moves)
                    probs = np.array([move_probs[move_to_index(m)] for m in legal_moves])
                else:
                    #Direct policy for lower ELOs
                    with torch.no_grad():
                        t = board_to_tensor(board).unsqueeze(0).to(Config.DEVICE)
                        policy_logits, value = self.model(t)
                        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    legal_moves = list(board.legal_moves)
                    legal_idx = [move_to_index(m) for m in legal_moves]
                    probs = policy[legal_idx]
                
                #Exploration strategy
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
                #Opponent move (Stockfish or random)
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
        lam = 0.95  #GAE parameter
        
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
            
            #Selection
            while node.children:
                node = max(node.children, key=lambda x: ucb_score(x))
            
            #Expansion
            if not node.board.is_game_over():
                legal_moves = list(node.board.legal_moves)
                
                #Get policy from model
                board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
                with torch.no_grad():
                    policy_logits, _ = self.model(board_tensor)
                policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                
                #Create child nodes
                for move in legal_moves:
                    child_board = node.board.copy()
                    child_board.push(move)
                    child = MCTSNode(child_board, node)
                    child.policy = policy[move_to_index(move)]
                    node.children.append(child)
            
            #Evaluation
            value = evaluate_position(node.board, self.model)
            
            #Backprop.....
            while node:
                node.visits += 1
                node.value += value
                node = node.parent
        
        #Return move probabilities
        move_probs = np.zeros(4672)  
        for child in root.children:
            move = child.board.peek()
            move_probs[move_to_index(move)] = child.visits
        
        if move_probs.sum() > 0:
            move_probs /= move_probs.sum()  
        
        return move_probs

    def update_model_ppo(self, game_data):
        if not game_data:
            return

        self.model.train()

        batch_size = Config.PHASE1_5.get('batch_size', 256)
        ppo_epochs = Config.PHASE1_5.get('ppo_epochs', 4)

        boards = torch.stack([board_to_tensor(d['board']) for d in game_data]).to(Config.DEVICE)
        old_logp = torch.tensor([d['log_prob'] for d in game_data], dtype=torch.float).to(Config.DEVICE)
        returns = torch.tensor([d['return'] for d in game_data], dtype=torch.float).to(Config.DEVICE)
        advantages = torch.tensor([d['advantage'] for d in game_data], dtype=torch.float).to(Config.DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_indexed = torch.tensor(
            [d['legal_indices'].index(move_to_index(d['action'])) for d in game_data],
            dtype=torch.long
        ).to(Config.DEVICE)

        for _ in range(ppo_epochs):
            perm = torch.randperm(len(game_data))

            for i in range(0, len(game_data), batch_size):
                idx = perm[i:i + batch_size]

                b = boards[idx]
                olp = old_logp[idx]
                rtn = returns[idx]
                adv = advantages[idx]
                aidx = action_indexed[idx].unsqueeze(1)

                policy_logits, value_preds = self.model(b)
                new_probs = F.softmax(policy_logits, dim=1)

                #Get per-sample legal indices for this batch
                legal_idx_batch = [torch.tensor(game_data[j]['legal_indices'], dtype=torch.long) for j in idx.tolist()]
                legal_idx_tensor = pad_sequence(legal_idx_batch, batch_first=True, padding_value=-1).to(Config.DEVICE)

                batch_probs = torch.gather(new_probs, 1, legal_idx_tensor)
                new_logp = torch.log(torch.gather(batch_probs, 1, aidx).squeeze(1) + 1e-10)
                ent = -(batch_probs * torch.log(batch_probs + 1e-10)).sum(dim=1).mean()

                #PPO objective
                ratio = torch.exp(new_logp - olp)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv    #clipping parameter --- 0.2 (Epsilon in my algo..)
                policy_loss = -torch.min(s1, s2).mean()

                value_preds = value_preds.squeeze(1)
                v_clipped = value_preds.detach() + (value_preds - value_preds.detach()).clamp(-0.5, 0.5)
                v_loss = torch.max(F.mse_loss(value_preds, rtn), F.mse_loss(v_clipped, rtn))

                entropy_coeff = INITIAL_ENTROPY * (1.0 - min(1.0, self.current_elo / Config.PHASE1_5["max_elo"]))
                total_loss = policy_loss + 0.5 * v_loss - entropy_coeff * ent

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.global_step += 1

                #Logging
                self.writer.add_scalar("Loss/policy", policy_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/value", v_loss.item(), self.global_step)
                self.writer.add_scalar("Loss/entropy", ent.item(), self.global_step)
                self.writer.add_scalar("Stats/entropy_coeff", entropy_coeff, self.global_step)

                kl = (olp - new_logp).mean()

                if kl > self.max_kl:
                    #print(f"Early stopping PPO epochs due to KL {kl:.4f}")
                    return
                
            self.max_kl = max(0.1, self.max_kl * 0.98)
            self.writer.add_scalar("Stats/max_KL", self.max_kl, self.global_step)


    def validate_model(self):
        """Validate against multiple Stockfish levels"""
        validation_results = {}
        for level in self.validation_levels:
            if level > self.current_elo + 300:  #Don't validate against much stronger opponents
                continue
                
            wins, losses, draws = 0, 0, 0
            for _ in range(Config.EVAL_GAMES // 2):  #Half games as white, half as black
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
        
        #Also save latest
        torch.save(checkpoint, "models/ELO1.5/latest.pth")


    def save_progress(self):
        """Save training progress to CSV and plot"""
        #CSV
        with open("Graphs/stockfish/progress.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ELO", "WinRate", "Timestamp"])
            for e, wr in zip(self.elo_prog, self.win_rates):
                writer.writerow([e, wr, time.strftime("%Y-%m-%d %H:%M:%S")])
        
        #Plot
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

    def train(self):
        epoch = 0
        best_win = 0

        while self.current_elo <= Config.PHASE1_5["max_elo"]:
            level = Config.STOCKFISH_LEVELS.get(self.current_elo, {"time":0.1,"depth":1})
            tau = max(0.5, 2.0 - (self.current_elo / Config.PHASE1_5["max_elo"]) * 1.5)

            game_data, results = [], []

            for game_idx in tqdm(range(Config.PHASE1_5["games_per_level"])):
                board = chess.Board()
                self.inject_opening(board)  #mid-game or opening injection

                is_white = (game_idx % 2 == 0)
                res, hist = self.play_game_ppo(board, level, is_white, tau)
                game_data.extend(hist)
                results.append(res)

                if (game_idx+1) % max(5, Config.PHASE1_5["update_freq"]) == 0:
                    self.update_model_ppo(game_data)
                    game_data = []

            win_rate = sum(r > 0.5 for r in results) / len(results)
            validation_wr = self.validate_model()
            avg_wr = sum(validation_wr.values()) / len(validation_wr)
            self.win_rates.append(avg_wr) 
            self.elo_prog.append(self.current_elo)         #or the new elo computed


            #TrueSkill rating update
            for res in results:
                if res > 0.5:
                    self.agent_rating, self.stock_rating = rate_1vs1(self.agent_rating, self.stock_rating)
                elif res < -0.5:
                    self.stock_rating, self.agent_rating = rate_1vs1(self.stock_rating, self.agent_rating)

            self.scheduler.step()

            #Early stop check
            if validation_wr[1] > best_win + 0.01:
                best_win = validation_wr[1]
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= EARLY_STOP_PATIENCE:
                print("Early stopping due to no validation progress")
                break

            epoch += 1
            self.save_checkpoint()
            self.save_progress()
            time.sleep(1)

        self.save_checkpoint(is_final=True)
        self.engine.quit()
        self.writer.close()

if __name__ == "__main__":
    print("[INFO] Starting PPO trainer...")
    trainer = PPOTrainer()
    print("[INFO] Initialized. Beginning training...")
    trainer.train()
    print("[INFO] Training complete.")
