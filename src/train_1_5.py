import torch
import numpy as np
from tqdm import tqdm
import chess
import chess.engine
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from src.model import ChessNet
from src.config import Config
from src.phase2_helpers import board_to_tensor, move_to_index
from src.utils import enforce_single_king
import torch.nn.functional as F

import os
import csv
import time

def get_game_result_new(board, is_model_white):
    """
    Returns result from model's perspective
    1.0 = model won
    -1.0 = model lost
    0.0 to 0.5 = draw (depending on game length)
    """
    if board.is_checkmate():
        #If model is white and checkmate is on black's turn (white won)
        #or model is black and checkmate is on white's turn (black won)
        model_won = (is_model_white and board.turn == chess.BLACK) or \
                   (not is_model_white and board.turn == chess.WHITE)
        return 1.0 if model_won else -1.0
    
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        #slight positive for longer games
        return 0.2 if len(board.move_stack) > 30 else 0.0
    #Intermediate rewards
    piece_value = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = piece_value.get(piece.piece_type, 0)
            #Add if piece is model's color, subtract if opponent's
            score += val if piece.color == (chess.WHITE if is_model_white else chess.BLACK) else -val
    return np.tanh(score * 0.1)  #Scaled to [-1, 1]

def plot_progress(elo_progression, win_rates):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(elo_progression)
    plt.title("ELO Progression")
    plt.xlabel("Training Stage")
    plt.ylabel("ELO Rating")
    
    plt.subplot(1, 2, 2)
    plt.plot(win_rates)
    plt.title("Win Rate Progression")
    plt.xlabel("Training Stage")
    plt.ylabel("Win Rate")
    
    plt.tight_layout()
    plt.savefig("Graphs/stockfish/phase1.5_progress.png")
    plt.close()
    
def play_stockfish_vs_model_ppo(board, model, engine, level_config, current_elo):
    """Play game where Stockfish is White vs model (Black) for PPO"""
    game_history = []
    value_history = []
    log_prob_history = []
    
    while not board.is_game_over() and len(board.move_stack) < Config.MAX_MOVES:
        if board.turn == chess.WHITE:  #Stockfish's turn
            result = engine.play(board, chess.engine.Limit(
                time=level_config["time"],
                depth=level_config["depth"]))
            move = result.move
        else:  #Model's turn
            #Get model's policy and value
            with torch.no_grad():
                board_tensor = board_to_tensor(board).unsqueeze(0).to(Config.DEVICE)
                policy, value = model(board_tensor)
                policy = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
            
            #Filter legal moves
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(move) for move in legal_moves]
            legal_probs = policy[legal_indices]
            legal_probs = np.clip(legal_probs, 1e-8, None)
            legal_probs /= legal_probs.sum()  # Renormalize
            
            #Sample move
            move_idx = np.random.choice(len(legal_moves), p=legal_probs)
            move = legal_moves[move_idx]
            
            #Store data for PPO
            log_prob = np.log(legal_probs[move_idx] + 1e-10)
            log_prob_history.append(log_prob)
            value_history.append(value)
            
            game_history.append({
                'board': board.copy(),
                'action': move,
                'log_prob': log_prob,
                'value': value,
                'legal_moves': legal_moves,
                'legal_indices': legal_indices
            })
        
        board.push(move)
    
    result = get_game_result_new(board,is_model_white=False)
    save_game_history(game_history, result, current_elo)
    
    #save in PGN format
    #save_pgn(game_history, result, current_elo)
    
    #Calculate returns (from model's perspective)
    returns = []
    discounted_return = result 
    for value in reversed(value_history): #Invert result since model is Black
        discounted_return = Config.PHASE1_5["gamma"] * discounted_return
        returns.insert(0, discounted_return)
    
    #Add returns to game history
    for i, data in enumerate(game_history):
        if 'value' in data:
            data['return'] = returns[i]
            data['advantage'] = returns[i] - data['value']
    
    return result, game_history  #Return result from model's perspective

def play_model_vs_stockfish_ppo(board, model, engine, level_config, current_elo):
    """Play game collecting data for PPO with NaN protection"""
    game_history = []
    value_history = []
    log_prob_history = []
    
    while not board.is_game_over() and len(board.move_stack) < Config.MAX_MOVES:
        if board.turn == chess.WHITE:  #Model's turn
            with torch.no_grad():
                board_tensor = board_to_tensor(board).unsqueeze(0).to(Config.DEVICE)
                policy, value = model(board_tensor)
                policy = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                value = value.item()
            
            #Filter legal moves with NaN protection
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(move) for move in legal_moves]
            legal_probs = policy[legal_indices]
            
            #Add small epsilon and renormalize to prevent NaN
            legal_probs = legal_probs + 1e-10
            legal_probs = legal_probs / legal_probs.sum()
            
            #Additional check for invalid probabilities
            if np.any(np.isnan(legal_probs)):
                print("Warning: NaN detected in probabilities, using uniform distribution")
                legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
            
            #Sample move with validation
            try:
                move_idx = np.random.choice(len(legal_moves), p=legal_probs)
                move = legal_moves[move_idx]
            except ValueError:
                print("Invalid probability distribution, using random move")
                move_idx = np.random.randint(len(legal_moves))
                move = legal_moves[move_idx]
            
            #Store data
            log_prob = np.log(legal_probs[move_idx] + 1e-10)
            log_prob_history.append(log_prob)
            value_history.append(value)
            
            game_history.append({
                'board': board.copy(),
                'action': move,
                'log_prob': log_prob,
                'value': value,
                'legal_moves': legal_moves,
                'legal_indices': legal_indices
            })
        else:  #Stockfish's turn
            result = engine.play(board, chess.engine.Limit(
                time=level_config["time"], 
                depth=level_config["depth"]))
            move = result.move
        
        board.push(move)
    
    result = get_game_result_new(board,is_model_white=True)
    save_game_history(game_history, result, current_elo)
    
    #save in PGN format
    #save_pgn(game_history, result, current_elo)
    
    #Calculate returns
    returns = []
    discounted_return = result
    for value in reversed(value_history):
        discounted_return = Config.PHASE1_5["gamma"] * discounted_return
        returns.insert(0, discounted_return)
    
    #Add returns to game history
    for i, data in enumerate(game_history):
        if 'value' in data:
            data['return'] = returns[i]
            data['advantage'] = returns[i] - data['value']
    
    return result, game_history

def train_against_stockfish_ppo():
    """Phase 1.5: Progressive training against Stockfish using PPO"""
    os.makedirs("game_states", exist_ok=True)
    #Clear previous runs
    open("game_states/final_positions.txt", "w").close()
    open("game_states/winning_moves.txt", "w").close()
    model = ChessNet().to(Config.DEVICE)
    checkpoint = torch.load("models/Kaggle/final.pth", map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=Config.PHASE1_5["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])   
    engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)
    
    #ELO progression tracking
    elo_progression = []
    win_rates = []
    current_elo = Config.PHASE1_5["start_elo"]
    
    try:
        #Initial performance check
        initial_win_rate = check_model_performance(model, engine, current_elo)
        print(f"Initial win rate at ELO {current_elo}: {initial_win_rate:.1%}")
        
        while current_elo <= Config.PHASE1_5["max_elo"]:
            level = Config.STOCKFISH_LEVELS[current_elo]
            print(f"\nTraining at ELO {current_elo} (Depth: {level['depth']}, Time: {level['time']}s), Stockfish Level: {level}")
            
            game_data = []
            results = []
            
            for game_idx in tqdm(range(Config.PHASE1_5["games_per_level"]), desc="Games"):
                board = chess.Board()
                if game_idx % 2 == 0:
                    result, history = play_model_vs_stockfish_ppo(board, model, engine, level, current_elo)
                else:
                    result, history = play_stockfish_vs_model_ppo(board, model, engine, level, current_elo)
                
                results.append(result)
                game_data.extend(history)
                
                #Update more frequently at lower levels
                update_freq = max(5, Config.PHASE1_5["update_freq"] - current_elo//200)
                if (game_idx + 1) % update_freq == 0 and game_data:
                    update_model_ppo(model, optimizer, game_data)
                    game_data = []
            
            #win_rate = np.mean([r == 1.0 for r in results])
            #Result counts
            wins = sum(1 for r in results if r > 0.5)
            draws = sum(1 for r in results if r == 0.0)
            losses = len(results) - wins - draws
            win_rate = wins / len(results)
            print("===========================================================================================================================================")
            print(results)
            print("===========================================================================================================================================")
            print(f"Games played: {len(results)}, Wins: {wins}, Losses: {losses}, Draws: {draws}, Win rate: {win_rate:.1%}")


            win_rates.append(win_rate)
            
            #Adaptive ELO progression
            if win_rate > Config.PHASE1_5["win_threshold"]:
                new_elo = current_elo + Config.PHASE1_5["elo_step"]
                print(f"Advancing from ELO {current_elo} to {new_elo} (Win rate: {win_rate:.1%})")
            else:
                #If losing badly, consider reducing ELO
                if win_rate < Config.PHASE1_5["win_threshold"]/2:
                    new_elo = max(Config.PHASE1_5["start_elo"], current_elo - Config.PHASE1_5["elo_step"]//2)
                    print(f"Reducing ELO from {current_elo} to {new_elo} (Win rate: {win_rate:.1%})")
                else:
                    new_elo = current_elo
                    print(f"Maintaining ELO {current_elo} (Win rate: {win_rate:.1%})")
            
            current_elo = new_elo
            elo_progression.append(current_elo)
            
            #Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'elo': current_elo,
                'win_rate': win_rate
            }, f"models/ELO1.5/phase1.5_elo_{current_elo}.pth")
            
            save_progress(elo_progression, win_rates)
            plot_progress(elo_progression, win_rates)
            
            #Early exit if performance is too poor
            if win_rate < 0.1 and current_elo > Config.PHASE1_5["start_elo"] + 200:
                print("Performance too poor - stopping training")
                break
    
    finally:
        engine.quit()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'elo': current_elo,
            'win_rates': win_rates
        }, "models/ELO1.5/phase1.5_final.pth")
        plot_progress(elo_progression, win_rates)
    
def update_model_ppo(model, optimizer, game_data, clip_epsilon=0.2, ppo_epochs=3, value_coeff=0.5, entropy_coeff=0.01):
    """More robust PPO update with additional checks"""
    if not game_data or len(game_data) == 0:
        return
    
    model.train()
    
    try:
        #Convert game data to tensors with validation
        board_tensors = torch.stack([board_to_tensor(d['board']) for d in game_data]).to(Config.DEVICE)
        old_log_probs = torch.tensor([d['log_prob'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        returns = torch.tensor([d['return'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        advantages = torch.tensor([d['advantage'] for d in game_data], dtype=torch.float32).to(Config.DEVICE)
        actions = [d['action'] for d in game_data]
        legal_moves_list = [d['legal_moves'] for d in game_data]
        legal_indices_list = [d['legal_indices'] for d in game_data]
        
        #Validate tensors
        if torch.isnan(board_tensors).any() or torch.isnan(old_log_probs).any() or torch.isnan(returns).any():
            print("Warning: NaN detected in input tensors, skipping update")
            return
        
        #Normalize advantages with protection
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(ppo_epochs):
            policy_logits, value_preds = model(board_tensors)
            
            #Validate model outputs
            if torch.isnan(policy_logits).any() or torch.isnan(value_preds).any():
                print("Warning: NaN in model outputs, skipping update")
                break
                
            new_probs = F.softmax(policy_logits, dim=1)
            new_log_probs = []
            entropies = []
            
            for i in range(len(game_data)):
                legal_indices = legal_indices_list[i]
                legal_probs = new_probs[i, legal_indices]
                
                legal_probs = legal_probs / (legal_probs.sum() + 1e-10)  #Safer normalization
                
                #Fallback to uniform if probabilities are invalid
                if torch.isnan(legal_probs).any() or (legal_probs <= 0).all():
                    legal_probs = torch.ones_like(legal_probs) / len(legal_probs)
                
                action_idx = legal_moves_list[i].index(actions[i])
                new_log_probs.append(torch.log(legal_probs[action_idx] + 1e-10))
                entropy = -torch.sum(legal_probs * torch.log(legal_probs + 1e-10))
                entropies.append(entropy)
            
            new_log_probs = torch.stack(new_log_probs)
            entropies = torch.stack(entropies)
            
            #Calculate ratios with protection
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios = torch.clamp(ratios, 1e-5, 1e5)  # Prevent extreme values
            
            #PPO objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            #Value loss with clipping
            value_loss = F.mse_loss(value_preds.squeeze(), returns)
            
            #Entropy bonus
            entropy_loss = -entropies.mean()
            
            #Total loss
            total_loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
            
            #Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
    except Exception as e:
        print(f"Error during PPO update: {str(e)}")
        traceback.print_exc()

def save_progress(elo_progression, win_rates, filename="Graphs/stockfish/phase1.5_progress.csv"):
    """Save training progress to CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ELO", "WinRate", "Timestamp"])
        for elo, wr in zip(elo_progression, win_rates):
            writer.writerow([elo, wr, time.strftime("%Y-%m-%d %H:%M:%S")])
            
def check_model_performance(model, engine, elo_level, num_games=10):
    """Diagnostic function to check model performance"""
    wins = 0
    for _ in range(num_games):
        board = chess.Board()
        result, _ = play_model_vs_stockfish_ppo(board, model, engine, 
                                               Config.STOCKFISH_LEVELS[elo_level], 
                                               elo_level)
        wins += int(result > 0.5)
    return wins / num_games

def save_board_state(board, filename="game_states/final_positions.txt"):
    """Save final board state to file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        f.write(f"FEN: {board.fen()}\n")
        f.write(f"Result: {board.result()}\n")
        f.write(f"{board}\n\n")

def save_winning_moves(game_history, filename="game_states/winning_moves.txt"):
    """Save last 2 moves of winning games"""
    if len(game_history) < 2:
        return
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        last_move = game_history[-1]['action'].uci()
        second_last_move = game_history[-2]['action'].uci()
        f.write(f"Winning moves: {second_last_move}, {last_move}\n")
        f.write(f"Final FEN: {game_history[-1]['board'].fen()}\n\n")


def save_game_history(game_history, result, elo_level, save_dir="game_states"):
    """Save complete game information including FEN sequence"""
    os.makedirs(save_dir, exist_ok=True)
    
    if result > 0.5:
        outcome = "win"
    elif result < -0.5:
        outcome = "loss"
    else:
        outcome = "draw"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/game_{timestamp}_{outcome}.txt"
    
    with open(filename, "w") as f:
        f.write(f"ELO Level: {elo_level}\n")
        f.write(f"Result: {outcome}\n")
        f.write(f"Total Moves: {len(game_history)}\n\n")
        
        f.write("FEN Sequence:\n")
        for i, h in enumerate(game_history):
            if 'board' in h:
                f.write(f"Move {i+1}: {h['board'].fen()}\n")
        
        f.write("\nMove List:\n")
        for i, h in enumerate(game_history):
            if 'action' in h:
                f.write(f"Move {i+1}: {h['action'].uci()}\n")
        
        f.write("\nFinal Position:\n")
        if game_history and 'board' in game_history[-1]:
            board = game_history[-1]['board']
            f.write(f"FEN: {board.fen()}\n")
            f.write(str(board))

