import os
import json
import numpy as np
from tqdm import tqdm
from src.mcts import board_to_tensor, evaluate_position
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataloader import ChessDataset
from src.model import ChessNet
from src.mcts import mcts_search
from src.config import Config
import torch.nn.functional as F
import chess
import chess.engine
from src.utils import enforce_single_king
import glob
import random
from src.mcts import MCTSNode, ucb_score

# ELO calculation utilities
def expected_score(rating_a, rating_b):
    """Calculate expected score between two ELO ratings"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k_factor=32):
    """Update ELO ratings after a game"""
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    return new_rating_a


def eval_against_stockfish(model, level=1, games=Config.EVAL_GAMES, current_elo=None):
    """Evaluate model and return ELO estimate"""

    try:
        engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)  ######ADDED try block for robustness
        engine.configure({"Skill Level": level})
    except Exception as e:
        print(f"[ERROR] Failed to launch Stockfish: {e}")  ######BETTER error handling
        return current_elo or 800

    #Approximate Stockfish ELO mapping
    stockfish_elos = {
        1: 800,    # Level 1
        5: 1200,
        10: 1800,
        15: 2500,
        20: 3200
    }
    opponent_elo = stockfish_elos.get(level, 800)

    results = {"wins": 0, "losses": 0, "draws": 0}

    for game_num in range(games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:  #AI's turn
                move = mcts_search(board, model, Config.PHASE2["sims_per_move"])
            else:  #Stockfish's turn
                try:
                    result = engine.play(board, chess.engine.Limit(time=Config.STOCKFISH_TIME))  ######Replace hardcoded time (Done..... )
                    move = result.move
                except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
                    print(f"[WARNING] Stockfish crashed during game {game_num + 1}. Restarting engine...") 
                    try:
                        engine.quit()
                    except:
                        pass
                    engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)  ######Restart engine
                    engine.configure({"Skill Level": level})
                    move = random.choice(list(board.legal_moves))  ######Fallback

            board.push(move)

        #Record result
        if board.is_checkmate():
            if board.turn == chess.BLACK:
                results["wins"] += 1
            else:
                results["losses"] += 1
        else:
            results["draws"] += 1

    try:
        engine.quit()  ######Ensure engine is quit gracefully  (!! Not closed at the time of training --not sure check at the time of pushing to GitHub)
    except:
        pass

    #Calculate ELO
    total_games = results["wins"] + results["losses"] + results["draws"]
    if total_games == 0:
        return current_elo or 800

    win_rate = results["wins"] / total_games
    draw_rate = results["draws"] / total_games
    score = results["wins"] + 0.5 * results["draws"]
    performance_rating = opponent_elo + 400 * ((score / total_games) - 0.5)

    if current_elo is not None:
        expected = expected_score(current_elo, opponent_elo)
        actual = score / total_games
        new_elo = current_elo + 32 * (actual - expected)
    else:
        new_elo = performance_rating

    print("\n" + "="*60)
    print(f"Evaluation vs Stockfish Level {level} (ELO ~{opponent_elo}):")
    print(f"Results: {results['wins']} Wins, {results['losses']} Losses, {results['draws']} Draws")
    print(f"Win Rate: {win_rate*100:.1f}% | Draw Rate: {draw_rate*100:.1f}%")
    print(f"Performance Rating: {performance_rating:.0f}")
    if current_elo is not None:
        print(f"ELO Change: {new_elo - current_elo:+.0f}")
    print(f"New ELO Rating: {new_elo:.0f}")
    print("="*60 + "\n")

    return new_elo


def load_selfplay_games(output_dir, max_games=None):
    """Load self-play games with error handling"""
    game_files = sorted([
        f for f in os.listdir(output_dir) 
        if f.endswith('.json') and os.path.isfile(os.path.join(output_dir, f))
    ])
    
    if max_games is not None:
        game_files = game_files[-max_games:]  #Get most recent games
    
    loaded_games = []
    
    for filename in game_files:
        filepath = os.path.join(output_dir, filename)
        try:
            #Try UTF-8 first    {Working fine with utf-8 encoding..... , the error initially was due to the file being empty}
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            #Validate and convert data
            validated_data = []
            for entry in data:
                try:
                    #Ensure all required fields exist
                    if not all(k in entry for k in ["fen", "move", "policy", "result"]):
                        raise ValueError("Missing required fields")
                    
                    validated_data.append({
                        "fen": str(entry["fen"]),
                        "move": chess.Move.from_uci(entry["move"]) if isinstance(entry["move"], str) else entry["move"],
                        "policy": {int(k): float(v) for k, v in entry["policy"].items()},
                        "result": float(entry["result"])
                    })
                except Exception as e:
                    print(f"Invalid entry in {filename}: {str(e)}")
                    continue
            
            loaded_games.extend(validated_data)
            
        except UnicodeDecodeError:
            try:
                #Fallback to latin-1 if UTF-8 fails
                with open(filepath, 'r', encoding='latin-1') as f:
                    data = json.load(f)
                loaded_games.extend(data)
            except Exception as e:
                print(f"Could not read {filename} even with latin-1: {str(e)}")
                continue
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return loaded_games

# Initial version of board_to_tensor function which was commented out due to requirement issues ( to comply, some fn's demand tensor output, some demand numpy output)
"""
def board_to_tensor(board: chess.Board) -> np.ndarray:
    #Convert chess board to 8x8x12 numpy array
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece.piece_type - 1 + (6 if piece.color == chess.WHITE else 0)
            row, col = chess.square_rank(square), chess.square_file(square)
            tensor[channel, row, col] = 1
    return tensor
"""


def board_to_tensor(board: chess.Board, return_numpy: bool = False):
    """Convert chess board to tensor with option for numpy output"""
    if return_numpy:
        array = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.WHITE else 0)
                row, col = chess.square_rank(square), chess.square_file(square)
                array[channel, row, col] = 1
        return array
    else:
        tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.WHITE else 0)
                row, col = chess.square_rank(square), chess.square_file(square)
                tensor[channel, row, col] = 1
        return tensor.to(Config.DEVICE)  
    
def save_selfplay_game(game_data, output_dir):
    """Save self-play games to JSON with proper data types"""
    os.makedirs(output_dir, exist_ok=True)
    game_id = len(os.listdir(output_dir))
    
    #Convert Move objects to UCI strings
    serializable_data = []
    for entry in game_data:
        serializable_data.append({
            "fen": entry["fen"],
            "move": entry["move"].uci() if isinstance(entry["move"], chess.Move) else entry["move"],
            "policy": entry["policy"],  #Already in correct format
            "result": float(entry["result"])
        })
    
    #Write with explicit UTF-8 encoding
    with open(f"{output_dir}/game_{game_id}.json", 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        
# This is a duplicate of the move_to_index function in mcts.py 
def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move to a unique index (0–4671).
    - Normal moves: from_square * 64 + to_square → 0–4095
    - Promotion moves: 4096–4671
    """
    if not move.promotion:
        return move.from_square * 64 + move.to_square

    #Promotion move encoding
    promo_offset = {
        chess.QUEEN: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3
    }

    from_sq = move.from_square
    file = chess.square_file(move.to_square)

    from_sq = move.from_square
    to_file = chess.square_file(move.to_square)  # 0-7 (a-h)
    promo_type = promo_offset[move.promotion]

    #Promotion index = 4096 + (from_square % 8) * 32 + (promo_type * 8) + to_file
    return 4096 + (from_sq % 8) * 32 + promo_type * 8 + to_file


"""def run_mcts(model, board, simulations):
   
    root_node = mcts_search(board, model, simulations, return_root=True)
    move_probs = np.zeros(4672)  # Total possible moves
    
    if not hasattr(root_node, 'children') or not root_node.children:
        # If we didn't get proper nodes, return uniform distribution
        legal_moves = list(board.legal_moves)
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
        for move in legal_moves:
            move_probs[move_to_index(move)] = uniform_prob
        return move_probs
    
    total_visits = sum(child.visits for child in root_node.children)
    
    for child in root_node.children:
        try:
            move = child.board.peek()
            move_idx = move_to_index(move)
            move_probs[move_idx] = child.visits / total_visits if total_visits > 0 else 0
        except (IndexError, AttributeError):
            continue
    
    # Normalize to ensure valid probability distribution
    if move_probs.sum() > 0:
        move_probs /= move_probs.sum()
    else:
        # Fallback to uniform if all zeros
        legal_moves = list(board.legal_moves)
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
        for move in legal_moves:
            move_probs[move_to_index(move)] = uniform_prob
    
    return move_probs"""

def run_mcts(board: chess.Board, model: ChessNet, simulations: int) -> np.ndarray:
    """Run MCTS and return move probabilities"""
    root = MCTSNode(board.copy)
    
    for _ in range(simulations):
        node = root
        
        #Selection
        while node.children:
            node = max(node.children, key=lambda x: ucb_score(x))
        
        #Expansion
        if not node.board.is_game_over():
            #Get legal moves from the BOARD, not the model
            legal_moves = list(node.board.legal_moves)
            
            #Get policy from model
            board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                policy_logits, _ = model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            
            #Create child nodes
            for move in legal_moves:
                child_board = node.board.copy()
                child_board.push(move)
                child = MCTSNode(child_board, node)
                child.policy = policy[move_to_index(move)]
                node.children.append(child)
        
        #Simulation
        value = evaluate_position(node.board, model)
        
        #Backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    #Return move probabilities
    move_probs = np.zeros(4672)  #Total possible moves
    for child in root.children:
        move = child.board.peek()
        move_probs[move_to_index(move)] = child.visits
    
    if move_probs.sum() > 0:
        move_probs /= move_probs.sum()  #Normalize..
    
    return move_probs

def sample_move_from_probs(board, move_probs, temperature=0.7):
    """Sample move from probability distribution"""
    legal_moves = []
    legal_probs = []
    
    for move in board.legal_moves:
        #Simulate the move to check king count
        test_board = board.copy(stack=False)
        test_board.push(move)
        white_kings = len(list(test_board.pieces(chess.KING, chess.WHITE)))
        black_kings = len(list(test_board.pieces(chess.KING, chess.BLACK)))
        
        if white_kings == 1 and black_kings == 1:
            legal_moves.append(move)
            legal_probs.append(move_probs[move_to_index(move)])
    
    if not legal_moves:
        return chess.Move.null()  #Fallback
    #Apply temperature
    probs = np.array(legal_probs) ** (1/temperature)
    probs /= probs.sum()
    
    return np.random.choice(legal_moves, p=probs)

def get_game_result(board):
    """Get game result as numeric value"""
    result = board.result()
    if result == "1-0":
        #print("Won")
        return 1.0
    elif result == "0-1":
        return -1.0
    return 0.0

def generate_selfplay_game(model, sims_per_move):
    """Generate self-play game with consistent data structure"""
    board = chess.Board()
    game_history = []
    
    while not board.is_game_over():
        board = enforce_single_king(board)
        #Get MCTS policy as numpy array
        move_probs = run_mcts(model, board, sims_per_move)
        #Sample move with temperature
        move = sample_move_from_probs(board, move_probs, Config.PHASE2["temperature"])
        
        #Convert policy to proper format
        policy_dict = {
            str(move_to_index(m)): float(move_probs[move_to_index(m)])
            for m in board.legal_moves
            if move_probs[move_to_index(m)] > 0  #Only include moves with probability > 0
        }
        game_history.append({
            "fen": board.fen(),
            "move": move,
            "policy": policy_dict,
            "result": None  #Will be filled later
        })
        board.push(move)
    #Assign final result to all positions
    result = get_game_result(board)
    for entry in game_history:
        entry["result"] = result
    
    return game_history

def train_on_selfplay_data(model, optimizer):
    game_files = sorted(glob.glob(os.path.join(Config.PHASE2["output_dir"], "*.json")))
    
    for game_file in game_files:
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
                
            for position in game_data:
                #Convert all data to tensors
                board_tensor = torch.FloatTensor(board_to_tensor(chess.Board(position['fen'])))
                move_target = torch.LongTensor([move_to_index(chess.Move.from_uci(position['move']))])
                value_target = torch.FloatTensor([position['result']])
                
                #Convert policy from dict to tensor
                policy_target = torch.zeros(4672)
                for idx, prob in position['policy'].items():
                    policy_target[int(idx)] = prob
                
                #Move to device
                board_tensor = board_tensor.to(Config.DEVICE)
                move_target = move_target.to(Config.DEVICE)
                value_target = value_target.to(Config.DEVICE)
                policy_target = policy_target.to(Config.DEVICE)
                
                #Training step
                optimizer.zero_grad()
                policy_pred, value_pred = model(board_tensor.unsqueeze(0))  #Add batch dimension 
                
                policy_loss = F.cross_entropy(policy_pred, move_target)
                value_loss = F.mse_loss(value_pred, value_target)
                total_loss = policy_loss + 0.4 * value_loss
                
                total_loss.backward()
                optimizer.step()
                
        except Exception as e:
            print(f"Error processing {game_file}: {str(e)}")
            continue

def generate_stockfish_game(model, engine, stockfish_mix):
    """Generate game with Stockfish moves blended in"""
    board = chess.Board()
    game_history = []
    
    while not board.is_game_over():
        if np.random.rand() < stockfish_mix:
            #Use Stockfish move
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
        else:
            #Use model move
            move_probs = run_mcts(model, board, Config.PHASE3["sims_per_move"])
            move = sample_move_from_probs(board, move_probs)
        
        game_history.append((board.fen(), move_probs, None))
        board.push(move)
    
    result = get_game_result(board)
    return [(fen, probs, result) for fen, probs, _ in game_history]
