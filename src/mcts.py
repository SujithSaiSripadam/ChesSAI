import chess
import numpy as np
import torch
import torch.nn.functional as F
from src.model import ChessNet
from src.config import Config
from typing import List, Dict, Optional
import numpy as np
import random
import os
import json
import copy
from src.utils import enforce_single_king

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
    #print(f"from_sq: {from_sq}, to_file: {to_file}, promo_type: {promo_type}")
    return 4096 + (from_sq % 8) * 32 + promo_type * 8 + to_file

def get_game_state_key(board: chess.Board) -> str:
    """Create a unique key for repetition detection"""
    return f"{board.board_fen()}_{board.turn}_{board.castling_rights}_{board.ep_square}"

def is_threefold_repetition(game_history: list) -> bool:
    """Check if current position has appeared 3 times"""
    state_counts = {}
    for state_key in game_history:
        state_counts[state_key] = state_counts.get(state_key, 0) + 1
        if state_counts[state_key] >= 3:
            return True
    return False

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

def evaluate_position(board: chess.Board, model: ChessNet) -> float:
    """Evaluate position using neural network"""
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
    
    with torch.no_grad():
        board_tensor = torch.FloatTensor(board_to_tensor(board)).unsqueeze(0).to(Config.DEVICE) 
        _, value = model(board_tensor)
        return value.item()

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.board = enforce_single_king(board)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.policy = None
        
def mcts_search(board: chess.Board, model: ChessNet, simulations: int = 800, return_root: bool = False):
    #root = MCTSNode(board.copy())
    root = MCTSNode(board.copy(stack=False))
    game_history = [get_game_state_key(root.board)]
    
    for _ in range(simulations):
        node = root
        current_history = game_history.copy()
        #Selection from root to leaf ---> Only select children with legal moves 
        while node.children:
            node = max(node.children, key=lambda x: ucb_score(x))
            current_history.append(get_game_state_key(node.board))
            
            if is_threefold_repetition(current_history) or node.board.halfmove_clock >= 50:
                break
    
        #Expansion
        if not node.board.is_game_over() and not is_threefold_repetition(current_history):
            board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
            policy_logits, _ = model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()
            
            for move in node.board.legal_moves:
                child_board = node.board.copy(stack=False)
                child_board.push(move)
                child = MCTSNode(child_board, node)
                child.policy = policy[move_to_index(move)]
                node.children.append(child)
                
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board_to_tensor(node.board)).unsqueeze(0).to(Config.DEVICE)
            policy_logits, value = model(board_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        #Simulation ---- > playout!!
        value = evaluate_position(node.board, model)
    
        #Backprop.......
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    #Return handling
    if return_root:
        return root
    else:
        if not root.children:
            return random.choice(list(root.board.legal_moves)) if root.board.legal_moves else None
        
        best_child = max(root.children, key=lambda x: x.visits)
        
        #Safely get the last move
        if best_child.board.move_stack:
            return best_child.board.peek()
        else:
            #Reconstruct the move that led to this position
            if best_child.parent and best_child.parent.board.move_stack:
                return best_child.parent.board.peek()
            return random.choice(list(root.board.legal_moves))
          

def causes_immediate_draw(board: chess.Board, game_history: list) -> bool:
    """Check if move would cause instant draw"""
    new_history = game_history + [get_game_state_key(board)]
    return (
        is_threefold_repetition(new_history) or 
        board.halfmove_clock >= 50 or
        board.is_fifty_moves() or 
        board.is_repetition(3)
    )

def ucb_score(node: MCTSNode, c: float = 1.4) -> float:
    """UCB1 formula for node selection"""
    if node.visits == 0:
        return float('inf')
    exploitation = node.value / node.visits      #W(s,a)/N(s,a) -- Refer Paper section 2.5
    exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits) # The second term there ( c_puc. P(s,a). (sq.root(N(s))/(1+N(s,a)))) -- Refer Paper section 2.5
    return exploitation + exploration

