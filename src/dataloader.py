import pandas as pd
import chess
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import torch

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert chess board to 8x8x12 tensor"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = piece.piece_type - 1 + (6 if piece.color == chess.WHITE else 0)
            row, col = chess.square_rank(square), chess.square_file(square)
            tensor[channel, row, col] = 1
    return tensor

class ChessDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        self.move_to_idx = self._build_move_index()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        fen = self.data.iloc[idx]['fen']
        move = self.data.iloc[idx]['move']
        eval_score = float(self.data.iloc[idx]['eval'])
        
        board = chess.Board(fen)
        board_tensor = board_to_tensor(board)
        move_idx = self._move_to_index(chess.Move.from_uci(move))
        
        return (
            torch.FloatTensor(board_tensor),
            torch.LongTensor([move_idx]),
            torch.FloatTensor([eval_score])
        )

    def _build_move_index(self) -> Dict[chess.Move, int]:
        """Build mapping of all possible chess moves to indices (0-4671)"""
        index = 0
        move_map = {}
        
        # Generate all possible moves
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                move = chess.Move(from_sq, to_sq)
                if move in move_map:
                    continue
                
                # Handle promotions
                if chess.square_rank(from_sq) in [1, 6] and chess.square_rank(to_sq) in [0, 7]:
                    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                        move_map[promo_move] = index
                        index += 1
                else:
                    move_map[move] = index
                    index += 1
        
        return move_map

    def _move_to_index(self, move: chess.Move) -> int:
        """Convert chess.Move to unique index (0-4671)"""
        return self.move_to_idx.get(move, 0)  # Default to 0 if move not found