import chess
#######################################################################################
# Better change the logic of removing duplicate king comparing to previous states...... 
########################################################################################
def enforce_single_king(board: chess.Board) -> chess.Board:
    """Ensure the board has exactly one king per side"""
    #Make a copy to avoid modifying the original
    corrected_board = board.copy(stack=False)
    
    #Count kings
    white_kings = list(corrected_board.pieces(chess.KING, chess.WHITE))
    black_kings = list(corrected_board.pieces(chess.KING, chess.BLACK))
    
    #Remove extra white kings (keep the first one found)
    for sq in white_kings[1:]:
        corrected_board.remove_piece_at(sq)
    
    #Remove extra black kings (keep the first one found)
    for sq in black_kings[1:]:
        corrected_board.remove_piece_at(sq)
    
    #If missing kings, restore standard positions
    if not white_kings:
        corrected_board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    if not black_kings:
        corrected_board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    
    return corrected_board



















"""import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
input_csv = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/processed/labels.csv"
train_csv = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/processed/train_labels.csv"
val_csv = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/processed/val_labels.csv"

# Load full dataset
df = pd.read_csv(input_csv)

# Split 90% train, 10% validation
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# Save
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print(f"✅ Saved {len(train_df)} train rows and {len(val_df)} val rows.")
"""