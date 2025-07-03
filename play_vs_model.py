###########################################################################
# To play via terminal --- very hard to play so better use gui file created
###########################################################################
import torch
import chess
import chess.svg
import numpy as np
import torch.nn.functional as F
from src.model import ChessNet
from src.phase2_helpers import board_to_tensor, move_to_index
import sys

MODEL_PATH = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/ckpt/best_30.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    model = ChessNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def get_model_move(model, board):
    with torch.no_grad():
        tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        policy_logits, _ = model(tensor)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    move_probs = np.array([policy[move_to_index(m)] for m in legal_moves])
    
    move_probs = np.clip(move_probs, 1e-10, None)
    move_probs /= move_probs.sum()
    
    chosen_move = np.random.choice(legal_moves, p=move_probs)
    return chosen_move

def main():
    model = load_model()
    board = chess.Board()

    print("Lets play Suji! ; )")
    side = input("Play as white or black? (w/b): ").strip().lower()
    human_color = chess.WHITE if side == "w" else chess.BLACK

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == human_color:
            move_uci = input("Your move SriSuSa (UCI format, e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move dude checkit!")
            except:
                print("Invalid input. Try again.")
        else:
            print("Wait let me thiiinkkk...")
            move = get_model_move(model, board)
            print(f"model plays: {move.uci()}")
            board.push(move)

    print("\nGame Over.")
    print("Result:", board.result())

if __name__ == "__main__":
    main()
