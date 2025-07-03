import pygame
import chess
import torch
import numpy as np
import torch.nn.functional as F
from src.model import ChessNet
from src.phase2_helpers import board_to_tensor, move_to_index
from stockfish import Stockfish
import time
pygame.init()

WIDTH, HEIGHT = 560, 560
SIDE_WIDTH = 360
WINDOW = (WIDTH + SIDE_WIDTH, HEIGHT)
SQUARE = WIDTH // 8
FONT_MAIN = pygame.font.SysFont("Arial", 33)
FONT_SIDE = pygame.font.SysFont("Arial", 22)
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (186, 202, 43)
PREV_HIGHL = (255, 182, 193, 150)
PIECE_COLOR = (10, 10, 10)

MODEL_PATH = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/Kaggle/better.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

stockfish = Stockfish(path="/opt/homebrew/bin/stockfish", parameters={"Threads": 3, "Skill Level": 10})

PIECE_LETTERS = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K'
}

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  
}

screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption("Model vs Stockfish Auto Play")

def count_captured(board):
    """Return dict of counts of captured pieces for white and black."""
    starting_counts = {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
        chess.KING: 1
    }


    counts = {
        chess.WHITE: {pt: 0 for pt in starting_counts},
        chess.BLACK: {pt: 0 for pt in starting_counts}
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            counts[piece.color][piece.piece_type] += 1

    #Captured = starting - current
    captured = {
        chess.WHITE: {pt: starting_counts[pt] - counts[chess.WHITE][pt] for pt in starting_counts},
        chess.BLACK: {pt: starting_counts[pt] - counts[chess.BLACK][pt] for pt in starting_counts}
    }
    return captured

def draw_board(board, selected=None, prev_move=None, model_eval=""):
    screen.fill(WHITE)

    #Draw squares and pieces
    for r in range(8):
        for f in range(8):
            sq = chess.square(f, 7 - r)
            rect = pygame.Rect(f * SQUARE, r * SQUARE, SQUARE, SQUARE)
            base = BROWN if (r + f) % 2 else WHITE
            pygame.draw.rect(screen, base, rect)

            if prev_move and (sq == prev_move.from_square or sq == prev_move.to_square):
                surf = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                surf.fill(PREV_HIGHL)
                screen.blit(surf, rect.topleft)

            if selected == sq:
                s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                s.fill(HIGHLIGHT)
                screen.blit(s, rect.topleft)

            piece = board.piece_at(sq)
            if piece:
                letter = PIECE_LETTERS[piece.piece_type]
                letter = letter.upper() if piece.color == chess.WHITE else letter.lower()
                text = FONT_MAIN.render(letter, True, PIECE_COLOR)
                screen.blit(text, text.get_rect(center=rect.center))

    #Side panel background
    side = pygame.Rect(WIDTH, 0, SIDE_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (230, 230, 230), side)

    #Stockfish top moves with safe centipawn division
    stockfish.set_fen_position(board.fen())
    top_moves = stockfish.get_top_moves(5)

    y = 20
    side_title = FONT_SIDE.render("Stockfish top moves:", True, (0, 0, 0))
    screen.blit(side_title, (WIDTH + 10, y))
    y += 30
    for mv in top_moves:
        centipawn = mv.get('Centipawn')
        if centipawn is None:
            eval_str = "N/A"
        else:
            eval_str = f"{centipawn / 100:.2f}"
        line = f"{mv['Move']}: {eval_str}"
        screen.blit(FONT_SIDE.render(line, True, (0, 0, 0)), (WIDTH + 10, y))
        y += 25

    y += 10
    model_eval_text = FONT_SIDE.render(f"Last move eval: {model_eval}", True, (0, 0, 0))
    screen.blit(model_eval_text, (WIDTH + 10, y))
    y += 40

    #Show captured pieces counts
    captured = count_captured(board)

    def captured_str(color):
        parts = []
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count = captured[color][pt]
            if count > 0:
                letter = PIECE_LETTERS[pt]
                letter = letter.upper() if color == chess.WHITE else letter.lower()
                parts.append(f"{letter}x{count}")
        return ", ".join(parts) if parts else "None"

    white_caps = captured_str(chess.WHITE)
    black_caps = captured_str(chess.BLACK)

    caps_title = FONT_SIDE.render("Captured pieces:", True, (0, 0, 0))
    screen.blit(caps_title, (WIDTH + 10, y))
    y += 25
    white_caps_text = FONT_SIDE.render(f"By Black (Stockfish Lv 10): {white_caps}", True, (0, 0, 0))
    screen.blit(white_caps_text, (WIDTH + 10, y))
    y += 25
    black_caps_text = FONT_SIDE.render(f"By White (My Model): {black_caps}", True, (0, 0, 0))
    screen.blit(black_caps_text, (WIDTH + 10, y))

    pygame.display.flip()

def load_model():
    model = ChessNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def get_model_move(model, board):
    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        logits, _ = model(t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    moves = list(board.legal_moves)
    p = np.array([probs[move_to_index(m)] for m in moves])
    p = np.clip(p, 1e-10, None)
    p /= p.sum()
    return np.random.choice(moves, p=p)

def promote_move(move, board):
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in [0, 7]:
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    return move

def main():
    board = chess.Board()
    model = load_model()
    running = True
    prev_move = None
    model_eval_str = ""

    clock = pygame.time.Clock()

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        if board.is_game_over():
            result = board.result()  # e.g. "1-0", "0-1", "1/2-1/2"
            print("Game Over:", result)

            if result == "1-0":
                print("Model (White) wins!")
            elif result == "0-1":
                print("Stockfish (Black) wins!")
            else:
                print("Draw!")
            
            pygame.time.wait(10000)
            running = False
            continue

        if board.turn == chess.WHITE:
            move = get_model_move(model, board)
            move = promote_move(move, board)
            board.push(move)
            prev_move = move

        else:
            stockfish.set_fen_position(board.fen())
            best_move_str = stockfish.get_best_move()
            if best_move_str is None:
                print("Stockfish no move available, game over or no moves")
                running = False
                continue
            move = chess.Move.from_uci(best_move_str)
            move = promote_move(move, board)
            board.push(move)
            prev_move = move

        stockfish.set_fen_position(board.fen())
        eval_score = stockfish.get_evaluation()
        if eval_score['type'] == 'cp':
            model_eval_str = f"{eval_score['value'] / 100:.2f}"
        elif eval_score['type'] == 'mate':
            model_eval_str = f"# {eval_score['value']}"
        else:
            model_eval_str = "N/A"

        draw_board(board, prev_move=prev_move, model_eval=model_eval_str)

        #clock.tick(1)  #1 move per second
        time.sleep(5) #5 seconds per move for better visualization else game would be completed in a few seconds

    pygame.quit()

if __name__ == "__main__":
    main()
