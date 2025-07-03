import pygame   # check if pygame is installed, at the time of writing, I'm using a different Conda env. Make sure to add pygame in req.txt or env.yaml
import chess
import torch
import numpy as np
import torch.nn.functional as F
from src.model import ChessNet
from src.phase2_helpers import board_to_tensor, move_to_index
from stockfish import Stockfish

pygame.init()

#UI constants
WIDTH, HEIGHT = 560, 560
SIDE_WIDTH = 240
WINDOW = (WIDTH + SIDE_WIDTH, HEIGHT)
SQUARE = WIDTH // 8
FONT_MAIN = pygame.font.SysFont("Arial", 33)
FONT_SIDE = pygame.font.SysFont("Arial", 22)
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (186, 202, 43)
PREV_HIGHL = (255, 182, 193, 150)
PIECE_COLOR = (10, 10, 10)

#Paths & device
MODEL_PATH = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/ckpt/best_30.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Load Stockfish (homebrew)
stockfish = Stockfish(path="/opt/homebrew/bin/stockfish", parameters={"Threads": 2, "Skill Level": 20})

PIECE_LETTERS = {
    chess.PAWN: 'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK: 'R',
    chess.QUEEN: 'Q',
    chess.KING: 'K'
}

screen = pygame.display.set_mode(WINDOW)
pygame.display.set_caption("Chess vs Your Model")

def draw_board(board, selected=None, prev_move=None, model_eval=None):
    screen.fill(WHITE)

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

    side = pygame.Rect(WIDTH, 0, SIDE_WIDTH, HEIGHT)
    pygame.draw.rect(screen, (230, 230, 230), side)
    stockfish.set_fen_position(board.fen())
    top_moves = stockfish.get_top_moves(5)

    y = 20
    side_title = FONT_SIDE.render("Stockfish evals:", True, (0, 0, 0))
    screen.blit(side_title, (WIDTH + 10, y))
    y += 30
    for mv in top_moves:
        line = f"{mv['Move']}: {mv['Centipawn'] / 100:.2f}"
        screen.blit(FONT_SIDE.render(line, True, (0, 0, 0)), (WIDTH + 10, y))
        y += 25

    if model_eval is not None:
        y += 15
        model_eval_text = FONT_SIDE.render(f"Model move eval: {model_eval}", True, (0, 0, 0))
        screen.blit(model_eval_text, (WIDTH + 10, y))

    pygame.display.flip()

def load_model():
    model = ChessNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
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
    selected = None
    prev_move = None
    model_move_eval = ""
    human_color = chess.WHITE

    draw_board(board, selected=selected, prev_move=prev_move, model_eval=model_move_eval)

    while running:
        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(3000)
            break

        #AI's turn
        if board.turn != human_color:
            move = get_model_move(model, board)
            move = promote_move(move, board)
            board.push(move)
            prev_move = move

            stockfish.set_fen_position(board.fen())
            eval_score = stockfish.get_evaluation()
            if eval_score['type'] == 'cp':
                model_move_eval = f"{eval_score['value'] / 100:.2f}"
            elif eval_score['type'] == 'mate':
                model_move_eval = f"# {eval_score['value']}"
            else:
                model_move_eval = "N/A"

            draw_board(board, selected=selected, prev_move=prev_move, model_eval=model_move_eval)
            continue

        #Player turn
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                x, y = ev.pos
                f = x // SQUARE
                r = 7 - (y // SQUARE)
                sq = chess.square(f, r)

                if selected is None:
                    if (piece := board.piece_at(sq)) and piece.color == human_color:
                        selected = sq
                        draw_board(board, selected=selected, prev_move=prev_move, model_eval=model_move_eval)
                else:
                    mv = chess.Move(selected, sq)
                    mv = promote_move(mv, board)
                    if mv in board.legal_moves:
                        board.push(mv)
                        prev_move = mv
                        selected = None
                        draw_board(board, selected=selected, prev_move=prev_move, model_eval=model_move_eval)
                    else:
                        selected = None
                        draw_board(board, selected=selected, prev_move=prev_move, model_eval=model_move_eval)

        pygame.time.wait(10)     # adjust delay for smooth UI : ) else itll play at a blink speed

    pygame.quit()

if __name__ == "__main__":
    main()
