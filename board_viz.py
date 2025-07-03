import json
import chess
import os
##############################################################################
# A Part of self play game validation script (Phase 2)
# This script checks for illegal moves in self-play games stored in JSON files.
# It reads each game, validates moves against the FEN position, and counts results.
# It also summarizes the number of illegal moves and game outcomes.

base_path = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/selfplay/epoch26"
total_illegal_moves = 0
white_wins = 0
black_wins = 0  
draws = 0
no_illegal_moves = 0

for idx in range(0, 20):  #Loop from 1 to 20
    file_path = os.path.join(base_path, f"game_{idx}.json")
    print(f"\n=== Checking {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path) as f:
        game_data = json.load(f)

    illegal_moves = []

    for i, state in enumerate(game_data, 1):
        fen = state["fen"]
        move_uci = state["move"]
        result = state.get("result", None)

        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(move_uci)
        except Exception as e:
            print(f"\nMove {i}: Failed to parse move {move_uci} — {e}")
            illegal_moves.append((i, move_uci))
            continue

        print(f"\nMove {i}: {move_uci}")
        print(board)
        print("-" * 25)

        if result is not None:
            if result == 1.0:
                white_wins += 1
                print("Result: White won")
            elif result == 0.0:
                black_wins += 1
                print("Result: Black won")
            elif result == 0.5:
                draws += 1
                print("Result: Draw")
            else:
                print(f"Result: Unexpected value ({result})")

        if move in board.legal_moves:
            board.push(move)
            print(f"Move {i} is legal.")
        else:
            print(f"Move {i} is **illegal** in given FEN: {fen}")
            illegal_moves.append((i, move_uci))
            break  

    print(f"\nIllegal moves found at indices: {illegal_moves}")
    print(f"Total moves checked in game_{idx}.json: {len(game_data)}")
    print(f"Total illegal moves in game_{idx}.json: {len(illegal_moves)}")
    print(f"Legal moves: {len(game_data) - len(illegal_moves)}")

    total_illegal_moves += len(illegal_moves)

print(f"\n=== Summary ===")
print(f"Checked 40 games")
print(f"White wins: {white_wins}")
print(f"Black wins: {black_wins}")
print(f"Draws: {draws}")
print(f"Total illegal moves found across all games: {total_illegal_moves}")
