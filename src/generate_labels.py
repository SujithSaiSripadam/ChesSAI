import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import signal
from contextlib import contextmanager
from datetime import datetime
import random

# --- Config ---
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
INPUT_PGN = "/data/raw/games.pgn"
OUTPUT_CSV = "/data/processed/labels.csv"
OUTPUT_CSV2 = "/data/processed/labels2.csv"
OUTPUT_CSV3 = "/data/processed/labels3.csv"

#  ---- Redundants ---
STOCKFISH_DEPTH = 10
MAX_CPU_TEMP = 85  #°C - throttle if exceeds this
CHECK_INTERVAL = 30  #seconds between temp checks
COOLDOWN_INTERVAL = 60  #seconds to pause if overheating
THROTTLE_FACTOR = 0.5  #Reduce workload by this factor when hot

# --- Adjust based on Requirements ---
NUM_WORKERS = max(1, cpu_count() - 1)
CHUNK_SIZE = 2000

#Training phase targets
PHASE_1_GAMES = 30000    #Imitation learning phase
PHASE_2_GAMES = 10000     #Self-play refinement phase
PHASE_3_GAMES = 5000      #High-ELO competition phase
MIN_ELO = 1600            #Minimum ELO to consider
MAX_ELO = 3000            #Maximum ELO to consider

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

@contextmanager
def pool_context(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()

def should_process_game(game, phase):
    """Determine if we should process this game based on phase and ELO"""
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        avg_elo = (white_elo + black_elo) / 2
        
        if phase == 1:  #Imitation learning - broad range
            return MIN_ELO <= avg_elo <= MAX_ELO and random.random() < 0.3  #Sample 30%
        elif phase == 2:  #Self-play refinement - mid range
            return 2000 <= avg_elo <= 2300 and random.random() < 0.5
        elif phase == 3:  #High-ELO competition
            return avg_elo >= 2200
        return False
    except (ValueError, TypeError):
        return False

def analyze_game(game):
    """Analyze a single game and return important positions"""
    positions = []
    board = game.board()
    
    #Skip first 5 moves (opening theory)
    mainline_moves = list(game.mainline_moves())
    if len(mainline_moves) < 10:  #Skip very short games
        return positions
    
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            for i, move in enumerate(mainline_moves):
                #Only analyze every 3rd move after move 5 to get diverse positions
                if i >= 5 and i % 3 == 0:
                    try:
                        info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                        eval_score = info["score"].white().score(mate_score=1000)
                        if eval_score is not None:
                            eval_score /= 100.0
                            positions.append([board.fen(), move.uci(), eval_score])
                    except Exception as e:
                        log(f"Skipping move due to error: {e}")
                board.push(move)
    except Exception as e:
        log(f"Engine error: {e}")
    return positions

def process_games(pgn_path, output_path, target_games, phase):
    """Process games with ELO filtering for specific phase"""
    games_processed = 0
    positions_collected = 0
    header_written = False
    
    with open(pgn_path) as pgn, \
         open(output_path, "a") as csv_file, \
         tqdm(total=target_games, desc=f"Processing Phase {phase} games") as pbar:

        while games_processed < target_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            if should_process_game(game, phase):
                positions = analyze_game(game)
                if positions:
                    df = pd.DataFrame(positions, columns=["fen", "move", "eval"])
                    df.to_csv(csv_file, mode="a", header=not header_written, index=False)
                    header_written = True
                    positions_collected += len(positions)
                    games_processed += 1
                    pbar.update(1)
                    
                    # Log progress every 100 games
                    #if games_processed % 100 == 0:
                        #log(f"Phase {phase}: Processed {games_processed}/{target_games} games, collected {positions_collected} positions")
    
    log(f"✅ Phase {phase} complete. Processed {games_processed} games, collected {positions_collected} positions")
    return positions_collected

def main():
    #Phase 1: Imitation learning (broad ELO range)
    log("Starting Phase 1 (Imitation Learning)")
    phase1_positions = process_games(INPUT_PGN, OUTPUT_CSV, PHASE_1_GAMES, 1)
    
    #Phase 2: Self-play refinement (mid ELO range)
    log("Starting Phase 2 (Self-play Refinement)")
    phase2_positions = process_games(INPUT_PGN, OUTPUT_CSV2, PHASE_2_GAMES, 2)
    
    #Phase 3: High-ELO competition
    log("Starting Phase 3 (High-ELO Competition)")
    phase3_positions = process_games(INPUT_PGN, OUTPUT_CSV3, PHASE_3_GAMES, 3)
    
    total_positions = phase1_positions + phase2_positions + phase3_positions
    log(f"🎉 All phases complete! Total positions collected: {total_positions}")

if __name__ == "__main__":
    main()