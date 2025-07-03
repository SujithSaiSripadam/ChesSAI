import torch

class Config:
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Make sure Stockfish is installed at this loc. -- if any issue check [$ which stockfish]


    MAX_MOVES = 500                             #Preventing excessively long games
    
    EVAL_GAMES = 10                             #Number of evaluation games
    STOCKFISH_TIME = 0.1                        #seconds per move for stockfish
    STOCKFISH_LEVELS = {
        200: {"depth": 1, "time": 0.1},
        800: {"depth": 2, "time": 0.3},
        1000: {"depth": 3, "time": 0.4},
        1200: {"depth": 4, "time": 0.5},
        1400: {"depth": 5, "time": 0.5},
        1500: {"depth": 6, "time": 0.6},
        1700: {"depth": 7, "time": 0.7},
        1800: {"depth": 8, "time": 0.8},
        1900: {"depth": 9, "time": 0.9},
        2000: {"depth": 10, "time": 1.0},
        2400: {"depth": 15, "time": 2.0},
        2500: {"depth": 20, "time": 3.0}
    }

    
    #Phase 1: Imitation Learning
    PHASE1 = {
        "epochs": 30,
        "batch_size": 256,
        "lr": 1e-4,
        "train_data": "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/processed/train_labels.csv",
        "val_data": "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/data/processed/val_labels.csv"
    }   
    
    #Phase 2: Self-Play RL ( Self combat for -- Exploration and exploitation ) -- Critical for Coming up with new startegies)
    PHASE2 = {
        "selfplay_games": 1000,
        "sims_per_move": 20,
        "output_dir": "data/selfplay/epoch26",
        "batch_size": 512,
        "lr": 1e-4,
        "temperature": 0.7,  #For exploration vs exploitation
        "eval_frequency": 20 #Evaluate every N games --- Elo Rank update 
    }
    
    
    #Phase 2.5: Training with Stockfish at different ELO levels 
    PHASE1_5 = { 
        "start_elo": 400,
        "max_elo": 2700,
        "games_per_level": 100,
        "update_freq": 10,
        "win_threshold": 0.6,
        "elo_step": 100,
        "gamma": 0.99,
        "mcts_sims": 200,
        "lr": 1e-5
    }                                          
    
    #Phase 3: High-ELO upper Rank Training
    PHASE3 = {
        "epochs": 50,
        "games_per_epoch": 200,  # 100 self-play + 100 vs Stockfish
        "sims_per_move": 800,
        "stockfish_mix": 0.1,  # 10% Stockfish moves
        "batch_size": 256,
        "lr": 1e-5,
        "stockfish_path": "/opt/homebrew/bin/stockfish"  # Update this path
    }