###################################################
# run trainvsstockfish_V2.py for Phase 2.5 training
################################333################
import argparse
import torch
from src.train import train_phase1, train_phase2
from src.config import Config
from src.model import ChessNet 
from src.train_1_5 import train_against_stockfish_ppo
import os
if __name__ == "__main__":
    os.makedirs("Graphs/Phase1", exist_ok=True)
    os.makedirs("Graphs/Phase2", exist_ok=True)
    os.makedirs("Graphs/stockfish", exist_ok=True)
    parser = argparse.ArgumentParser(description="Train Chess AI Phases")
    parser.add_argument('--phase', type=int, choices=[1, 5, 2, 3], required=True,
                        help='Training phase to run: 1 = Imitation, 2 = Self-play, 3 = High-ELO')
    args = parser.parse_args()

    if args.phase == 1:
        print("=== PHASE 1: IMITATION LEARNING ===")
        train_phase1()
        
    elif args.phase == 5:
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            print("=== PHASE 1.5: Imitation Learning with additional data ===")
            train_against_stockfish_ppo()
        else:
            print("!!!!!!!!!  GPU or MPS not available for Phase 1.5")

    elif args.phase == 2:
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            print("=== PHASE 2: SELF-PLAY RL ===")
            train_phase2()
        else:
            print("!!!!!!!!!! GPU or MPS not available for Phase 2")

    elif args.phase == 3:
        if torch.cuda.is_available():
            print("=== PHASE 3: HIGH-ELO TRAINING ===")
            print("!!!!!!!!!!!  Phase 3 is currently incomplete and requires CUDA (NVIDIA GPU) for training.")
            #Call train_phase3()           #  ------------------------ Incomplete function -------------------------------
        else:
            print("!!!!!!!!!!!! Phase 3 requires CUDA (NVIDIA GPU)")

