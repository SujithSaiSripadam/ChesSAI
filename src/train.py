import os
import json
import numpy as np
from tqdm import tqdm
from src.mcts import board_to_tensor, evaluate_position
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataloader import ChessDataset
from src.model import ChessNet
from src.mcts import mcts_search
from src.config import Config
import torch.nn.functional as F
import chess
from src.utils import enforce_single_king
import glob
import random
from src.phase2_helpers import  update_elo, eval_against_stockfish, load_selfplay_games, board_to_tensor, move_to_index, run_mcts, sample_move_from_probs, get_game_result, generate_selfplay_game, save_selfplay_game, train_on_selfplay_data, generate_stockfish_game
import csv
import matplotlib.pyplot as plt 
import time

def train_phase1():
    model = ChessNet().to(Config.DEVICE)
    loss_log = []
    #load checkpoint
    checkpoint_path = "/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/phase1_epoch9.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
        start_epoch = 10 
    else:
        start_epoch = 0   
    
    optimizer = optim.Adam(model.parameters(), lr=Config.PHASE1["lr"])
    train_dataset = ChessDataset(Config.PHASE1["train_data"])
    train_loader = DataLoader(train_dataset, batch_size=Config.PHASE1["batch_size"], shuffle=True)
    
    for epoch in range(start_epoch, Config.PHASE1["epochs"]):
        model.train()
        epoch_policy_loss = 0
        epoch_value_loss = 0
        epoch_total_loss = 0
        batch_count = 0
        print(f"Running epoch: {epoch}\n")
        for batch_idx, batch in enumerate(train_loader):
            board_tensor, move_target, eval_target = batch
            board_tensor = board_tensor.to(Config.DEVICE)
            move_target = move_target.to(Config.DEVICE)
            eval_target = eval_target.to(Config.DEVICE)
            
            optimizer.zero_grad()
            policy_pred, value_pred = model(board_tensor)
            
            policy_loss = F.cross_entropy(policy_pred, move_target.squeeze())
            value_loss = F.mse_loss(value_pred, eval_target)
            total_loss = policy_loss + 0.2 * value_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                print(f"Total loss: {total_loss} ; Policy Loss: {policy_loss} ; value_loss: {value_loss}\n")    
                 
        avg_policy = epoch_policy_loss / batch_count
        avg_value = epoch_value_loss / batch_count
        avg_total = epoch_total_loss / batch_count
        loss_log.append([epoch, avg_policy, avg_value, avg_total])
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/phase1_epoch{epoch}.pth")
        
        with open("Graphs/Phase1/loss_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Policy Loss", "Value Loss", "Total Loss"])
            writer.writerows(loss_log)

        #Plotss
        epochs = [row[0] for row in loss_log]
        policy = [row[1] for row in loss_log]
        value = [row[2] for row in loss_log]
        total = [row[3] for row in loss_log]

        plt.figure()
        plt.plot(epochs, policy, label="Policy Loss")
        plt.plot(epochs, value, label="Value Loss")
        plt.plot(epochs, total, label="Total Loss")
        plt.title("Phase 1 Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("Graphs/Phase1/loss_plot.png")
        plt.close()
            
        
        
def train_phase2():
    model = ChessNet().to(Config.DEVICE)
    checkpoint = torch.load("models/phase1_epoch24.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=Config.PHASE2["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])   
    model = ChessNet().to(Config.DEVICE)
    current_elo = 600  #Starting ELO... 
    best_elo = current_elo
    
    for game_idx in range(Config.PHASE2["selfplay_games"]):
        start_time = time.time()
        print(f"\nGame {game_idx + 1}/{Config.PHASE2['selfplay_games']}")
        #1. Generate and save game
        game_data = generate_selfplay_game(model, Config.PHASE2["sims_per_move"])
        save_selfplay_game(game_data, Config.PHASE2["output_dir"])
        #2. Train every 10 games
        if (game_idx + 1) % 5 == 0:
            print("Training on recent games...")
            train_on_selfplay_data(model, optimizer)
            #Save checkpoint
            torch.save(model.state_dict(), f"models/phase_2/phase2_checkpoint_{game_idx}.pth")
            #3. Evaluate ELO every 50 games (adjust frequency as needed ---- more frequent for lower ELO models would be better)
            if (game_idx + 1) % 5 == 0:
                print("\nEvaluating current model strength...")
                if current_elo < 800:
                    #Use lower Stockfish level for lower ELO models
                    current_elo = eval_against_stockfish(
                        model, 
                        level=1,  #Adjust level based on expected strength
                        current_elo=current_elo
                    )
                elif current_elo < 1000:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=2,  #Low level for beginner ELO
                        current_elo=current_elo
                    )
                elif current_elo < 1100:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=3,  #Low level for beginner ELO
                        current_elo=current_elo
                    )
                elif  current_elo < 1200:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=4,  #Medium level for mid-range ELO
                        current_elo=current_elo
                    )
                elif current_elo < 1300:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=5,  #Medium level for mid-range ELO
                        current_elo=current_elo
                    )
                elif current_elo < 1400:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=6,  #Medium level for mid-range ELO
                        current_elo=current_elo
                    )
                elif current_elo < 1450:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=7,  #Medium level for mid-range ELO
                        current_elo=current_elo
                    )      
                elif current_elo < 1500:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=8,  #Higher level for higher ELO
                        current_elo=current_elo
                    )
                    
                elif current_elo < 1800:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=10,  #Higher level for higher ELO
                        current_elo=current_elo
                    )
                elif current_elo < 2000:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=12,  #Higher level for higher ELO
                        current_elo=current_elo
                    )
                elif current_elo < 2500:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=15,  #Higher level for higher ELO
                        current_elo=current_elo
                    )
            
                else:
                    current_elo = eval_against_stockfish(
                        model, 
                        level=20,  #Top level for high ELO models
                        current_elo=current_elo
                    )

                #Saving best model
                torch.save(model.state_dict(), "models/phase_2/ELO/phase2_{current_elo}.pth")
                if current_elo > best_elo:
                    best_elo = current_elo
                    torch.save(model.state_dict(), "models/phase_2/ELO/phase2_{current_elo}.pth")
                    print(f"New best model saved (ELO: {best_elo:.0f})")
                print(f"Current ELO: {current_elo:.0f} | Best ELO: {best_elo:.0f}")
                

        print(f"Game {game_idx + 1} completed in {time.time() - start_time:.2f} seconds")
    #Final evaluation and save.....
    print("\nFinal evaluation...")
    final_elo = eval_against_stockfish(model, level=20, current_elo=current_elo)
    torch.save(model.state_dict(), "models/phase_2/phase2_final.pth")
    print(f"\nPhase 2 complete! Final ELO: {final_elo:.0f}")
            

def train_phase3():
    """High-ELO training against Stockfish and self-play"""
    model = ChessNet().to(Config.DEVICE)
    model.load_state_dict(torch.load("/Users/sujithsaisripadam/Desktop/Cool_project/chessai/models/phase2_final.pth"))  #Load Phase 2 model
    
    #Initialize Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)
    
    #Training loop
    for epoch in range(Config.PHASE3["epochs"]):
        print(f"Epoch {epoch + 1}/{Config.PHASE3['epochs']}")
        game_data = []
        
        #Generate games (50% self-play, 50% vs Stockfish)
        for _ in tqdm(range(Config.PHASE3["games_per_epoch"] // 2)):
            #Self-play games
            self_play_data = generate_selfplay_game(model, Config.PHASE3["sims_per_move"])
            game_data.extend(self_play_data)
            
            #Vs Stockfish games
            stockfish_data = generate_stockfish_game(model, engine, Config.PHASE3["stockfish_mix"])
            game_data.extend(stockfish_data)
        
        #Train on collected data
        train_on_phase3_data(model, game_data)
        
        #Save checkpoint
        os.makedirs("/models/phase3", exist_ok=True)
        torch.save(model.state_dict(), f"/models/phase3/phase3_epoch{epoch}.pth")
    
    
######################################################################################
# Phase 3 Incomplete Functions
# These functions are placeholders and need to be implemented.
# They are not fully functional and require additional logic to work correctly....
# They are provided here to maintain the structure of the code and indicate where
# further development is needed.
#####################################################################################3
def create_phase3_batches(game_data, batch_size):
    " create batches..."
def train_on_phase3_data(model, game_data):
    """Train model on Phase 3 data"""
    optimizer = optim.Adam(model.parameters(), lr=Config.PHASE3["lr"])
    
    for batch in create_phase3_batches(game_data, Config.PHASE3["batch_size"]):
        board_tensors, target_policies, target_values = batch
        
        optimizer.zero_grad()
        policy_pred, value_pred = model(board_tensors)
        
        #Policy loss (KL divergence)
        policy_loss = F.kl_div(
            F.log_softmax(policy_pred, dim=1),
            target_policies,
            reduction='batchmean'
        )
        
        #Value loss (MSE)
        value_loss = F.mse_loss(value_pred, target_values)
        
        total_loss = policy_loss + value_loss
        total_loss.backward()
        optimizer.step()