# ♟️ ChesSAI: A Hybrid Reinforcement Learning-Based Self-Evolving Chess Engine

A chess engine combining **PPO** and **MCTS**, trained via self-play, imitation learning, and dynamic ELO-based curriculum.

---

## 🚀 Overview

This repository implements a self-evolving chess AI that blends **Proximal Policy Optimization (PPO)** and **Monte Carlo Tree Search (MCTS)** into a unified hybrid architecture. Inspired by AlphaZero, our system introduces:

- Progressive training
- Curriculum-based self-play
- Reward shaping
- Dynamic policy mixing based on ELO

---

## 🧠 Core Components

- ✅ Supervised Pretraining using expert human games  
- ♻️ Self-Play Reinforcement Learning with MCTS-guided policy improvement  
- 🤖 Adversarial PPO Training against Stockfish  
- 🔁 **Hybrid Action Selection**:  
  \[
  \pi_{\text{play}} = \alpha(\text{ELO}) \cdot \pi_{\text{PPO}} + (1 - \alpha(\text{ELO})) \cdot \pi_{\text{MCTS}}
  \]
- 🌲 MCTS with Neural Guidance using UCB + policy priors  
- 🧭 Reward Shaping based on positional central control  

---

## 📂 Repository Structure

```
.
├── data/                # Preprocessed, self-play, and backup data
├── models/              # Saved checkpoints across all training phases
├── src/                 # All training, model, and MCTS source code
├── play_vs_model.py     # GUI for human vs model play
├── modelvsstockfish.py  # GUI for Stockfish vs model play
├── environment.yaml     # Conda environment file
├── requirements.txt     # Python requirements
├── Chessai.pdf          # Detailed LaTeX report
```

---

## 🏗️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/SujithSaiSripadam/ChesSAI.git
cd chessai
```

2. **Set up environment (Conda recommended)**  
```bash
conda env create -f environment.yaml
conda activate chessai
```

3. **Install Stockfish Engine**  
Download from: https://stockfishchess.org/download/

For MAC 🍎 users:

```bash
brew install stockfish
```

Add it to path
```bash
mv path/to/stockfish /usr/local/bin/
chmod +x /usr/local/bin/stockfish
```

For Linux 🐧 Users:
```bash
sudo apt install stockfish
```

To check stockfish:
```bash
which stockfish
```

---

## 🧪 Running the Code

### ✅ Supervised Pretraining (Phase 1)
```bash
python -m src.train --phase 1
```

### ♻️ Self-Play Training (Phase 2)
```bash
python -m src.train --phase 2
```

### 🤖 PPO vs Stockfish Training (Phase 2.5)
```bash
python -m src.train_with_stockfish_V2
```

---

## 📊 Evaluation and Logging

TensorBoard logs are saved under:
```
./runs/PPO_...
```

To view logs:
```bash
tensorboard --logdir ./runs
```

---

## 📚 Algorithms Explained

Full technical explanation is available in [`Chessai.pdf`](Chessai.pdf), covering:

- PPO loss derivation  
- MCTS pseudocode  
- Network architecture  
- Phase-wise training pipeline  

---

## 👨‍💻 Author

**Sripadam Sujith Sai**  
[GitHub]((https://github.com/SujithSaiSripadam/ChesSAI)) • [LinkedIn](https://www.linkedin.com/in/sripadam-sujith-sai/)   
_Evolving AI with algorithms and chess._

---

## 📃 License

This project is open-sourced under the MIT License.
