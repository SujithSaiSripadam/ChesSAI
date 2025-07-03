import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

#===========================================================
# ChessNet Model Definition 
#===========================================================
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.resblocks = nn.ModuleList([ResBlock(256) for _ in range(5)])
        self.policy_head = PolicyHead(256)
        self.value_head = ValueHead(256)
        
    def forward(self, x):
        x = x.to(Config.DEVICE).float()
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.resblocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

#===========================================================
# ResBlock Definition
#===========================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

#===========================================================
# Policy and Value Heads Definition
#===========================================================
class PolicyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.fc = nn.Linear(2 * 8 * 8, 4672)  #All possible chess moves

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2 * 8 * 8)
        return self.fc(x)

class ValueHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 8 * 8)
        x = F.relu(self.fc1(x))
        return self.tanh(self.fc2(x))