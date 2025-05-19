from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import numpy as np
import math
from isibc_targetid import TargetIdentifier
from concurrent.futures import ThreadPoolExecutor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_preprocess(data: np.ndarray, tag) -> np.ndarray:
    """数据预处理函数"""
    # 归一化处理
    data = np.concatenate((np.real(data), np.imag(data)), axis=0)  # 将实部和虚部拼接
    data = (data - np.mean(data)) / np.std(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    tag_tensor = torch.tensor(tag, dtype=torch.float32).to(device)
    return data_tensor, tag_tensor

class fusion_net(nn.Module):
    def __init__(self, K, N, channel: int = 2, batch_size: int = 2, output_size: int = 8) -> None:
        super(fusion_net,self).__init__()
        self.theta_fc1 = nn.Linear(1, 64)
        self.theta_fc2 = nn.Linear(64, 128)
        self.mat_conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        self.mat_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.mat_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fuse_fc1 = nn.Linear(128+32*K*N/64, 2*(128+32*K*N/64))
        self.fuse_fc2 = nn.Linear(128+32*K*N/64, 128)
        self.fuse_fc3 = nn.Linear(128, output_size)
    
    def forward(self, data: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        self.theta_output = F.relu(self.theta_fc1(theta))
        self.theta_output = F.relu(self.theta_fc2(self.theta_output))
        self.mat_output = F.relu(self.mat_conv1(data))
        self.mat_output = self.pooling(self.mat_output)
        self.mat_output = F.relu(self.mat_conv2(self.mat_output))
        self.mat_output = self.pooling(self.mat_output)
        self.mat_output = F.relu(self.mat_conv3(self.mat_output))
        self.mat_output = self.pooling(self.mat_output)
        self.fuse_output = torch.cat((self.mat_output, self.theta_output), dim=1)
        self.fuse_output = F.relu(self.fuse_fc1(self.fuse_output))
        self.fuse_output = F.relu(self.fuse_fc2(self.fuse_output))
        self.fuse_output = F.softmax(self.fuse_fc3(self.fuse_output))
        return self.fuse_output
    
    def loss(input,tag):
        cross_entropy = nn.CrossEntropyLoss(input, tag)
        return cross_entropy

        
        