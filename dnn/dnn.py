from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import numpy as np
import math
from isibc_targetid import TargetIdentifier
from concurrent.futures import ThreadPoolExecutor

def data_preprocess(data: np.ndarray, real_value: dict,K,M,N) -> np.ndarray:
    """数据预处理函数"""
    # 归一化处理
    data = data.reshape(K,N,M)
    data = np.transpose(data, (2, 0, 1))
    data = np.concatenate((np.real(data), np.imag(data)), axis=0)  # 将实部和虚部拼接
    data = (data - np.mean(data)) / np.std(data)
    angles_est = list(real_value.values())
    idx = [i for i,val in enumerate(angles_est) if (np.array(val).size == 0)]
    angles_est = np.array([[0,0] if i in idx else [1,float(val[0])] for i, val in enumerate(angles_est)])
    return data, angles_est

def process(K, M, N, snr):
    ti = TargetIdentifier(K, M, N, snr)
    num_sources = np.random.randint(1, 5)
    theta_true = np.random.uniform(-0.4, 0.4, num_sources)
    theta_env_true = np.random.uniform(-0.4, 0.4, 1)
    data, true_value = ti.generate_signal(theta_true, theta_env_true)
    data_preprocessed, angle = data_preprocess(data, true_value, K, M, N)
    return data_preprocessed, angle

@torch.no_grad()
def estimation_loss(model, n_test, loss_fn, device, K, M, N, batch_size,n_thread):
    out = []
    loss = torch.zeros(n_test)
    model.eval()
    snr = np.random.uniform(0, 20)
    for k in range(n_test):
        data_batch = []
        tag_batch = []
        with ThreadPoolExecutor(max_workers=n_thread) as executor:  # 设置线程数，例如 8
            futures = [executor.submit(process, K, M, N, snr) for _ in range(batch_size)]
            for future in futures:
                data_preprocessed, angle = future.result()
                data_batch.append(data_preprocessed)
                tag_batch.append(angle)
# 将结果转换为 numpy 数组
        data_batch = np.array(data_batch)
        tag_batch = np.array(tag_batch)
        data_tensor = torch.tensor(data_batch, dtype=torch.float32).to(device)
        tag_tensor = torch.tensor(tag_batch, dtype=torch.float32).to(device)
        output = model(data_tensor)
        loss[k] = loss_fn(output, tag_tensor)
    out.append(loss.mean().item())
    model.train()
    return out

class conv_network(nn.Module):
    """全连接神经网络"""
    def __init__(self, channels: int = 2, batch_size: int = 2, output_size: int = 10, K:int=64, M:int=32) -> None:
        super(conv_network, self).__init__()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(128,32, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*2*1, 512)
        self.fc2 = nn.Linear(512, 2*output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        # x = F.relu(self.conv4(x))
        # x = self.maxpool(x)
        # x = F.relu(self.conv5(x))
        # x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 2, -1)
        x[:, 0, :] = torch.sigmoid(x[:, 1, :])
        x[:, 1, :] = torch.tanh(x[:, 0, :])/2
        return x
    
class loss_func(nn.Module):
    """自定义损失函数"""
    def __init__(self, bce: float = 0.5, mse: float = 0.5) -> None:
        super(loss_func, self).__init__()
        self.bce_weight = bce
        self.mse_weight = mse
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, K, C = output.size()
        pred_prob, pred_angle = output[:, 0, :], output[:, 1, :]
        target = target.transpose(1,2)
        target_prob, target_angle = target[:, 0, :], target[:, 1, :]
        # 计算二进制交叉熵损失
        bce = self.bce_loss(pred_prob.reshape(B*C, 1), target_prob.reshape(B*C, 1))
        mse = self.mse_loss(pred_angle.reshape(B*C, 1), target_angle.reshape(B*C, 1))
        masked_mse = torch.sum(mse * target_prob.reshape(B*C, 1)) / (torch.sum(target_prob.reshape(B*C, 1)) + 3e-15)
        loss = self.bce_weight * bce + self.mse_weight * masked_mse
        return loss