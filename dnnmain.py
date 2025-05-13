from isibc_targetid import TargetIdentifier, plot_results
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from dnn import data_preprocess, conv_network, loss_func, estimation_loss
import torch
import torch.nn as nn
import torch.optim as optim
import os

def process_sample(K, M, N, snr):
    ti = TargetIdentifier(K, M, N, snr)
    num_sources = np.random.randint(1, 5)
    theta_true = np.random.uniform(-0.4, 0.4, num_sources)
    theta_env_true = np.random.uniform(-0.4, 0.4, 1)
    data, true_value = ti.generate_signal(theta_true, theta_env_true)
    data_preprocessed, angle = data_preprocess(data, true_value, K, M, N)
    return data_preprocessed, angle

def main(n_thread:int = 8):
    # 参数设置（与MATLAB demo保持一致）
    np.random.seed(36)  # 固定随机种子确保可重复性
    n_train = 10000
    n_test = 10
    eval_iters = 200
    batch_size = 96
    K=16
    M=30
    N=8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = conv_network(channels=2*M, batch_size=batch_size, output_size=N, K=K, M=M).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6) ##模型参数量
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    for epoch in range(n_train):
        snr = 20#np.random.uniform(0, 20)
        optimizer.zero_grad()
        data_batch = []
        tag_batch = []
        with ThreadPoolExecutor(max_workers=n_thread) as executor:  # 设置线程数，例如 8
            futures = [executor.submit(process_sample, K, M, N, snr) for _ in range(batch_size)]
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
        bce = 0
        mse = 1-bce
        loss_fn = loss_func(bce, mse).to(device)
        loss = loss_fn(output, tag_tensor)
        optimizer.step()
        if epoch % eval_iters == 0:
            loss_train = loss.item()
            loss_test = estimation_loss(model, n_test, loss_fn, device, K, M, N, batch_size, 24)
            print(f'iter: {epoch}, loss train: {loss_train} loss val {loss_test[0]}')


if __name__ == "__main__":
    max_threads = os.cpu_count()
    print(max_threads)
    n_thread = int(input("number of threads: "))
    main(n_thread)