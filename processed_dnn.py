from isibc_targetid import TargetIdentifier, plot_results
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from dnn import
import torch
import torch.nn as nn
import torch.optim as optim
import os

def training(model, n_train, n_test, eval_iters, batch_size, device):
    optimizer = optim.AdamW(model.parameters(), lr=0.1)
    for epoch in range(n_train):
        optimizer.zero_grad()
        snr = 20 
        data_batch = []
        tag_batch = []
        
        model.loss(output, tag)
        optimizer.step()
        if epoch % eval_iters == 0:
            print(f'iter: {epoch}, loss train: {loss_train} loss val {loss_test[0]}')