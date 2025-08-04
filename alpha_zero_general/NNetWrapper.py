import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from othello.pytorch.NNet import NNetWrapper as NN
from NeuralNet import NeuralNet
from RenjuGame import GomokuRenju
from tqdm import tqdm
from utils import *

class NNetWrapper(NN):
    def __init__(self, game: GomokuRenju, lr=1e-3, epochs=10, batch_size=64):
        super().__init__(game)

        self.n_x, self.n_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*self.n_x*self.n_y, 256),
            nn.ReLU(),
        )

        self.pi_head = nn.Linear(256, self.action_size)
        self.v_head = nn.Linear(256, 1)

        self.model.to(self.device)
        self.pi_head.to(self.device)
        self.v_head.to(self.device)

        params = list(self.model.parameters()) + list(self.pi_head.parameters()) + list(self.v_head.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, examples):
        self.model.train()

        for epoch in range(self.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            np.random.shuffle(examples)

            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            
            t = tqdm(range(0, len(examples), self.batch_size), desc='Training Net')
            for i in t:
                batch = examples[i:i+self.batch_size]
                boards, pis, vs = zip(*batch)

                x = torch.tensor(np.array(boards)[:, None, :, :], dtype=torch.float32, device=self.device)
                target_pi = torch.tensor(np.array(pis), dtype=torch.float32, device=self.device)
                target_v = torch.tensor(np.array(vs), dtype=torch.float32, device=self.device).view(-1, 1)

                self.optimizer.zero_grad()
                h = self.model(x)

                out_pi = self.pi_head(h)
                out_v = self.v_head(h)

                loss_pi = -torch.mean(torch.sum(target_pi * nn.LogSoftmax(dim=1)(out_pi), dim=1))
                loss_v  = torch.mean((out_v - target_v)**2)
                loss = loss_pi + loss_v

                pi_losses.update(loss_pi.item(), x.size(0))
                v_losses.update(loss_v.item(), x.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                loss.backward()
                self.optimizer.step()

    def predict(self, board):
        self.model.eval()

        b = torch.tensor(board[None, None, :, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            h = self.model(b)
            logits = self.pi_head(h).cpu().numpy().flatten()
            v = self.v_head(h).cpu().numpy().flatten()[0]

        pi = np.exp(logits) / np.sum(np.exp(logits))

        return pi, v
    
    def save_checkpoint(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'pi_head': self.pi_head.state_dict(),
            'v_head': self.v_head.state_dict(),
        }, os.path.join(folder, filename))

    def load_checkpoint(self, folder, filename):
        path = os.path.join(folder, filename)
        ckpt = torch.load(path, map_location='cpu')

        model_sd = ckpt['model']
        pi_sd = ckpt['pi_head']
        v_sd = ckpt['v_head']

        self.model.load_state_dict(model_sd)
        self.pi_head.load_state_dict(pi_sd)
        self.v_head.load_state_dict(v_sd)

        self.model.to(self.device)
        self.pi_head.to(self.device)
        self.v_head.to(self.device)