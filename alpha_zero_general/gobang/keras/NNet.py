import os
import sys
import time
import numpy as np
from tqdm import tqdm

# 프로젝트 최상위가 alpha_zero_general/ 라고 가정
sys.path.append('../../') 

from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp  import grad_scaler, autocast_mode
from .GobangNNet import GobangNNet as onnet

cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

args = dotdict({
    'lr': 0.0005,
    'dropout': 0.2,
    'epochs': 8,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size     = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        input_boards, target_pis, target_vs = zip(*examples)
        input_boards = np.asarray(input_boards, dtype=np.float32)
        target_pis   = np.asarray(target_pis,   dtype=np.float32)
        target_vs    = np.asarray(target_vs,    dtype=np.float32)

        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        scaler = grad_scaler.GradScaler("cuda")

        for epoch in range(args.epochs):
            print(f'EPOCH ::: {epoch+1}')
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses  = AverageMeter()

            batch_count = len(examples) // args.batch_size
            loop = tqdm(range(batch_count), desc='Training Net')
            for _ in loop:
                idxs = np.random.randint(len(examples), size=args.batch_size)
                boards = torch.from_numpy(input_boards[idxs])
                pis    = torch.from_numpy(target_pis[idxs])
                vs     = torch.from_numpy(target_vs[idxs])

                if args.cuda:
                    boards, pis, vs = boards.cuda(), pis.cuda(), vs.cuda()

                optimizer.zero_grad()

                with autocast_mode.autocast("cuda"):
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(pis, out_pi)
                    l_v  = self.loss_v(vs, out_v)
                    loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(),  boards.size(0))
                loop.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                '''
                loss.backward()
                optimizer.step()
                '''

    def predict(self, board):
        board = torch.from_numpy(board.astype(np.float32))
        if args.cuda:
            board = board.cuda()
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            if args.cuda:
                with autocast_mode.autocast('cuda'):
                    log_pi, v = self.nnet(board)
            else:
                log_pi, v = self.nnet(board)

        pi = torch.exp(log_pi).cpu().numpy()[0]
        v = v.cpu().numpy()[0]

        ssum = np.sum(pi)
        if not np.isfinite(ssum) or ssum <= 0:
            pi = np.ones(self.action_size, dtype=np.float32 / float(self.action_size))

        return pi, v
    
    

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'

        checkpoint = torch.load(filepath, map_location=map_location)

        self.nnet.load_state_dict(checkpoint['state_dict'])

'''
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .GobangNNet import GobangNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})



class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]
        
        pi, v = self.nnet.model.predict(board, verbose=False)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
'''