import logging

import coloredlogs

from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game

from gobang.keras.NNet import NNetWrapper as nn
from RenjuGame import RenjuGame as Game

import torch

#import torch.nn as nn

#from gobang.GobangGame import GobangGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn


from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration. / 100
    'tempThreshold': 15,        #
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 800,          # Number of games moves for MCTS to simulate. / 25
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted. / 40
    'cpuct': 1,

    # 6번
    'checkpoint': './models/',
    'load_model': True,
    'load_folder_file': ('./models','checkpoint_29.pth.tar'),
    'numItersForTrainExamplesHistory': 30,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet.nnet.to(device)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')

    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

