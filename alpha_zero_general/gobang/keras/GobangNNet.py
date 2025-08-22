import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class GobangNNet(nn.Module):
    def __init__(self, game, args):
        super(GobangNNet, self).__init__()

        self.args = args
        bx, by = game.getBoardSize()
        self.action_size = game.getActionSize()
        nc = args.num_channels

        self.conv1 = nn.Conv2d(1,    nc, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(nc)
        self.conv2 = nn.Conv2d(nc,   nc, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(nc)
        self.conv3 = nn.Conv2d(nc,   nc, 3, padding=1)      # valid
        self.bn3   = nn.BatchNorm2d(nc)
        self.conv4 = nn.Conv2d(nc,   nc, 3, padding=1)      # valid
        self.bn4   = nn.BatchNorm2d(nc)

        # Global Average Pooling 후 채널 축소
        self.head_channels = 128
        self.reduce = nn.Conv2d(nc, self.head_channels, 1)

        # Policy head
        self.pi_conv1 = nn.Conv2d(nc, 32, kernel_size=1)
        self.pi_bn1 = nn.BatchNorm2d(32)
        self.pi_conv2 = nn.Conv2d(32, 1, kernel_size=1)
        #self.pi_fc = nn.Linear(self.head_channels, self.action_size)

        # Value head
        self.v_reduce = nn.Conv2d(nc, 128, kernel_size=1)
        self.v_fc1 = nn.Linear(128, 64)
        self.v_fc2 = nn.Linear(64, 1)


        '''
        flat_dim = nc * (bx-2-2) * (by-2-2)
        self.fc1  = nn.Linear(flat_dim, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2  = nn.Linear(1024,    512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        self.fc_pi = nn.Linear(512,    self.action_size)
        self.fc_v  = nn.Linear(512,    1)
        '''

    def forward(self, s):
        # s: batch x bx x by
        b, _, _ = s.shape
        x = s.view(b, 1, *s.shape[1:])     # batch x 1 x bx x by
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feat = F.relu(self.bn4(self.conv4(x)))
        #x = F.relu(self.bn4(self.conv4(x)))


        # Policy
        p = F.relu(self.pi_bn1(self.pi_conv1(feat)))
        p = self.pi_conv2(p).view(b, -1)
        pass_logit = p.new_full((b, 1), -10.0)
        pi_logits = torch.cat([p, pass_logit], dim=1)
        log_pi = F.log_softmax(pi_logits, dim=1)

        # Value
        v = F.relu(self.v_reduce(feat))          # (B,128,n,n)
        v = F.adaptive_avg_pool2d(v, 1).view(b, -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return log_pi, v

        '''
        x = x.view(b, -1)
        x = F.dropout(F.relu(self.bn_fc1(self.fc1(x))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.bn_fc2(self.fc2(x))), p=self.args.dropout, training=self.training)

        pi = self.fc_pi(x)                # logits
        v  = self.fc_v(x)                 # scalar
        return F.log_softmax(pi, dim=1), torch.tanh(v)
        '''



'''
import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class GobangNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))

'''