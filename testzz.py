
from custom_function.immediate_win import find_immediate_win
from custom_function.renju_util import is_forbidden_nb

from alpha_zero_general.gobang.GobangPlayers import RandomPlayer
from alpha_zero_general.MCTS import MCTS, filter_near_stones
from alpha_zero_general.gobang.keras.NNet import NNetWrapper
from alpha_zero_general.utils import dotdict
from alpha_zero_general.Arena import Arena
from alpha_zero_general.utils import *

from collections import namedtuple
from RenjuGame import RenjuGame
from random import choice

import os, pickle, torch, pytest, argparse, math, time
import numpy as np



game = RenjuGame(n=15)
nnet = NNetWrapper(game)

asize = game.getActionSize()
board = game.getInitBoard()

board = game.getCanonicalForm(board, 1)
valids = game.getValidMoves(board, 1)

n = game.n

def wilson_low(p_hat, n, z=1.645):
    if n == 0: return 0.0
    den = 1 + (z*z)/n
    center = (p_hat + (z*z)/(2*n)) / den
    margin = z * math.sqrt((p_hat*(1-p_hat))/n + (z*z)/(4*n*n)) / den
    return center - margin

def pi_entropy(pi):
    p = np.asarray(pi[:-1], dtype=np.float64)
    s = p.sum()
    if s <= 0: return 0.0
    p = p / s
    return float(-(p * np.log(p + 1e-12)).sum())

def centrality_metrics(pi, n=15, inner=2):
    P = np.asarray(pi[:-1], dtype=np.float64).reshape(n, n)
    cy = cx = n // 2
    Y, X = np.ogrid[:n, :n]
    dist = np.abs(Y - cy) + np.abs(X - cx)
    center_mask = (dist <= inner)
    rim_mask    = ~center_mask
    total = P.sum() + 1e-12
    center_sum = P[center_mask].sum() / total
    rim_sum    = P[rim_mask].sum()    / total
    ratio = center_sum / (rim_sum + 1e-12)
    top = np.unravel_index(np.argmax(P), (n, n))
    return dict(center_mass=center_sum, rim_mass=rim_sum, ratio=ratio,
                top=top, top_val=float(P.max()), entropy=pi_entropy(pi))

def rotation_consistency(nnet, board, k=1):
    n = board.shape[0]
    pi1, _ = nnet.predict(board)
    b2 = np.rot90(board, k)
    pi2, _ = nnet.predict(b2)
    m1 = np.asarray(pi1[:-1]).reshape(n, n)
    m2 = np.asarray(pi2[:-1]).reshape(n, n)
    m2_back = np.rot90(m2, -k)
    return float(np.mean(np.abs(m1 - m2_back)))

def immediate_win_bot_factory(game):
    n = game.n
    def bot(board):
        valids = game.getValidMoves(board, 1).astype(np.int8)
        win = find_immediate_win(board, valids, n, player=+1)
        if np.any(win):
            # 첫 번째 승리 수 선택
            return int(np.flatnonzero(win)[0])
        # 없으면 합법 중 첫 수
        v = np.flatnonzero(valids)
        return int(v[0]) if len(v) else n*n  # pass fallback
    return bot

def eval_vs(game, nnet, args, opponent_play, games=60, sims=250):
    # 평가용 args 구성(동적 sims 쓰면 hi= s ims로 맞춰짐)
    class A: pass
    eval_args = A()
    for k, v in vars(args).items(): setattr(eval_args, k, v)
    eval_args.numMCTSSims = sims

    mcts = MCTS(game, nnet, eval_args, eval_mode=True)
    def my_play(b):
        # temp=0 정책(가장 방문수 높은 수)
        pi = mcts.getActionProb(b, temp=0)   # 확률 벡터
        return int(np.argmax(pi))

    arena = Arena(my_play, opponent_play, game)
    w, l, d = arena.playGames(games)
    n = w + l
    p_hat = (w / n) if n > 0 else 0.5
    lcb = wilson_low(p_hat, n)
    return dict(W=w, L=l, D=d, p=p_hat, LCB=lcb)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (pth.tar)")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--sims", type=int, default=250, help="eval MCTS sims")
    ap.add_argument("--games", type=int, default=60, help="games per benchmark")
    args_cli = ap.parse_args()

    game = RenjuGame(n=args_cli.n)
    nnet = NNetWrapper(game)
    if args_cli.ckpt:
        nnet.load_checkpoint(*split_dir_file(args_cli.ckpt))

    # 0) 빠른 위생 체크
    empty = np.zeros((args_cli.n, args_cli.n), dtype=int)
    pi0, v0 = nnet.predict(empty)
    cm = centrality_metrics(pi0, n=args_cli.n, inner=2)
    rot_score = rotation_consistency(nnet, empty, k=1)

    print("== Policy (빈판) ==")
    print(f" center_mass={cm['center_mass']:.3f}, rim_mass={cm['rim_mass']:.3f}, "
          f"ratio(center/rim)={cm['ratio']:.2f}")
    print(f" top={cm['top']} val={cm['top_val']:.4f}, entropy={cm['entropy']:.3f}")
    print(f" rotation_consistency(L1, 90deg)={rot_score:.4e}")
    print(f" value(empty)={float(v0):.3f}")
    print()

    # 1) 랜덤/즉시승봇 벤치마크
    #   - 시간 오래 안 걸리도록 게임 수는 기본 60
    class DummyArgs: pass
    dummy = DummyArgs()
    dummy.__dict__.update(dict(numMCTSSims=args_cli.sims,
                               cpuct=1.5, tempThreshold=12))  # 필요한 최소 필드

    print("== Benchmarks ==")
    t0 = time.time()
    # vs Random
    rand_play = RandomPlayer(game).play
    R = eval_vs(game, nnet, dummy, rand_play, games=args_cli.games, sims=args_cli.sims)
    print(f" vs Random: W-L-D = {R['W']}-{R['L']}-{R['D']}, "
          f"p̂={R['p']:.3f}, LCB(95%)={R['LCB']:.3f}")

    # vs ImmediateWin bot
    imw_play = immediate_win_bot_factory(game)
    I = eval_vs(game, nnet, dummy, imw_play, games=args_cli.games, sims=args_cli.sims)
    print(f" vs ImmediateWin: W-L-D = {I['W']}-{I['L']}-{I['D']}, "
          f"p̂={I['p']:.3f}, LCB(95%)={I['LCB']:.3f}")
    print(f"(elapsed {time.time()-t0:.1f}s)")

def split_dir_file(path):
    import os
    d, f = os.path.split(path)
    if d == "": d = "."
    return d + ("" if d.endswith(os.sep) else os.sep), f


if __name__ == "__main__":
    main()




'''
nnet.load_checkpoint('./models/','best.pth.tar')
pi, v = nnet.predict(np.zeros((15,15),dtype=int))
print(pi[:-1].reshape(15,15))  # 중심 부근 확률이 높아야 정상
print(v) 
'''

'''
arena = Arena(leftmost_player, rightmost_player, game)
    w1, w2, d = arena.playGames(50, verbose=False)
    print("leftmost vs rightmost", w1, w2, d)
'''

'''
# 흑 착수
board[1, 1] = 1
board[4, 1] = 1

board[2, 3] = 1
board[10, 3] = 1
board[11, 3] = 1

board[2, 4] = 1
board[9, 4] = 1

board[5, 5] = 1
board[10, 5] = 1

board[5, 6] = 1

board[6, 7] = 1
board[7, 7] = 1

board[11, 8] = 1

board[5, 9] = 1
board[10, 9] = 1
board[12, 9] = 1

board[1, 10] = 1
board[11, 10] = 1

board[3, 11] = 1
board[4, 11] = 1

board[1, 12] = 1

# 백 착수
board[0, 8] = -1
board[1, 8] = -1
board[2, 8] = -1
board[3, 8] = -1

board[8, 9] = -1
board[14, 9] = -1

board[14, 11] = -1
board[14, 12] = -1
board[14, 13] = -1

board[0, 14] = -1
board[1, 14] = -1
board[2, 14] = -1
board[3, 14] = -1
board[5, 14] = -1
board[6, 14] = -1
board[7, 14] = -1
board[8, 14] = -1
board[10, 14] = -1
board[11, 14] = -1
board[12, 14] = -1
board[13, 14] = -1

valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)

for i in range(0, 15**2, 15):
    print(valids[i:i+15])


#4.8

'''
'''
import time

sum_ = 0
for i in range(10):
    s = time.time()
    for i in range(10):
        valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)
    e = time.time()
    print(e - s)

    sum_ += (e-s)

valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)

for i in range(0, 15**2, 15):
    print(valids[i:i+15])

print(sum_ / 10)
'''

'''
# 보드에서 1이 흑 / 흑이 착수할 차례에는 흑이 1이 되기때문

valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)

for i in range(0, 15**2, 15):
    print(valids[i:i+15])
'''
'''
valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)

for i in range(0, 15**2, 15):
    print(valids[i:i+15])
'''

'''
import time

s = time.time()
for i in range(10):
    valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)
print(time.time() - s)
'''

'''
occupied = (board != 0)
forbidden_only = (~occupied) & (valids_grid == 0)

forbidden_mask = np.zeros_like(board, dtype=np.int8)

for y in range(15):
    for x in range(15):
        if board[y, x] != 0:
            continue

        if is_forbidden_nb(board, x, y, 15, stone=1):
            forbidden_mask[y, x] = 1

print(forbidden_mask)
print(forbidden_only.astype(np.int8))

for i in range(0, 15**2, 15):
    print(valids[i:i+15])
'''

'''
result = find_immediate_win(board, valids, 15, player=-1)

for i in valids:
    print(i)
'''

    

'''
for i in range(0, 15**2, 15):
    print(valids[i:i+15])
'''

# 장목, 44, 33 막음

'''
args = dotdict({
    'numMCTSSims': 20,
    'cpuct': 1.0,
})


game = RenjuGame(n=15)
nnet = NNetWrapper(game)
mcts = MCTS(game, nnet, args)

board = game.getInitBoard()

for i in range(4):
    board[4, i+2] = 1

nnet.load_checkpoint('./models/','temp.pth.tar')

pi = mcts.getActionProb(game.getCanonicalForm(board, 1))
action = int(np.argmax(pi))
y, x = divmod(action, game.n)

print(y, x)
'''

'''
valids = game.getValidMoves(board, 1)
win_mask = find_immediate_win(board, valids, game.n)

for i in range(0, n**2, n):
    print(valids[i:i+n])

print("-------------------------")

for i in range(0, n**2, n):
    print(win_mask[i:i+n])

safe = valids.copy()
opp = -1
for a in np.where(valids == 1)[0]:
    b2, _ = game.getNextState(board, 1, a)
    opp_valids = game.getValidMoves(b2, opp)

    for j in np.where(opp_valids == 1)[0]:
        b3, _ = game.getNextState(b2, opp, j)
        if game.getGameEnded(b3, opp) == 1:
            safe[a] = 0
            break

if safe.sum() > 0:
    valids = safe
    

print(valids)
'''


'''
args = dotdict({
    'checkpoint': './models/',      # 체크포인트 폴더 경로
    'best_model_file': 'checkpoint_1.pth.tar',  # 테스트할 모델 파일명
    'arenaCompare': 20                  # 랜덤 플레이어와 대결할 게임 수
})


def print_result(desc, result, expected):
    status = "✅" if result == expected else "❌"
    print(f"{status} {desc}: got {result}, expected {expected}")

def test_horizontal_five():
    game = RenjuGame(n=15, nir=5)
    board = np.zeros((15,15), dtype=int)
    # 가로 5목 (player=1)이면 1 반환
    for x in range(2, 7):
        board[7, x] = -1
    res = game.getGameEnded(board, -1)
    print_result("Horizontal five for player 1", res, -1)

def test_vertical_five():
    game = RenjuGame(n=15, nir=5)
    board = np.zeros((15,15), dtype=int)
    # 세로 5목 (player=-1)이면 -1 반환
    for y in range(5, 10):
        board[y, 4] = -1
    res = game.getGameEnded(board, -1)
    print_result("Vertical five for player -1", res, -1)

def test_no_winner():
    game = RenjuGame(n=15, nir=5)
    board = np.zeros((15,15), dtype=int)
    res = game.getGameEnded(board, 1)
    print_result("Empty board", res, 0)

def test_draw():
    game = RenjuGame(n=5, nir=5)  # 작은 보드로 테스트
    board = np.array(
        [[1, -1, 1, -1, 1],
         [-1, -1, 1, -1 ,-1],
         [1, 1, -1, 1, 1],
         [-1, -1, 1, 1, -1],
         [-1, 1, -1, -1, 1]]
    )

    # 모두 채워졌으니 반환값이 1e-4
    res = game.getGameEnded(board, -1)
    print_result("Full board draw", res, 1e-4)

def test_renju():
    import time

    n = 6

    game = RenjuGame(n=n, nir=5)
    board = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0],
        [0, 0, 0, -1, -1, 0],
        [0, 0, -1, 0, -1, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    s = time.time()

    valids = np.array(game.getValidMoves(board, 1))
    
    e = time.time()
    #print(e- s)
    # 1.82~1.84

    for i in range(0, n**2, n):
        print(valids[i:i+n])




    

if __name__ == "__main__":
    #test_renju()

    test_horizontal_five()
    test_vertical_five()
    test_no_winner()
    test_draw()
'''


'''
game = RenjuGame()
nnet = NNetWrapper(game)

nnet.load_checkpoint('./models/','temp.pth.tar')
pi, v = nnet.predict(np.zeros((15,15),dtype=int))
print(pi[:-1].reshape(15,15))  # 중심 부근 확률이 높아야 정상
print(v) 
'''

'''
from gomoku_renju import GomokuRenju
from alpha_zero_general.NNetWrapper import NNetWrapper
'''

'''
game = RenjuGame(n=15)
nnet = NNetWrapper(game)

ckpt = nnet.load_checkpoint('./models/','temp.pth.tar')

print(ckpt.keys())
'''

'''
# 흑 착수
board[0, 2] = 1
board[1, 2] = 1
board[3, 2] = 1
board[4, 2] = 1
board[5, 2] = 1


board[3, 3] = 1

board[9, 4] = 1
board[10, 4] = 1

board[10, 5] = 1

board[3, 6] = 1
board[10, 6] = 1
board[11, 6] = 1

board[7, 7] = 1
board[8, 7] = 1
board[9, 7] = 1

board[3, 8] = 1

board[3, 9] = 1
board[6, 9] = 1
board[7, 9] = 1
board[9, 9] = 1

board[7, 10] = 1

# 백 착수
board[10, 3] = -1

board[6, 7] = -1

board[14, 6] = -1
board[14, 7] = -1
board[14, 8] = -1

board[14, 10] = -1
board[14, 11] = -1
board[14, 12] = -1
board[14, 13] = -1

board[0, 14] = -1
board[1, 14] = -1
board[2, 14] = -1
board[3, 14] = -1
board[5, 14] = -1
board[6, 14] = -1
board[7, 14] = -1
board[8, 14] = -1
board[10, 14] = -1
board[11, 14] = -1
board[12, 14] = -1
board[13, 14] = -1








# 흑 착수
board[1, 1] = 1
board[4, 1] = 1

board[2, 3] = 1
board[10, 3] = 1
board[11, 3] = 1

board[2, 4] = 1
board[9, 4] = 1

board[5, 5] = 1
board[10, 5] = 1

board[5, 6] = 1

board[6, 7] = 1
board[7, 7] = 1

board[11, 8] = 1

board[5, 9] = 1
board[10, 9] = 1
board[12, 9] = 1

board[1, 10] = 1
board[11, 10] = 1

board[3, 11] = 1
board[4, 11] = 1

board[1, 12] = 1

# 백 착수
board[0, 8] = -1
board[1, 8] = -1
board[2, 8] = -1
board[3, 8] = -1

board[8, 9] = -1
board[14, 9] = -1

board[14, 11] = -1
board[14, 12] = -1
board[14, 13] = -1

board[0, 14] = -1
board[1, 14] = -1
board[2, 14] = -1
board[3, 14] = -1
board[5, 14] = -1
board[6, 14] = -1
board[7, 14] = -1
board[8, 14] = -1
board[10, 14] = -1
board[11, 14] = -1
board[12, 14] = -1
board[13, 14] = -1
'''