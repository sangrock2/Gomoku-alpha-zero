# eval_prev_vs_new.py
import os, time, math, argparse, numpy as np

# 프로젝트 import (경로 필요시 수정)
from RenjuGame import RenjuGame
from alpha_zero_general.gobang.keras.NNet import NNetWrapper as NNet
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.Arena import Arena

try:
    # dotdict이 utils에 있다면
    from alpha_zero_general.utils import dotdict
except Exception:
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

def wilson_lcb(wins, n, z=1.96):
    if n <= 0: return 0.0
    p = wins / n
    denom = 1 + (z*z)/n
    center = p + (z*z)/(2*n)
    margin = z * math.sqrt((p*(1-p) + (z*z)/(4*n))/n)
    return max(0.0, (center - margin)/denom)

def split_folder_file(path):
    folder = os.path.dirname(path)
    if folder == "": folder = "."
    return folder, os.path.basename(path)

def make_eval_args(sims, cpuct=1.0):
    a = dotdict({})
    a.numMCTSSims = int(sims)
    a.cpuct = float(cpuct)
    # 평가 모드: 노이즈/온도 끔
    a.epsilon = 0.0
    a.dirichletAlpha = 0.0
    a.tempThreshold = 0
    return a

def load_model(game, path):
    nnet = NNet(game)
    folder, fn = split_folder_file(path)
    nnet.load_checkpoint(folder=folder, filename=fn)
    return nnet

def evaluate(game, new_model_path, prev_model_path, sims=300, games=40, cpuct=1.0, early=False):
    nnet_new  = load_model(game, new_model_path)
    nnet_prev = load_model(game, prev_model_path)

    args_eval = make_eval_args(sims, cpuct)
    mcts_new  = MCTS(game, nnet_new,  args_eval, eval_mode=True)
    mcts_prev = MCTS(game, nnet_prev, args_eval, eval_mode=True)

    def p_new(x):  return np.argmax(mcts_new.getActionProb(x,  temp=0))
    def p_prev(x): return np.argmax(mcts_prev.getActionProb(x, temp=0))

    # 양쪽 색 번갈아 평가 (선공 편향 제거)
    g1 = games // 2
    g2 = games - g1

    start = time.time()

    # 새 모델이 선공
    arena1 = Arena(p_new, p_prev, game)
    if early and hasattr(arena1, "playGames_early"):
        w1, l1, d1 = arena1.playGames_early(max_games=g1, target_wins=(g1//2 + 1))
    else:
        w1, l1, d1 = arena1.playGames(g1)

    # 새 모델이 후공
    arena2 = Arena(p_prev, p_new, game)
    if early and hasattr(arena2, "playGames_early"):
        l2, w2, d2 = arena2.playGames_early(max_games=g2, target_wins=(g2//2 + 1))
    else:
        l2, w2, d2 = arena2.playGames(g2)

    elapsed = time.time() - start

    new_wins  = w1 + w2
    new_loss  = l1 + l2
    new_draws = d1 + d2
    decided   = new_wins + new_loss

    rate = (new_wins / decided) if decided > 0 else 0.5
    lcb  = wilson_lcb(new_wins, decided)

    return {
        "W": new_wins, "L": new_loss, "D": new_draws,
        "decided": decided,
        "win_rate": rate, "wilson_lcb95": lcb,
        "sims": sims, "games": games, "elapsed_sec": elapsed,
        "split": {"new_as_P1": (w1, l1, d1), "new_as_P2": (w2, l2, d2)}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new",  type=str, required=True, help="새 모델 체크포인트 경로 (예: ./models/best.pth.tar)")
    ap.add_argument("--prev", type=str, required=True, help="이전 모델 체크포인트 경로")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--cpuct", type=float, default=1.0)
    ap.add_argument("--board", type=int, default=15)
    ap.add_argument("--early", action="store_true", help="조기종료 사용(playGames_early가 있을 때만)")
    args = ap.parse_args()

    game = RenjuGame(n=args.board)
    out = evaluate(game, args.new, args.prev, sims=args.sims, games=args.games, cpuct=args.cpuct, early=args.early)

    print(f"== External Eval (sims={out['sims']}, games={out['games']}) ==")
    print(f"W-L-D = {out['W']}-{out['L']}-{out['D']}  (decided={out['decided']})")
    print(f"win_rate = {out['win_rate']:.3f}   LCB95 = {out['wilson_lcb95']:.3f}")
    print(f"split  P1(new)={out['split']['new_as_P1']},  P2(new)={out['split']['new_as_P2']}")
    print(f"(elapsed {out['elapsed_sec']:.1f}s)")

if __name__ == "__main__":
    main()
