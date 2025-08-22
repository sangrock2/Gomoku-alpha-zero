from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.utils import dotdict


from alpha_zero_general.gobang.keras.NNet import NNetWrapper as nn
from RenjuGame import RenjuGame as Game
from custom_function.renju_util import is_forbidden_nb

import numpy as np
import torch

import logging
from typing import List

# 실행어
# uvicorn ai_service:app --reload --host 0.0.0.0 --port 8000

# 경로 추가 CMD 파일 root에서
# set PYTHONPATH=%cd%\alpha_zero_general

# 기존 numpy 버전 2.2.6

app = FastAPI()
log = logging.getLogger("uvicorn.error")  # uvicorn 콘솔로 나감

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

game = Game()
nnet = nn(game)

nnet.load_checkpoint(folder='./models/', filename='best.pth.tar')

args = dotdict({
    'numMCTSSims': 1000,
    'cpuct': 1.0,
})

mcts = MCTS(game, nnet, args)

class CkeckReq(BaseModel):  
    board: List[List[int]]  # -1=흑, 1=백, 0=빈  (프론트→백엔드)
    x: int
    y: int
    player: int

class HintRequest(BaseModel):
    board: list[list[int]]
    player: int
    difficulty: int

class CheckReq(BaseModel):
    board: List[List[int]]   # 1=흑, -1=백, 0=빈  (프론트 그대로 보냄)
    x: int
    y: int
    player: int 

class ForbiddenMapReq(BaseModel):
    board: list[list[int]]
    player: int

class Move(BaseModel):
    x: int
    y: int

@app.post('/api/ai/hint', response_model=Move)
def hint(req: HintRequest):
    b = np.array(req.board, dtype=int)

    log.info(pretty_board(req.board, "req.board(from frontend)"))

    if b.shape != (game.n, game.n):
        raise HTTPException(400, f'board shpae must be ({game.n}, {game.n})')

    '''
    sims_map = {1:10, 2:20, 3:30, 4:40, 5:100}
    mcts.args['numMCTSSims'] = sims_map.get(req.difficulty, 50)
    '''

    pi = mcts.getActionProb(game.getCanonicalForm(b, req.player))

    action = int(np.argmax(pi))

    '''
    if req.difficulty < 4:
        temp_map = {1:4.0, 2:2.0, 3:1.0}
        T = temp_map[req.difficulty]

        pi = np.power(pi, 1.0 / T)
        pi = pi / np.sum(pi)

        action = int(np.random.choice(len(pi), p=pi))
    else:
        action = int(np.argmax(pi))
    '''

    y, x = divmod(action, game.n)

    return Move(x=x, y=y)

@app.post("/check")
def check_forbidden(req: CheckReq):
    b = np.array(req.board, dtype=np.int8)
    # 금수는 흑에게만 적용
    if req.player != 1:
        return {"forbidden": False}
    # 빈칸 아니면 금수 이전에 불가
    if b[req.y, req.x] != 0:
        return {"forbidden": True, "reason": "occupied"}
    # ★ 진짜 금수만 판정
    forb = bool(is_forbidden_nb(b, req.x, req.y, n=b.shape[0], stone=+1))
    return {"forbidden": forb}

@app.post("/api/renju/forbidden-map")
def forbidden_map(req: ForbiddenMapReq):
    n = len(req.board)
    b = game.getCanonicalForm(np.array(req.board, dtype=np.int8), req.player)

    out = []
    for y in range(n):
        for x in range(n):
            if b[y, x] != 0:
                continue
            if is_forbidden_nb(b, x, y, n=n, stone=+1):
                out.append({"x": x, "y": y})

    return {"forbidden": out}

def pretty_board(b: List[List[int]], title="board") -> str:
    if not b:
        return f"[{title}] <empty>"
    h, w = len(b), len(b[0])
    lines = [f"[{title}] shape={h}x{w}  (y=row, x=col; -1=흑, 1=백, 0=빈)"]
    for yi, row in enumerate(b):
        lines.append(f"y={yi:02d} | " + " ".join(f"{v:2d}" for v in row))
    lines.append("       " + "-" * (3*w-1))
    lines.append("       " + " ".join(f"{xi:2d}" for xi in range(w)))
    return "\n".join(lines)