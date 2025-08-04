from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from alpha_zero_general.NNetWrapper import NNetWrapper
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.utils import dotdict
from RenjuGame import GomokuRenju

import numpy as np

# 실행어
# uvicorn ai_service:app --host 0.0.0.0 --port 8000

# 경로 추가 CMD 파일 root에서
# set PYTHONPATH=%cd%\alpha_zero_general

# 기존 numpy 버전 2.2.6

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

game = GomokuRenju(n=17)

nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./models/', filename='best.pth.tar')

args = dotdict({
    'numMCTSSims': 100,
    'cpuct': 1.0,
})

mcts = MCTS(game, nnet, args)

class HintRequest(BaseModel):
    board: list[list[int]]
    player: int
    difficulty: int

class Move(BaseModel):
    x: int
    y: int

@app.post('/api/ai/hint', response_model=Move)
def hint(req: HintRequest):
    b = np.array(req.board, dtype=int)

    sims_map = {1:20, 2:40, 3:80, 4:160, 5:320}
    mcts.args['numMCTSSims'] = sims_map.get(req.difficulty, 100)

    pi = mcts.getActionProb(game.getCanonicalForm(b, req.player))

    if req.difficulty < 4:
        temp_map = {1:5.0, 2:2.0, 3:1.0}
        T = temp_map[req.difficulty]

        pi = np.power(pi, 1.0 / T)
        pi = pi / np.sum(pi)

        action = int(np.random.choice(len(pi), p=pi))
    else:
        action = int(np.argmax(pi))

    y, x = divmod(action, game.size)

    return Move(x=x, y=y)