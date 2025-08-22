import logging
import math

from numba import njit
import numpy as np

from custom_function.immediate_win import find_immediate_win

EPS = 1e-8

log = logging.getLogger(__name__)

def filter_near_stones(board, indices, radius=3):
    n = board.shape[0]
    occ = (board != 0).astype(np.int32)

    ii = np.pad(occ.cumsum(0).cumsum(1), ((1,0),(1,0)))
    kept = []

    for idx in indices:
        y, x = divmod(idx, n)
        y0, y1 = max(0, y-radius), min(n-1, y+radius)
        x0, x1 = max(0, x-radius), min(n-1, x+radius)

        # 직사각형 합 = ii[y1+1,x1+1]-ii[y0,x1+1]-ii[y1+1,x0]+ii[y0,x0]
        s = ii[y1+1, x1+1] - ii[y0, x1+1] - ii[y1+1, x0] + ii[y0, x0]
        if s > 0:
            kept.append(idx)

    if not kept:  # 초기 무착의 중앙 가중 로직은 기존과 동일하게 유지
        c = n // 2
        kept = [idx for idx in indices if abs(divmod(idx,n)[0]-c)<=1 and abs(divmod(idx,n)[1]-c)<=1]
    return np.array(kept, dtype=np.int32)

def _safe_normalize(vec, mask=None):
    v = np.asarray(vec, dtype=np.float64)
    if mask is not None:
        v = v * mask.astype(np.float64)

    s = v.sum()
    if (not np.isfinite(s)) or s <= 0:
        if mask is not None:
            m = int(np.count_nonzero(mask))
            if m > 0:
                v = mask.astype(np.float64) / float(m)  # 유효수 균등
            else:
                v = np.ones_like(v, dtype=np.float64) / float(v.size)  # 전수 균등
        else:
            v = np.ones_like(v, dtype=np.float64) / float(v.size)
    else:
        v /= s

    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = v.sum()
    if (not np.isfinite(s2)) or s2 <= 0:
        v = np.ones_like(v, dtype=np.float64) / float(v.size)
    return v

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, eval_mode=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.eval_mode = eval_mode


    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        game = self.game
        board = canonicalBoard.copy()

        # 1) 한 번만: 전체 합법 수
        orig_valids = np.array(game.getValidMoves(board, 1), dtype=np.int8)

        if not self.eval_mode:
            # 3) 즉시 승
            my_win = find_immediate_win(board, orig_valids, game.n, player=+1)
            if np.any(my_win):
                return _safe_normalize(my_win).tolist()
            
            # 4) 상대 즉시 승
            empties = (board.ravel(order="C") == 0).astype(np.int8, copy=False)
            empties = np.concatenate([empties, np.array([0], dtype=np.int8)])
            opp_wins_any = find_immediate_win(board, empties, game.n, player=-1)
            block = (opp_wins_any.astype(np.int8) & orig_valids)
            if np.any(block):
                return _safe_normalize(block).tolist()
            
            valids = orig_valids.copy()
        
        if self.eval_mode:
            valids = orig_valids

        #test
        s = game.stringRepresentation(canonicalBoard)
        self.Vs[s] = valids

        stones = int(np.count_nonzero(board))
        lo, mid, hi = 200, 400, self.args.numMCTSSims
        eff_sims = int(np.interp(stones, [0, 15, 60], [lo, mid, hi]))

        # 6) 시뮬레이션
        for _ in range(eff_sims):
            self.search(canonicalBoard)

        #s = game.stringRepresentation(canonicalBoard)

        counts = np.array([self.Nsa.get((s, a), 0) for a in range(game.getActionSize())], dtype=np.float64)
        counts *= valids  # 마스킹

        # 7) 분포 생성 (fallback → temp → 정규화)
        if counts.sum() <= 0:
            return _safe_normalize(valids).tolist()

        if temp == 0:
            best = int(np.argmax(counts))
            pi = np.zeros_like(counts)
            pi[best] = 1.0
            return pi.tolist()
        
        counts = counts ** (1.0 / float(temp))
        counts = np.clip(counts, 0, None)
    
        return _safe_normalize(counts).tolist()

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]
        
        if s not in self.Ps:
            # test
            valids = self.Vs.get(s)

            if valids is None:
                valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids
            self.Ps[s] = _safe_normalize(self.Ps[s], mask=valids)

            '''
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            '''
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        #test
        for a in np.flatnonzero(valids):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
            if u > cur_best:
                cur_best, best_act = u, a
        
        '''
        # pick the action with the highest upper confidence 
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
        '''

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
    
