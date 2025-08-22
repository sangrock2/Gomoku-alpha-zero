from alpha_zero_general.gobang.GobangGame import GobangGame as BaseGoBang
from custom_function.renju_util import is_forbidden_nb
from alpha_zero_general.MCTS import filter_near_stones
from collections import deque

import numpy as np
import re

from numba import njit, int8

@njit
def has_exactly_five(board: np.ndarray, player: int, n:int):
    dirs = ((1,0),(0,1),(1,1),(1,-1))
    N = board.shape[0]
    for y in range(N):
        for x in range(N):
            if board[y, x] != player:
                continue
            for dx, dy in dirs:
                cnt = 1
                # 앞으로
                nx, ny = x+dx, y+dy
                while 0 <= nx < N and 0 <= ny < N and board[ny, nx] == player:
                    cnt += 1
                    nx += dx; ny += dy
                # 뒤로
                bx, by = x-dx, y-dy
                while 0 <= bx < N and 0 <= by < N and board[by, bx] == player:
                    cnt += 1
                    bx -= dx; by -= dy
                if cnt == n:
                    # 양끝 차단 검사
                    end1 = 0 <= bx < N and 0 <= by < N and board[by, bx] == player
                    end2 = 0 <= nx < N and 0 <= ny < N and board[ny, nx] == player
                    if not end1 and not end2:
                        return True
    return False 

class RenjuGame(BaseGoBang):
    def __init__(self, n=15, nir=5):
        super().__init__(n, nir)

        self._valid_cache = {}
        self._valid_order = deque()
        self._valid_cap = 50000
        self._ended_cache = {}
        self._ended_order = deque()
        self._ended_cap = 50000

    def _cache_key(self, board, player):
        b = np.ascontiguousarray(board)
        return (b.tobytes(), int(player))
    
    def _cache_put_valid(self, key, val_arr):
        if key in self._valid_cache:
            return
        if len(self._valid_order) >= self._valid_cap:
            old = self._valid_order.popleft()
            self._valid_cache.pop(old, None)
        self._valid_cache[key] = val_arr.astype(np.int8, copy=True)
        self._valid_order.append(key)

    def _cache_get_valid(self, key):
        v = self._valid_cache.get(key)
        if v is None:
            return None
        try:
            self._valid_order.remove(key)
        except ValueError:
            pass
        self._valid_order.append(key)
        return v.copy()

    def _cache_put_ended(self, key, val):
        if key in self._ended_cache:
            return
        if len(self._ended_order) >= self._ended_cap:
            old = self._ended_order.popleft()
            self._ended_cache.pop(old, None)
        self._ended_cache[key] = float(val)
        self._ended_order.append(key)

    def _cache_get_ended(self, key):
        v = self._ended_cache.get(key)
        if v is None:
            return None
        try:
            self._ended_order.remove(key)
        except ValueError:
            pass
        self._ended_order.append(key)
        return float(v)
        
    def getValidMoves(self, board, player):
        key = self._cache_key(board, player)
        cached = self._cache_get_valid(key)

        if cached is not None:
            return cached
        
        valids = (board.flatten() == 0).astype(np.int8)
        action_size = self.getActionSize()
        if len(valids) < action_size:
            valids = np.concatenate([valids, np.zeros(action_size - len(valids), np.int8)])
        
        # 지금이 누구 차례
        stones = int(np.count_nonzero(board))  # 돌 개수
        black_to_move = (stones % 2 == 0)      # 짝수 → 흑 차례

        idxs = np.flatnonzero(valids[:-1])
        n = board.shape[0]

        if stones == 0:
            kept = np.array([(n//2) * n + (n//2)])
        else:
            dyn_radius = 2 if stones < 15 // 2 else 3
            kept  = filter_near_stones(board, idxs, radius=dyn_radius)

        narrowed = np.zeros_like(valids)
        if len(kept) > 0:
                narrowed[kept] = 1
        else:
            c = (n//2) * n + (n//2); narrowed[c] = 1
        valids = narrowed

    
        if black_to_move:
            b8 = board.astype(np.int8, copy=True)
            for idx in kept:
                y, x = divmod(idx, n)

                if is_forbidden_nb(b8, x, y, n, stone=+1):
                    valids[idx] = 0
         
        self._cache_put_valid(key, valids)
        return valids

    def getGameEnded(self, board, player):
        key = self._cache_key(board, player)
        cv = self._cache_get_ended(key)
        if cv is not None:
            return cv

        if self._has_exactly_five(board, 1):
            self._cache_put_ended(key, 1.0)
            return 1
        
        if self._has_exactly_five(board, -1):
            self._cache_put_ended(key, -1.0)
            return -1

        if not np.any(board == 0):
            self._cache_put_ended(key, 1e-4)
            return 1e-4
        
        self._cache_put_ended(key, 0.0)
        return 0
    
    # -------------------- 내부 유틸리티 --------------------
    def _has_exactly_five(self, board, player):
        return has_exactly_five(board, player, self.n_in_row)

'''
def getValidMoves(self, board, player):
        valids = (board.flatten() == 0).astype(np.int8)

        action_size = self.getActionSize()
        if len(valids) < action_size:
            valids = np.concatenate([valids, np.zeros(action_size - len(valids), np.int8)])
        
        # 지금이 누구 차례
        stones = int(np.count_nonzero(board))  # 돌 개수
        black_to_move = (stones % 2 == 0)      # 짝수 → 흑 차례

        if black_to_move:
            idxs = np.flatnonzero(valids[:-1])
            kept = filter_near_stones(board, idxs, radius=3)
            narrowed = np.zeros_like(valids)
            narrowed[kept] = 1

            n = board.shape[0]
            b8 = board.astype(np.int8, copy=True)
            for idx in kept:
                y, x = divmod(idx, n)

                if is_forbidden_nb(b8, x, y, n, stone=+1):
                    valids[idx] = 0
         
        return valids

    def getGameEnded(self, board, player):
        if self._has_exactly_five(board, 1):
            return 1
        
        if self._has_exactly_five(board, -1):
            return -1

        if not np.any(board == 0):
            return 1e-4
        
        return 0
'''



'''
@njit
def count_sliding(pattern: str, text: str) -> int:
    cnt = 0
    plen = len(pattern)
    for i in range(len(text) - plen + 1):
        match = True
        for j in range(plen):
            if text[i+j] != pattern[j]:
                match = False
                break

        if match:
            cnt += 1
    return cnt




dirs = ((1, 0), (0, 1), (1, 1), (1, -1))

@njit
def count_open_three(board, x, y, n):
    """
    (x,y)에 돌을 두었을 때 생기는 '0 -1 -1 -1 0' open-three 패턴의
    서로 다른 방향 개수를 세어 반환.
    """
    cnt = 0
    board[y, x] = -1  # 가상 착수

    # 각 방향마다 9칸(−4..+4) 슬라이딩
    for dx, dy in dirs:
        # k ∈ [−4,4] → line 길이 9, 중앙 인덱스 4가 착수점
        line = np.empty(9, dtype=int8)
        for k in range(-4, 5):
            i = x + k * dx
            j = y + k * dy
            if 0 <= i < n and 0 <= j < n:
                line[k + 4] = board[j, i]
            else:
                line[k + 4] = 2  # 경계 밖 sentinel

        # 5칸 윈도우 슬라이딩 (s=0..4)
        for s in range(5):
            # 반드시 중앙(k=0 → idx 4)이 포함되도록
            if not (s <= 4 <= s + 4):
                continue
            # 정확히 [0, -1, -1, -1, 0] 패턴
            if (line[s]   == 0 and
                line[s+1] == -1 and
                line[s+2] == -1 and
                line[s+3] == -1 and
                line[s+4] == 0):
                cnt += 1
                break  # 이 방향당 한 번만 세기

    board[y, x] = 0  # 복원
    return cnt

@njit
def count_open_four(board, x, y, n):
    """
    (x,y)에 돌을 두었을 때 생기는 '0 -1 -1 -1 -1 0' open-four 패턴의
    서로 다른 방향 개수를 세어 반환.
    """
    cnt = 0
    board[y, x] = -1  # 가상 착수

    # 각 방향마다 11칸(−5..+5) 슬라이딩
    for dx, dy in dirs:
        line = np.empty(11, dtype=int8)
        for k in range(-5, 6):
            i = x + k * dx
            j = y + k * dy
            if 0 <= i < n and 0 <= j < n:
                line[k + 5] = board[j, i]
            else:
                line[k + 5] = 2  # 경계 밖 sentinel

        # 6칸 윈도우 슬라이딩 (s=0..5)
        for s in range(6):
            # 반드시 중앙(k=0 → idx 5)이 포함되도록
            if not (s <= 5 <= s + 5):
                continue
            # 정확히 [0, -1, -1, -1, -1, 0] 패턴
            if (line[s]   == 0 and
                line[s+1] == -1 and
                line[s+2] == -1 and
                line[s+3] == -1 and
                line[s+4] == -1 and
                line[s+5] == 0):
                cnt += 1
                break  # 이 방향당 한 번만 세기

    board[y, x] = 0  # 복원
    return cnt

@njit
def is_forbidden(board, x, y, n):
    # 착수 가능한 빈 칸이 아닐 때도 금수 처리
    if board[y, x] != 0:
        return True

    board[y, x] = -1  # 가상 착수

    # 1) 장목(overline) 검사: 동일 방향 돌 개수 ≥ 6
    for dx, dy in dirs:
        c = 1
        # 앞으로
        i, j = x + dx, y + dy
        while 0 <= i < n and 0 <= j < n and board[j, i] == -1:
            c += 1
            i += dx; j += dy
        # 뒤로
        i, j = x - dx, y - dy
        while 0 <= i < n and 0 <= j < n and board[j, i] == -1:
            c += 1
            i -= dx; j -= dy
        if c >= 6:
            board[y, x] = 0
            return True

    # 2) 이중 삼삼
    if count_open_three(board, x, y, n) >= 2:
        board[y, x] = 0
        return True

    # 3) 이중 사사
    if count_open_four(board, x, y, n) >= 2:
        board[y, x] = 0
        return True

    board[y, x] = 0  # 복원

    return False

def _is_forbidden(self, x, y, board):
        b = board.copy()
        b[y][x] = -1

        overline = False
        open_three = open_four = 0

        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            cnt = 1

            for sign in (1, -1):
                nx, ny = x, y

                while 0 <= nx + dx*sign < self.n and 0 <= ny + dy*sign < self.n and b[ny + dy*sign][nx + dx*sign] == -1:
                    cnt += 1
                    nx += dx*sign; ny += dy*sign

            if cnt >= 6:
                overline = True
                
            line = ''
            for t in range(-self.n, self.n+1):
                nx, ny = x + dx*t, y + dy*t

                if 0 <= nx < self.n and 0 <= ny < self.n:
                    line += 'X' if b[ny][nx] == -1 else ('.' if b[ny][nx] == 0 else 'O')

            open_three += self._count_overlapping(r'\.XXX\.', line)
            open_four  += self._count_overlapping(r'\.XXXX\.', line)

        return overline or open_three >= 2 or open_four >= 2
    

    def _has_exactly_five(self, board, player):
        return has_exactly_five(board, player, self.n_in_row)

        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        win_n = self.n_in_row

        for y in range(self.n):
            for x in range(self.n):
                if board[y, x] != player:
                    continue

                for dx, dy in dirs:
                    cnt = 1
                    # 앞으로 세기
                    nx, ny = x + dx, y + dy
                    while 0 <= nx < self.n and 0 <= ny < self.n and board[ny, nx] == player:
                        cnt += 1
                        nx += dx; ny += dy
                    # 뒤로 세기
                    bx, by = x - dx, y - dy
                    while 0 <= bx < self.n and 0 <= by < self.n and board[by, bx] == player:
                        cnt += 1
                        bx -= dx; by -= dy

                    # 정확히 n개 연속인지, 양끝이 끊겼는지 검사
                    if cnt == win_n:
                        # '앞끝'이 연속이었는지
                        end1_blocked = (0 <= bx < self.n and 0 <= by < self.n and board[by, bx] == player)
                        # '뒷끝'이 연속이었는지
                        end2_blocked = (0 <= nx < self.n and 0 <= ny < self.n and board[ny, nx] == player)
                        if not end1_blocked and not end2_blocked:
                            return True
        return False


    def _has_five(self, board, player):
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        win_n = self.n_in_row

        for y in range(self.n):
            for x in range(self.n):
                if board[y, x] != player:
                    continue

                for dx, dy in dirs:
                    cnt = 1
                    nx, ny = x + dx, y + dy
                    while 0 <= nx < self.n and 0 <= ny < self.n and board[ny, nx] == player:
                        cnt += 1
                        nx += dx; ny += dy

                    if cnt >= win_n:
                        return True
        return False

    def _count_overlapping(self, pattern, text):
        return count_sliding(pattern.strip('\\.'), text)
        #return sum(1 for _ in re.finditer(f'(?=({pattern}))', text))

'''