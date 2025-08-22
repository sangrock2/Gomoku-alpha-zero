import numpy as np

try:
    from numba import njit
    _njit = njit(cache=True, fastmath=True)
except Exception:
    _njit = lambda f: f

@_njit
def find_immediate_win(board, valids, n, player=1, require_exact_five=False):
    size = n * n
    L = len(valids)
    upto = size if L >= size else L
    out = np.zeros(L, dtype=np.int8)

    for idx in range(upto):
        if valids[idx] == 0:
            continue

        y, x = divmod(idx, n)
        if board[y, x] != 0:
            continue

        # 네 방향(가로, 세로, 두 대각)
        for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
            cnt = 1  

            tx, ty = x + dx, y + dy
            while 0 <= tx < n and 0 <= ty < n and board[ty, tx] == player:
                cnt += 1; tx += dx; ty += dy
            # -방향
            tx, ty = x - dx, y - dy
            while 0 <= tx < n and 0 <= ty < n and board[ty, tx] == player:
                cnt += 1; tx -= dx; ty -= dy

            if cnt >= 5:
                # (옵션) Renju 흑의 장목 금수 같은 걸 여기서 배제하고 싶다면 True
                # 보통은 금수는 getValidMoves에서 이미 걸러집니다.
                if require_exact_five and cnt != 5:
                    continue
                out[idx] = 1
                break
    return out