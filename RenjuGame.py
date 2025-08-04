import numpy as np
from gobang.GobangGame import GobangGame as BaseGoBang

class RenjuGame(BaseGoBang):
    def __init__(self, n=15, nir=5):
        super().__init__(n, nir)
        
    def getValidMoves(self, board, player):
        valids = super().getValidMoves(board, player)

        if player == -1:
            for idx, v in enumerate(valids):
                if v == 1:
                    y, x = divmod(idx, self.n)
                    
                    if self._is_forbidden(x, y, board):
                        valids[idx] = 0
        return valids
    
    def getGameEnded(self, board, player):
        if self._has_exactly_five(board, player):
            return 1
        
        if self._has_five(board, -player):
            return -1
        
        if not np.any(board == 0):
            return 1e-4
        
        return 0
    
    
    # -------------------- 내부 유틸리티 --------------------
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

            open_three += line.count('.XXX.')
            open_four  += line.count('.XXXX.')

        return overline or open_three >= 2 or open_four >= 2
    
    def _has_exactly_five(self, board, player):
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