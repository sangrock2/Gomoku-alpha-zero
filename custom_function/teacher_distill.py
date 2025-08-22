# teacher_distill.py
import math, numpy as np
from alpha_zero_general.MCTS import MCTS

EPS = 1e-12

def _entropy(pi: np.ndarray) -> float:
    p = np.clip(pi.astype(np.float64), EPS, 1.0)
    return float(-(p * np.log(p)).sum())

def _board_key(game, board) -> str:
    # 게임에 이미 stringRepresentation이 있으면 그걸 쓰는 게 제일 안전
    if hasattr(game, "stringRepresentation"):
        return game.stringRepresentation(board)
    # 없으면 bytes로
    return board.tobytes()

def make_teacher_targets(
    game,
    nnet,
    examples,                   # list of (state, pi, z)  또는 (state, cur, pi, _)
    base_args=None,
    max_positions=512,          # 교사 재탐색할 포지션 수(작게 시작)
    numMCTSSims=2000,           # 교사 MCTS 심도
    sample_ratio=None,          # None 이면 max_positions 사용, 아니면 비율(예: 0.1)
    entropy_topk=True,          # 엔트로피 높은 포지션 우선
    min_entropy=None,           # 엔트로피 하한(예: 3.5)
    eval_mode=True              # 교사 MCTS는 평가 모드 권장
):
    """
    반환: dict[key -> pi_teacher(np.ndarray)]
    key는 stringRepresentation(board).
    """
    # 1) (state, pi, z) 형태로 정규화
    normed = []
    for ex in examples:
        if len(ex) == 3:
            s, pi, _ = ex
        elif len(ex) == 4:
            s, _, pi, _ = ex
        else:
            raise ValueError("example format must be (s, pi, z) or (s, cur, pi, _)")
        normed.append((s, pi))

    # 2) 후보 점수화(엔트로피)
    scored = []
    for s, pi in normed:
        H = _entropy(np.asarray(pi))
        scored.append((H, s))

    # 3) 샘플 선택
    if sample_ratio is not None:
        k = max(1, int(len(scored) * float(sample_ratio)))
    else:
        k = int(max_positions)

    if entropy_topk:
        scored.sort(key=lambda x: x[0], reverse=True)  # 높은 H 우선
        candidates = [s for (H, s) in scored if (min_entropy is None or H >= min_entropy)]
        candidates = candidates[:k]
    else:
        # 무작위 샘플(하한 조건 적용)
        pool = [s for (H, s) in scored if (min_entropy is None or H >= min_entropy)]
        np.random.shuffle(pool)
        candidates = pool[:k]

    # 4) 고심도 MCTS로 π* 생성
    #    평가용 args 얕게 복제
    class EvalArgs: pass
    ea = EvalArgs()

    if not hasattr(ea, "cpuct"):         # ✅ 에러 원인
        ea.cpuct = 1.0
    # 기본값들
    for k_attr in getattr(nnet, "args", {}):
        setattr(ea, k_attr, getattr(nnet.args, k_attr))
    # 변경: 심도만 올리고 temp/노이즈는 평가 모드에 따르게
    ea.numMCTSSims = int(numMCTSSims)

    tmcts = MCTS(game, nnet, ea, eval_mode=eval_mode)

    targets = {}
    for s in candidates:
        # temp=1로 “분포” 그 자체를 받음
        pi_star = np.asarray(tmcts.getActionProb(s, temp=1), dtype=np.float32)
        # 안전 정규화
        psum = float(pi_star.sum())
        if psum <= 0:
            # 비상시 균등(valid만) 사용
            valids = np.asarray(game.getValidMoves(s, 1), dtype=np.float32)
            pi_star = valids / max(valids.sum(), 1.0)
        else:
            pi_star /= psum
        targets[_board_key(game, s)] = pi_star
    return targets
