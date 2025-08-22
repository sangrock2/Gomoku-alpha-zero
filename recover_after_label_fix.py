# recover_after_label_fix.py
import os, glob, torch
from RenjuGame import RenjuGame
from alpha_zero_general.gobang.keras.NNet import NNetWrapper as NNet
import torch.nn as nn

CHECKPOINT_DIR = "./models"      # <- 네 경로로 바꿔도 됨
LOAD_FILE      = "best.pth.tar"  # 혹은 temp.pth.tar
SAVE_FILE      = "best.pth.tar"  # 덮어쓰기 싫으면 "best_recovered.pth.tar"
BOARD_SIZE     = 15              # n

def delete_example_files(folder):
    cnt = 0
    for fp in glob.glob(os.path.join(folder, "*.examples")):
        try:
            os.remove(fp); cnt += 1
        except Exception as e:
            print("[warn] remove failed:", fp, e)
    print(f"[recovery] deleted {cnt} example files under {folder}")

def reset_value_head(nnet_module: nn.Module):
    """
    정책(π)는 보존하고 가치(v) 헤드만 재초기화.
    보편적인 이름(v_conv/v_bn/v_fc1/v_fc2/v_reduce) 우선, 실패시 fallback 검색.
    """
    names = ["v_conv", "v_bn", "v_fc1", "v_fc2", "v_reduce"]
    hit = False
    for name in names:
        if hasattr(nnet_module, name):
            layer = getattr(nnet_module, name)
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            if hasattr(layer, "reset_running_stats"):
                layer.reset_running_stats()
            hit = True
    if not hit:
        # fallback: value head로 추정되는 단일 출력 모듈/BN 초기화
        for mod in nnet_module.modules():
            cls = mod.__class__.__name__
            if isinstance(mod, (nn.BatchNorm2d,)) and hasattr(mod, "reset_running_stats"):
                mod.reset_running_stats()
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "reset_parameters"):
                # out_channels==1 (conv) or out_features==1 (linear)를 value 헤드로 간주
                if (hasattr(mod, "out_channels") and getattr(mod, "out_channels") == 1) or \
                   (hasattr(mod, "out_features") and getattr(mod, "out_features") == 1):
                    mod.reset_parameters()
                    hit = True
    print("[recovery] value head reset:", hit)

def main():
    # 1) 예제 파일(.examples) 모두 삭제
    delete_example_files(CHECKPOINT_DIR)

    # 2) 모델 로드
    game = RenjuGame(n=BOARD_SIZE)
    nnet = NNet(game)
    nnet.load_checkpoint(CHECKPOINT_DIR + ("" if CHECKPOINT_DIR.endswith(os.sep) else os.sep), LOAD_FILE)
    print("[recovery] loaded checkpoint:", os.path.join(CHECKPOINT_DIR, LOAD_FILE))

    # 3) value 헤드만 재초기화
    reset_value_head(nnet.nnet)

    # 4) 옵티마이저 리셋 (Adam 모멘텀 초기화)
    lr = getattr(nnet.nnet.args, "lr", 0.001)
    l2 = getattr(nnet.nnet.args, "l2",  0.0001)
    nnet.optimizer = torch.optim.Adam(nnet.nnet.parameters(), lr=lr, weight_decay=l2)
    print(f"[recovery] optimizer reset (Adam lr={lr}, wd={l2})")

    # 5) 복구된 모델 저장
    nnet.save_checkpoint(CHECKPOINT_DIR, SAVE_FILE)
    print("[recovery] saved checkpoint:", os.path.join(CHECKPOINT_DIR, SAVE_FILE))

    # 6) 위생 체크(선택): 빈판 value가 0 근처인지 확인
    import numpy as np
    pi, v = nnet.predict(np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=int))
   
if __name__ == "__main__":
    main()
