from alpha_zero_general.main import main

if __name__ == '__main__':
    import sys

    sys.argv = [
        'main.py',
        '--game=RenjuGame',    # Game 클래스명을 명시 RenjuGame
        '--nnet=alpha_zero_general.gobang.keras.NNetWrapper'
        '--numIters=100',         # 반복 횟수
        '--numEps=200',           # self-play 에피소드
        '--tempThreshold=15',
        '--updateThreshold=0.55',
        '--maxlenOfQueue=200000',
        '--numMCTSSims=70',
        '--cpuct=1.0',
    ]

    main()