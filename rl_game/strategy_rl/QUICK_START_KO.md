# VEX U Override RL 코드 설치 · 실행 안내

## 1. 압축 풀기

기존 VSCode에서 `strategy-sim` 폴더를 열고, 다운로드한 `VEX_Strategy_RL_Full.zip`을 `strategy-sim` 폴더에 놓으세요.

터미널 위치가 아래처럼 나와야 합니다.

```
(base) PS C:\Users\...\Documents\strategy-sim>
```

전체 명령어를 복붙하면 됩니다.

```powershell
$ErrorActionPreference = "Stop"
Expand-Archive -Path ".\VEX_Strategy_RL_Full.zip" -DestinationPath "." -Force
cd ".\rl_game\strategy_rl"
python -m pip install -r requirements.txt
python run.py test
python run.py smoke
python run.py baselines --episodes 30 --opponent greedy
python run.py curriculum --steps-per-stage 20000 --eval-episodes 20 --out-dir outputs/curriculum
```

- 새 코드 위치: `strategy-sim/rl_game/strategy_rl/`
- 기존 `v1`, `agents`, `env`, `training` 폴더는 **절대 삭제하지 않습니다**.
- 기존 `VEX_RL_Complete.zip`을 따로 적용할 필요 없습니다. 이번 버전이 새 독립 구현입니다.

## 2. 첫 학습은 소규모로 검증하기

오래 학습시키기 전에 `python run.py curriculum --steps-per-stage 1000 --eval-episodes 3 --bc-episodes 2 --out-dir outputs/quickcheck` 로 먼저 실행해도 됩니다. 이 단계의 모델 성능은 비교할 만한 수준으로 학습되었다는 뜻이 아닙니다.

## 3. 학습 결과 평가

```powershell
python run.py eval --policy ppo --model outputs/curriculum/ppo_selfplay.pt --opponent greedy --episodes 50 --csv outputs/eval_greedy.csv
python run.py eval --policy coordination --opponent greedy --episodes 50 --csv outputs/eval_coordination.csv
python run.py plot --histories outputs/curriculum/history_no_opponent.csv outputs/curriculum/history_greedy.csv outputs/curriculum/history_mixed.csv outputs/curriculum/history_selfplay.csv --out outputs/training_curve.png
```

학습 결과가 저장되는 위치: `strategy_rl/outputs/`. 모델은 `ppo_no_opponent.pt`, `ppo_greedy.pt`, `ppo_mixed.pt`, `ppo_selfplay.pt` 입니다.

## 4. Action 소요시간 입력하는 방법

`config.json`을 열고 `robot_speed_units_per_second` 및 `action_seconds`를 수정하면 됩니다. 초(s) 단위입니다. PPO 네트워크 및 훈련 알고리즘은 건드릴 필요 없습니다.

## 5. 주의할 점

이는 **실제 VEX 공식 게임의 정밀 물리 시뮬레이터가 아닙니다.** 공식 VEX U 경기 시간과 득점 체계의 핵심 개념을 반영한 고수준 학습용 환경입니다. 정확한 Goal 모양·충돌·Pin/Cup 겹침·AWP 심판 판정 등은 단순화되어 있습니다. 팀원의 GameEnv가 완료되면 `TEAMMATE_INTERFACE.md`대로 연결하고 재학습/평가하세요.

도움말: `python run.py --help`. 전체 설명: `README.md`. 상세 가정: `GAME_ASSUMPTIONS.md`.
