# Blockchain Failure Predictor (BFP)

고성능컴퓨팅 지원사업을 통해 개발한 블록체인 노드 장애 예측 데모 레포지토리입니다.
KT Cloud H200 GPU 4장 환경에서 시계열 모델을 학습하고, 예측 결과에 따라 자동 대응
플레이북을 실행하는 예제를 제공합니다.

## 주요 기능

- CSV 기반 시계열 피처 로딩 (`bfp.data.LogDataset`)
- GRU 기반 장애 예측 모델 (`bfp.model.FailurePredictorModel`)
- H200 GPU 4장 기준 분산 학습(DDP) + AMP (`bfp.train_ddp`)
- 단일 CSV에 대한 추론 스크립트 (`bfp.predict`)
- 예측 점수 기반 자동 대응 예제 (`bfp.auto_responder`, `examples/sample_playbook.yaml`)

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 학습 (분산 학습 예시)

```bash
python -m bfp.train_ddp   --train-csv data/train_logs.csv   --valid-csv data/valid_logs.csv   --sequence-length 60   --batch-size 256   --epochs 20   --hidden-dim 256   --num-layers 4   --output-dir checkpoints_h200
```

## 추론 예시

```bash
python -m bfp.predict   --checkpoint checkpoints_h200/best_model_ddp.pt   --input-csv data/valid_logs.csv   --sequence-length 60
```

## 자동 대응 예시

```bash
python -m bfp.auto_responder   --threshold 0.8   --score 0.92   --playbook examples/sample_playbook.yaml
```

## 라이선스

본 레포지토리 코드는 Apache License 2.0 기준으로 공개하는 것을 권장합니다.
실제 깃허브에 올릴 때는 Apache 2.0 본문을 포함한 `LICENSE` 파일을 추가해 주세요.
