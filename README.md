# Blockchain Failure Predictor (BFP)

고성능컴퓨팅 지원사업을 통해 개발한 블록체인 노드 장애 예측 데모 레포지토리입니다.
KT Cloud **H200 GPU 4장** 환경에서 시계열 모델을 학습하고, 예측 결과에 따라 자동 대응
플레이북을 실행하는 예제를 제공합니다.

## 주요 기능
- CSV 기반 시계열 피처 로딩 (`bfp.data.LogDataset`)
- GRU 기반 장애 예측 모델 (`bfp.model.FailurePredictorModel`)
- **H200 GPU 4장 기준 분산 학습(DDP) + AMP** (`bfp.train_ddp`)
- 단일 CSV에 대한 추론 스크립트 (`bfp.predict`)
- 예측 점수 기반 자동 대응 예제 (`bfp.auto_responder`, `examples/sample_playbook.yaml`)

## 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 학습 (분산 학습 예시: H200 x 4)
```bash
python -m bfp.train_ddp   --train-csv data/train_logs.csv   --valid-csv data/valid_logs.csv   --sequence-length 60   --batch-size 256   --epochs 20   --hidden-dim 256   --num-layers 4   --output-dir checkpoints_h200
```

## 추론 예시
```bash
python -m bfp.predict   --checkpoint weights/best_model_ddp.pt   --input-csv data/valid_logs.csv   --sequence-length 60
```

## Weights (Checkpoint)
- 공개 가중치(체크포인트): `weights/best_model_ddp.pt`  
  (데모 용도로 제공되는 경량 체크포인트입니다. 실제 운영용 가중치는 고객/보안 요건에 따라 별도 관리합니다.)

## Release (선택)
Releases를 사용할 경우, `best_model_ddp.pt`를 Assets로 첨부해도 됩니다. (예: `v0.1-demo`).

## 라이선스
코드는 Apache License 2.0을 따릅니다. (본 레포지토리의 `LICENSE` 파일 참조)
