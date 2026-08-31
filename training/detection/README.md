# Multi-instance detection and batch inference

`synthetic-v4-conveyor`는 한 장에 5개 부품이 있는 검정 belt 장면을 위한 train-only detection/segmentation 자료입니다. 전체 장면을 처리하려면 기존 classifier와 별도의 detector 또는 segmenter를 학습해야 합니다.

## 권장 pipeline

```text
여러 입력 사진
  -> full-scene component/status detector 또는 instance segmenter
  -> component별 bbox/mask와 confidence
  -> 선택: 검출 crop에 defect classifier/segmenter 적용
  -> 사진별·부품별 결과와 review threshold 저장
```

첫 단계는 8개 component status 또는 class-agnostic component detection으로 구성할 수 있습니다. 결함 위치까지 필요하면 7-class defect detection/semantic segmentation head를 함께 학습합니다. `normal_proxy`에는 defect target이 없습니다.

Annotation은 task별로 분리되어 있습니다.

- Component status: `annotations/coco/component_status_train.json` 또는 `labels/yolo_component_status/train`
- Defect localization: `annotations/coco/defects_train.json`, `labels/yolo_defects/train`, `masks/defect_semantic/train`
- Instance segmentation 보조: `masks/component_visible_instances/train`

COCO/PNG mask를 canonical annotation으로 사용하고 YOLO bbox는 derived export로 취급합니다. Component status와 defect localization의 class ID namespace를 섞지 않습니다.

## 기존 checkpoint와의 호환성

`training/results/final-stratified-seed-*`의 `model_final.pt` 3개는 `synthetic-v2-700` 단일 부품 crop으로 학습한 7-class ResNet-18 classifier입니다.

- 전체 장면에서 여러 component를 찾아내지 못합니다.
- `normal_proxy` 또는 실제 `OK` class가 없습니다.
- v3/v4 condition data로 재학습한 checkpoint가 아닙니다.
- v4 scene 또는 실제 conveyor domain 성능이 검증되지 않았습니다.

따라서 이 checkpoint를 full-scene detector처럼 사용하면 안 됩니다. Detector가 만든 crop의 보조 분류기로 시험할 수는 있지만, 그 사용도 v4와 독립 실물 test에서 별도 검증해야 합니다.

## 여러 사진 일괄 검증

다음 명령은 v4 release의 384개 synthetic image와 모든 label/mask를 한 번에 replay 검사합니다.

```powershell
py -3.14 -B scripts\validate_synthetic_v4_conveyor.py
```

이는 dataset integrity validator입니다. 폴더에 넣은 새 사진을 자동 판독하는 inference CLI는 아니며, repository에는 아직 v4로 학습된 detector checkpoint나 실물 성능 결과가 없습니다.

실제 batch inference 구현 시에는 최소한 다음 출력이 필요합니다.

- 입력 image ID와 image-level 처리 상태
- component별 bbox/mask, predicted status, confidence
- defect별 bbox/mask, class, confidence
- confidence 미달 또는 class 충돌에 대한 `REVIEW/HOLD`
- 사용 model hash, threshold, preprocessing와 실행 시각

## 평가 조건

v4 전체는 하나의 synthetic-restored base family에서 파생되어 `TRAIN_ONLY / evaluation_eligible=NO`입니다. v4 내부 random split score는 독립 실물 성능이 아닙니다. 성능 보고에는 실제 normal과 실제 결함을 포함한 독립 specimen 단위 locked test set이 필요하며, 최소한 class별 precision/recall/AP, false accept/reject, 미검출률, 장면당 latency를 따로 보고해야 합니다.

`normal_proxy`는 paired-clean 합성본이지 `OK_confirmed`가 아닙니다. Detector가 `normal_proxy`를 예측해도 실제 정상품, 전기적 정상 또는 출하 합격을 보증하지 않습니다.

또한 v4는 장면마다 서로 다른 status 5개를 배치한 balanced pilot입니다. 동일 결함 반복이나 정상 다수의 실제 생산 분포는 포함하지 않으므로, 배포용 학습에는 별도 `production_mix` 장면과 독립 실물 test가 필요합니다.

이 자료와 문서는 공개 재사용 라이선스가 없는 All Rights Reserved 자료입니다. 사전 서면 허가 없는 AI/ML 학습·평가·fine-tuning을 포함한 재사용은 허가되지 않습니다. 자세한 조건은 [LICENSE_STATUS.md](../../LICENSE_STATUS.md)를 따릅니다.
