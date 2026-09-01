# Multi-instance detection and batch inference

`synthetic-v4-conveyor`와 `synthetic-v5-illumination`은 한 장에 5개 부품이 있는 검정 belt 장면을 위한 train-only detection/segmentation 자료입니다. v5는 v4의 384개 composition을 두 조명 조건씩 replay한 768개 auxiliary scene이며 새 독립 specimen은 아닙니다. 전체 장면을 처리하려면 기존 classifier와 별도의 detector 또는 segmenter를 학습해야 합니다.

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

- v4 component status: `synthetic/v4_conveyor/annotations/coco/component_status_train.json` 또는 `synthetic/v4_conveyor/labels/yolo_component_status/train`
- v4 defect localization: `synthetic/v4_conveyor/annotations/coco/defects_train.json`, `synthetic/v4_conveyor/labels/yolo_defects/train`, `synthetic/v4_conveyor/masks/defect_semantic/train`
- v5는 동일한 release-relative 구조를 `synthetic/v5_illumination` 아래에 제공하며 `masks/shadow_attenuation/train`과 조명 metadata도 추가합니다.
- Instance segmentation 보조: 각 release의 `masks/component_visible_instances/train`

COCO/PNG mask를 canonical annotation으로 사용하고 YOLO bbox는 derived export로 취급합니다. Component status와 defect localization의 class ID namespace를 섞지 않습니다.

## 기존 checkpoint와의 호환성

`training/results/final-stratified-seed-*`의 C0 checkpoint 3개와
`training/results/v3-conditions-seed-*`의 C2 checkpoint 3개는
`synthetic-v2-700` 단일 부품 crop을 판정하는 7-class ResNet-18
classifier입니다. C2는 gradient-train parent의 v3 조명·카메라 variant를
추가로 사용했지만 detector는 아닙니다.

- 전체 장면에서 여러 component를 찾아내지 못합니다.
- `normal_proxy` 또는 실제 `OK` class가 없습니다.
- C0는 v3 condition을 사용하지 않았고 C2는 v3만 사용했습니다. 둘 다
  v4/v5 full-scene data로 학습한 checkpoint가 아닙니다.
- v4/v5 scene 또는 실제 conveyor domain 성능이 검증되지 않았습니다.

따라서 이 checkpoint를 full-scene detector처럼 사용하면 안 됩니다. Detector가 만든 crop의 보조 분류기로 시험할 수는 있지만, 그 사용도 v4/v5와 독립 실물 test에서 별도 검증해야 합니다.

## 여러 사진 일괄 검증

다음 명령은 v4의 384개 image와 v5의 768개 image, 관련 label/mask를 release별로 replay 검사합니다.

```powershell
py -3.14 -B scripts\validate_synthetic_v4_conveyor.py
py -3.14 -B scripts\validate_synthetic_v5_illumination.py
```

이는 dataset integrity validator입니다. 폴더에 넣은 새 사진을 자동 판독하는 inference CLI는 아니며, repository에는 아직 v4/v5로 학습된 detector checkpoint나 실물 성능 결과가 없습니다.

## 즉시 실행 가능한 transfer learning baseline

`training/scripts/train_eval_detector.py`는 `FasterRCNN-MobileNetV3-Large-FPN`의 공식 COCO weight를 시작점으로 사용하는 v4/v5 component detector baseline입니다. 기본 task는 status와 무관하게 모든 부품을 하나의 foreground로 찾는 `component_localization`(`background=0`, `component=1`)입니다. 선택적으로 `--task component_status`를 지정하면 별도 8-class component-status namespace(`background + 8 status`)를 사용합니다. 두 mode 모두 `defects_train.json`을 읽지 않으며 component-status ID와 defect ID를 혼용하지 않습니다.

일반 Python 설치에는 이 repository에서 사용한 `torch`/`torchvision`이 없을 수 있습니다. Codex가 표시한 bundled workspace Python 실행 파일을 shell 변수로 지정해 실행합니다. 사용자별 절대 경로는 문서나 config에 기록하지 않습니다.

```powershell
$CODEX_PYTHON = "<Codex workspace dependencies에 표시된 bundled Python executable>"

& $CODEX_PYTHON -B training/scripts/train_eval_detector.py --check-only

& $CODEX_PYTHON -B training/scripts/train_eval_detector.py `
  --smoke `
  --device auto `
  --output training/results/component-detector-smoke-run-01

& $CODEX_PYTHON -B training/scripts/train_eval_detector.py `
  --task component_status `
  --device auto `
  --output training/results/component-status-fixed-run-01
```

Preflight는 다음을 hard gate로 검사합니다.

- v4 384장, v5 768장, 장면당 5개 component와 canonical 8-class category
- manifest↔COCO의 release-relative path, scene/image ID, 크기, bbox, label, 실제 image SHA-256
- config에 고정된 manifest, COCO, release metadata, 전체 image-content fingerprint, authoritative v2 parent catalog와 split-assignment digest
- v4/v5가 참조하는 unique parent가 정확히 168개이고 모두 `model_split=gradient_train`이며 validation/test parent가 0개인지 확인
- 재시작하는 COCO ID를 `(release_key, image_id)`로 분리하고 384개 ID 충돌을 의도적으로 확인
- 384개 composition family마다 `v4 1장 + v5 variant 2장`
- 전체 source-parent graph가 단일 연결이며 모든 row가 `TRAIN_ONLY`, `evaluation_eligible=NO`
- local torchvision cache의 `fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`가 SHA-256 `fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533`와 일치

Weight는 검증된 local cache에서 직접 읽으며 HTTP/network download를 호출하지 않습니다. 파일이 없거나 hash가 다르면 실행을 중단합니다. `TORCH_HOME` 또는 `XDG_CACHE_HOME`을 별도로 지정할 때도 UNC/network share가 아닌 local disk cache를 사용해야 합니다.

학습은 bbox-aware `torchvision.transforms.v2`의 약한 horizontal flip, affine, color jitter, Gaussian blur만 train input에 적용합니다. `RandomErasing`, `CutMix`, `MixUp`, `Mosaic`는 금지되어 있습니다. `FamilyVariantSampler`는 epoch마다 각 family에서 v4/v5 중 정확히 한 장만 선택하고 세 variant를 순환시켜 파생 조명 장면의 과표집을 막습니다. 기본 batch size는 1이며 CUDA에서는 AMP를 사용합니다.

`--smoke`는 최대 8개의 **실제 finite optimizer update**만 수행하고 checkpoint, `run_metadata.json`, `training_history.json`, 같은 training sample의 diagnostic overlay를 생성합니다. AMP overflow로 gradient norm이 non-finite이면 해당 optimizer update를 건너뛰고 scale을 낮춘 뒤 같은 batch를 재시도하며, 허용 횟수를 넘으면 실패합니다. 각 update의 loss·gradient norm과 model parameter/buffer를 finite 검사하고 JSON은 `NaN`/`Infinity`를 거부합니다. full run도 fixed epoch만 사용하며 validation/early stopping/model selection을 수행하지 않습니다. 모든 loss와 sample prediction은 `TRAIN_DIAGNOSTIC_ONLY`이고 실제 검출 성능 주장이 아닙니다. Config, pipeline code, manifest/COCO/image fingerprint, official weight의 hash와 seed, runtime environment metadata가 결과에 함께 기록됩니다.

실제 batch inference 구현 시에는 최소한 다음 출력이 필요합니다.

- 입력 image ID와 image-level 처리 상태
- component별 bbox/mask, predicted status, confidence
- defect별 bbox/mask, class, confidence
- confidence 미달 또는 class 충돌에 대한 `REVIEW/HOLD`
- 사용 model hash, threshold, preprocessing와 실행 시각

## v5 조명 auxiliary 사용 규칙

v5의 4개 multi-light rig는 한 화면에 2–3개 광원을 동시에 합성하고, `P0`–`P5` proxy와 두 shadow regime을 조합합니다. 양의 조명 변화는 component에만 적용하며 검정 belt에는 contact/directional shadow가 나타납니다.

`capture_plan_target_lux` 50/100/200/400/800/1600은 미래 실물 촬영 계획의 목표점입니다. 합성 image는 `measured_illuminance_lux=null`, `absolute_lux_eligible=NO`이므로 proxy별 검출률을 실제 lux별 검출률로 보고하면 안 됩니다.

동일 v4 composition에서 나온 v5 두 variant는 같은 `composition_family_id`를 공유합니다. v4 source와 두 variant를 하나의 train family로 묶고, family 단위 sampler를 사용해 같은 composition이 과도하게 반복되지 않도록 합니다. 이 release는 `classification_eligible=NO`이므로 full scene을 기존 single-component classifier manifest에 넣지 않습니다.

실제 lux·다각도 조명·그림자 검증용 촬영은 [실물 조명 촬영 프로토콜](../../docs/REAL_LIGHTING_CAPTURE_PROTOCOL.md)과 [capture manifest template](../../annotations/real_lighting_capture_template.csv)을 사용합니다. 현재 해당 프로토콜로 추가된 실제 이미지는 0장입니다.

## 평가 조건

v4/v5 전체는 하나의 synthetic-restored base family에서 파생되어 `TRAIN_ONLY / evaluation_eligible=NO`입니다. v4/v5 내부 random split score는 독립 실물 성능이 아닙니다. 성능 보고에는 실제 normal과 실제 결함을 포함한 독립 specimen 단위 locked test set이 필요하며, 최소한 class별 precision/recall/AP, false accept/reject, 미검출률, 조명 조건별 성능과 장면당 latency를 따로 보고해야 합니다.

`normal_proxy`는 paired-clean 합성본이지 `OK_confirmed`가 아닙니다. Detector가 `normal_proxy`를 예측해도 실제 정상품, 전기적 정상 또는 출하 합격을 보증하지 않습니다.

또한 v4/v5는 장면마다 서로 다른 status 5개를 배치한 balanced pilot입니다. 동일 결함 반복이나 정상 다수의 실제 생산 분포는 포함하지 않으므로, 배포용 학습에는 별도 `production_mix` 장면과 독립 실물 test가 필요합니다.

이 자료와 문서는 공개 재사용 라이선스가 없는 All Rights Reserved 자료입니다. 사전 서면 허가 없는 AI/ML 학습·평가·fine-tuning을 포함한 재사용은 허가되지 않습니다. 자세한 조건은 [LICENSE_STATUS.md](../../LICENSE_STATUS.md)를 따릅니다.
