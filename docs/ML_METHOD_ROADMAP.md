# ML method roadmap

이 문서는 현재 자료로 즉시 실행할 수 있는 학습 실험의 순서와 판정 규칙을 고정합니다. 합성 자료는 모델 구현·수렴·조건 강건성 확인을 위한 보조 학습 자료이며, 여기서 얻은 점수는 실제 제품의 검출률이나 출하 판정 성능이 아닙니다.

## 현재 상태

| ID | 상태 | 비고 |
|---|---|---|
| `C0` | 기준 실험 완료 | 기존 ImageNet-pretrained ResNet-18 classifier reference |
| `C2` | 3-seed 완료 | synthetic same-base macro-F1 `0.984178`, mild recall `0.962233`; C0 대비 각각 `+0.013087`, `+0.024631` |
| `C3` | 3-seed 완료·현재 recipe 기각 | C0와 같은 180 updates에서 macro-F1 `0.964601`, mild recall `0.919540` |
| `C4` | soft-voting 완료 | macro-F1 `0.986186`; threshold/`HOLD` calibration은 `NOT VERIFIED` |
| `D0` | v4-only control config 예정 | v5 효과를 분리할 기준 실험; 현재 combined config로는 실행 불가 |
| `D1` | combined pipeline·CUDA smoke PASS | v4/v5 family당 variant 1개 순환; full training/성능 평가는 미실행 |
| `D2` | 예정 | direct 8-class detector와 detector→crop classifier cascade 비교 |

Classifier와 detector는 별도 track으로 진행합니다.

```text
Classifier: C0 reference → C2 → C3 → C4
Detector:   D0 → D1 → D2
최종 선택: 독립 실물 validation에서만 threshold와 배포 후보 결정
```

## 즉시 실험

### C0 — transfer-learning reference

- ImageNet-pretrained ResNet-18 backbone과 고정 ROI를 사용합니다.
- v2 base train만 사용하고 약한 flip·회전·이동을 적용합니다.
- 기존 immutable split, seed, weight hash와 결과물을 비교 기준으로 유지합니다.
- 이 결과는 `synthetic_same_base_sanity_only`이며 실물 일반화 성능이 아닙니다.

### C2 — condition auxiliary

- C0에 v3의 under/overexposure, warm/cool directional light, shadow/vignette, specular/sensor 조건을 train-only auxiliary로 추가합니다.
- v3 variant는 새 specimen이 아니라 기존 train parent의 파생본입니다.
- 세 seed를 동일 split과 설정으로 실행하고 평균·분산을 기록합니다.
- 단순 append는 condition 적용 가능성을 확인하는 기준이며, 파생본 수가 학습 가중치가 되지 않도록 C3와 비교합니다.
- 완료된 C2에서는 macro-F1과 mild recall이 C0보다 높았지만 discoloration recall은 `0.995370 → 0.990741`로 소폭 낮아졌습니다. Seed별 실제 optimizer update는 `1,050 / 840 / 1,134`로 C0/C3의 180보다 크므로 condition 자체의 순효과로 확정하지 않습니다.

### C3 — family-balanced condition sampling

- 먼저 `family_split_id`를 균등 추출한 뒤 해당 parent의 base 또는 v3 condition variant를 선택합니다.
- 같은 parent의 여러 파생본을 서로 독립 표본처럼 한 epoch에 모두 소비하지 않습니다.
- 공정 비교는 epoch 수가 아니라 optimizer update 수, batch size, learning-rate schedule과 seed를 맞춰야 합니다. 이번 C3는 C0와 180 updates를 맞췄지만 C2의 더 큰 budget과는 맞추지 못했습니다.
- class 빈도와 condition 선택 빈도를 각각 기록해 sampler가 의도대로 동작하는지 검사합니다.
- 구현 후 세 seed를 180 updates로 실행한 결과 macro-F1은 `0.964601`, mild recall은 `0.919540`이었습니다. C0보다 각각 `-0.006490`, `-0.018062`, C2보다 `-0.019577`, `-0.042693`입니다.
- 따라서 현재 C3 recipe는 유지 후보에서 제외합니다. 이 결과는 condition diversity를 적은 update budget에 넣으면 학습량이 부족할 수 있음을 보여주지만, C2 개선이 condition 자체 때문인지 더 많은 update 때문인지는 분리하지 못합니다.
- 다음 공정 비교는 C0와 family-balanced condition recipe 모두에 동일한 더 큰 fixed update budget을 적용하는 것입니다.

### C4 — ensemble, calibration, reject option

- 동일 데이터·split·recipe의 세 seed checkpoint에 soft-voting을 적용합니다.
- temperature와 confidence/margin 기반 `HOLD` 기준은 validation에서만 정합니다.
- 실제 `OK_confirmed`가 없으므로 낮은 confidence뿐 아니라 합성 `normal_proxy` 예측도 출하 `PASS`로 변환하지 않습니다.
- ensemble의 정확도 이득과 함께 batch-1 latency, VRAM과 처리량 증가를 보고합니다.
- C2 세 checkpoint의 equal-weight probability soft voting 결과 macro-F1은 `0.986186`으로 단일 모델 평균보다 `+0.002008`이었습니다.
- body-crack과 discoloration recall은 단일 모델 평균보다 각각 `-0.009259`, `-0.004630`이므로 모든 class에서 우월한 방법은 아닙니다.
- 실제 OK와 독립 실물 validation이 없어 threshold, temperature, `HOLD` coverage는 계산하지 않았고 `NOT VERIFIED`로 고정했습니다.

### D0 — component localization baseline

- full scene에서 부품 위치를 찾는 class-agnostic detector를 pretrained backbone으로 시작합니다.
- v4 component annotation의 status class를 하나의 `component` target으로 합쳐 localization부터 검증합니다.
- 출력은 scene별 component bbox/mask, confidence와 검출 개수입니다.
- 현재 v4 전체는 train-only이므로 synthetic detector validation/test 점수를 만들지 않습니다.
- 현재 실행 config는 v4+v5 combined만 허용하므로 엄밀한 D0 v4-only control은 아직 구현되지 않았습니다. v5 효과를 주장하려면 별도 v4-only config를 같은 update budget으로 실행해야 합니다.

### D1 — multi-light robustness training

- D0에 v5의 multi-light rig, P0–P5 proxy와 shadow regime을 보조 train data로 추가합니다.
- 먼저 `composition_family_id`를 추출하고, 그 family의 v4 neutral 또는 v5 variant 하나를 선택합니다.
- v4 한 장과 v5 두 장을 단순 합쳐 같은 geometry에 세 배 가중치를 주지 않습니다.
- P0–P5는 synthetic proxy입니다. 실제 50–1600 lux label, regression target 또는 실측 성능 구간으로 사용하지 않습니다.
- 현재 sampler는 384 family마다 v4/v5 variant 중 정확히 하나를 epoch별로 선택하고 3-epoch cycle을 검증합니다. 전체 source-parent graph가 한 connected component이므로 synthetic validation/test와 early stopping은 만들지 않습니다.
- `FasterRCNN-MobileNetV3-Large-FPN` COCO pretrained weight를 local cache에서 hash 검증 후 읽는 combined pipeline을 구현했습니다. bbox-aware 약한 증강, AMP, finite-gradient/state gate와 fixed epoch/no-validation 규칙을 적용합니다.
- CUDA smoke는 실제 finite optimizer update 8회를 완료했지만 이는 수렴·정확도 검증이 아니라 실행 가능성 점검입니다. Smoke checkpoint는 배포 또는 성능 주장에 사용하지 않습니다.

### D2 — direct detector와 cascade ablation

- A안: full scene에서 `normal_proxy`와 7개 defect status를 직접 예측하는 8-class detector
- B안: D0으로 component를 찾고 crop classifier 또는 defect segmenter가 결함 class를 판정하는 cascade
- 작은 scratch/crack의 recall, false positives/image, component 누락과 latency를 함께 비교합니다.
- defect localization이 부족하면 semantic mask를 사용한 CE/Focal + Dice segmentation을 후속 실험으로 추가합니다.
- 현재 synthetic 결과만으로 A/B의 최종 승자를 정하지 않고 실물 validation 전까지 후보로 유지합니다.

## Split 및 누수 규칙

v4의 source parent들은 여러 scene에 교차 배치됩니다. source parent를 node, 한 scene의 동시 배치를 edge로 보면 현재 parent graph는 단일 connected component입니다. 따라서 다음 분할은 모두 금지합니다.

- v4 scene을 무작위 train/validation/test로 분리
- 같은 `composition_family_id`의 v4와 v5 variant를 서로 다른 split에 배치
- instance crop, mask 또는 YOLO/COCO export를 새 표본처럼 재분할
- composition만 분리하고 공유 `source_parent_id`를 다른 split에 남김

엄격한 source-parent group split을 적용하면 현재 v4/v5 전체가 하나의 group이므로 detector용 synthetic validation/test를 만들 수 없습니다. v4/v5는 전부 `TRAIN_ONLY / evaluation_eligible=NO`를 유지합니다.

향후 synthetic development split이 필요하면 source parent를 class×severity별로 먼저 나눈 뒤, 각 pool 내부 parent만 사용해 scene을 다시 생성해야 합니다. 이 경우에도 동일 synthetic base와 `SPEC-A` 계보이므로 독립 실물 test를 대체하지 않습니다.

실물 자료는 최소한 다음 식별자의 연결 성분 전체를 같은 split에 둡니다.

- physical `specimen_id`
- capture session과 연속 frame/scene group
- 같은 원본의 crop·resize·augmentation·재인코딩본
- exact SHA-256 및 near-duplicate 검사에서 연결된 image

Locked real test는 architecture, augmentation, threshold, checkpoint 또는 `HOLD` 기준 선택에 사용하지 않습니다.

## Family-balanced sampler 계약

1. family를 균등하게 선택합니다.
2. 선택된 family 안에서 condition variant를 선택합니다.
3. class와 condition별 실제 draw 수를 epoch마다 저장합니다.
4. 파생 variant 수가 많은 family가 자동으로 높은 가중치를 갖지 않게 합니다.
5. 비교 실험은 동일 optimizer update budget을 사용합니다.
6. augmentation과 모든 annotation은 원본 family·split을 상속합니다.

실제 생산 class prior를 알기 전에는 balanced synthetic sampling과 production threshold를 분리합니다. 합성 class 균형을 실제 불량률로 간주하지 않습니다.

## 증강 정책

허용 후보:

- 작은 회전·이동·scale과 물리적으로 허용되는 flip
- exposure, gamma, contrast의 제한된 변화
- warm/cool channel gain과 component-only gradient/hotspot
- 약한 blur, sensor noise와 JPEG degradation
- mask와 함께 변환되는 contact/directional shadow
- bbox·mask를 동일 좌표 변환하고 defect visibility QC를 다시 통과한 결과

초기 실험에서 금지하거나 보류:

- defect를 지울 수 있는 `RandomErasing`
- 작은 scratch/crack을 잘라내는 강한 random crop
- label을 유지한 채 defect 가시성을 없애는 강한 blur·노출·압축
- mask를 갱신하지 않는 기하 변환
- 작은 국소 결함의 의미가 깨질 수 있는 강한 `CutMix`·`MixUp`
- 현재 multi-instance 합성과 중복되는 무제한 Mosaic
- 실제 측정 근거 없이 synthetic proxy를 lux ground truth로 변경

모든 augmentation은 label preservation을 우선하며, 새 효과를 추가할 때 기존 v3/v5 visibility gate 또는 그보다 엄격한 검사를 적용합니다.

## 평가 지표

Classifier:

- macro-F1, balanced accuracy
- class별 recall과 worst-class recall
- mild/moderate/severe recall
- confusion matrix
- ECE/Brier score
- `HOLD` coverage 대비 selective error

Detector/segmenter:

- mAP50-95, AP50과 class별 recall/AP
- false positives/image와 component miss rate
- scene별 component count error
- defect 크기·severity별 recall
- proxy, rig, shadow regime별 worst-slice 결과
- 같은 composition의 paired prediction consistency
- batch-1 latency, 처리량과 peak VRAM
- segmentation 사용 시 mask AP/mIoU와 scratch/crack boundary metric

48개 세부 condition cell의 class별 표본은 작으므로 개별 cell 점수로 모델을 선택하지 않습니다. proxy, rig, shadow의 상위 slice와 worst-slice를 함께 봅니다. Detector의 현재 synthetic paired 결과는 condition stress diagnostic이며 validation/test 성능이 아닙니다.

## Acceptance gates

### G0 — data integrity

- dataset validator, image/label/mask hash와 annotation namespace 검사 PASS
- split 간 exact/near-duplicate와 family overlap 없음
- invalid 또는 빈 bbox/mask 없음

### G1 — reproducibility

- code, config, dataset release와 pretrained weight SHA 기록
- seed, library/device version과 optimizer update 수 기록
- 동일 baseline 재실행 가능, NaN/Inf 없음

### G2 — method comparison

- 세 seed의 평균과 sample standard deviation 보고
- 주 metric 개선과 함께 worst-class/mild recall의 퇴보 여부 확인
- family 단위 paired comparison 사용
- classifier synthetic 지표는 sanity/ablation 용도로만 해석

### G3 — condition robustness

- neutral 조건을 희생하지 않으면서 worst proxy/rig/shadow slice와 paired consistency가 개선되는지 확인
- 합성 P0–P5 결과를 실제 lux별 검출률로 보고하지 않음

### G4 — real validation

- 독립 specimen의 real validation에서 threshold와 `HOLD` 정책 선택
- 실제 class별 recall, false accept/reject와 false positives/image 확인
- confidence interval은 image가 아니라 specimen/session group 단위로 계산
- 운영 허용치와 target hardware latency 요구가 정해진 뒤 PASS 기준 확정

### G5 — deployment 및 연구 주장

현재는 독립 real `OK`/NG locked test가 없으므로 `NOT VERIFIED`입니다. 합성 점수가 높아도 실제 생산 검출률, 출하 합격 판정, lux별 성능 또는 연구 novelty의 근거로 사용하지 않습니다. 합성 pretraining 뒤 독립 실물 train/validation/test를 확보하고, specimen-level 성능과 실패 사례를 검증해야 이 gate를 닫을 수 있습니다.
