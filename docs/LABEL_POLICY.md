# Label policy R5

## 판정 축

사진에는 서로 다른 두 판정 축을 사용합니다.

1. `visible_*`: 해당 사진에 실제로 보이는 결함
2. `specimen_status`: 동일 physical specimen의 모든 view를 합친 출하 gate

한 view에서 결함이 보이지 않아도 반대면이 미관찰이면 `OK`가 아닙니다.

## 결함 상태

- `positive`: 조명 또는 view 변화에도 같은 부품 위치에서 반복되는 결함
- `review`: 후보는 있으나 단독 확정이 어려움
- `ambiguous`: 현상은 보이지만 class subtype을 구분할 수 없음
- `not_observed`: 해당 관찰면에서 보이지 않음
- `not_visible`: 해당 면이 사진에 없음
- `unobservable`: 초점, 노출, 가림, 해상도 때문에 판독 불가

## Gate

- `OK`: 필요한 면이 모두 판독 가능하고 모든 필수 결함 class가 negative
- `NG`: 하나 이상의 결함 positive 또는 동일 specimen의 다른 view에서 NG 확인
- `HOLD`: OK나 NG로 확정할 근거가 부족함

`HOLD`와 `no_visible_defect_on_view`는 normal negative로 사용하지 않습니다.

## Split leakage

같은 specimen의 다각도 사진, 연속 frame, crop, augmentation은 반드시 같은 train/val/test group에 둡니다. 컨베이어 171640/171641/171652도 하나의 scene group입니다.

## Synthetic multi-instance label

`synthetic-v4-conveyor`와 이를 조명 조건별로 replay한 `synthetic-v5-illumination`은 실제 사진 label과 분리된 두 annotation namespace를 사용합니다.

1. Component status: `normal_proxy`와 7개 결함 status, 총 8개 class
2. Defect localization: scratch, surface spot, discoloration, contamination, lead breakage, body chip, body crack, 총 7개 class

`normal_proxy`는 paired-clean synthetic instance이며 `OK`, `OK_confirmed`, 실제 normal 또는 전기적 정상으로 해석하지 않습니다. Component status target에는 포함되지만 defect localization target은 비어 있습니다. `classification_eligible=NO`이므로 기존 단일-image classifier manifest에 합치지 않습니다.

Component status와 defect localization의 YOLO/COCO class ID는 서로 다른 namespace입니다. Loader는 task별 class map을 따로 사용해야 하며 ID 숫자만 보고 두 annotation을 병합하면 안 됩니다.

### Multi-instance split leakage

v4의 384개 장면과 1,920개 instance는 모두 `TRAIN_ONLY / evaluation_eligible=NO`입니다. Source는 v2의 `gradient_train` parent만 사용하며 validation/test parent를 파생하지 않습니다.

- 동일 source parent와 paired-clean은 같은 `family_split_id`로 묶습니다.
- 같은 scene의 5개 instance와 모든 label/mask/crop은 `composition_family_id` 단위로 묶습니다.
- scene 또는 instance crop을 다시 나누어 validation/test로 이동하지 않습니다.
- 모든 source가 하나의 synthetic-restored physical base family에서 왔으므로 v4 내부 random split은 independent-specimen 평가가 아닙니다.

v5는 v4 384개 composition마다 조명 조건 2개를 만들어 768개 장면을 제공합니다. 두 variant는 rig, synthetic illuminance proxy와 shadow regime이 서로 다르지만 동일한 source `composition_family_id`를 공유합니다.

- source v4 장면과 v5 두 variant를 모두 같은 train family로 유지합니다.
- 동일 composition의 variant를 train/validation/test로 나누지 않습니다.
- v5 image, instance crop, COCO/YOLO export와 mask를 독립 sample로 재분할하지 않습니다.
- v5도 `TRAIN_ONLY / evaluation_eligible=NO / classification_eligible=NO`이며 내부 random split metric을 독립 실물 성능으로 보고하지 않습니다.

### Synthetic illuminance proxy

v5의 `P0`–`P5`는 조명 조건을 균형화하기 위한 synthetic proxy ID입니다. `capture_plan_target_lux` 50/100/200/400/800/1600은 향후 실물 촬영 계획의 목표값이며, 합성 image의 측정 조도 label이 아닙니다.

- Synthetic row: `photometry_domain=SYNTHETIC_PROXY`, `absolute_lux_eligible=NO`, `measured_illuminance_lux=null`
- Real row: calibrated meter와 측정 plane을 기록한 실제 관측값만 measured lux로 사용
- `image_plane_azimuth_deg`와 `elevation_proxy_deg`는 2D rendering parameter이며 실제 fixture의 3D 각도 측정값이 아님
- 합성 proxy별 성능을 실제 lux 구간별 성능으로 이름 바꾸거나 보고하지 않음

실물 데이터는 [실물 조명 촬영 프로토콜](REAL_LIGHTING_CAPTURE_PROTOCOL.md)과 [`real_lighting_capture_template.csv`](../annotations/real_lighting_capture_template.csv)의 별도 schema를 사용합니다. 합성 row와 실물 row를 하나의 photometry field로 섞지 않습니다.

### Paired-clean label의 의미

v5는 동일 composition·lighting·shadow·sensor의 paired-clean reference와 defect image를 비교해 결함 semantic 영역의 가시성을 자동 확인합니다. 이 reference와 `normal_proxy`는 합성 QA용 counterfactual이며 실제 `OK_confirmed` label이 아닙니다. 자동 visibility gate 통과도 사람의 결함 판독 또는 실제 제품 합격 판정으로 기록하지 않습니다.

### Batch validation의 의미

`validate_synthetic_v4_conveyor.py`와 `validate_synthetic_v5_illumination.py`는 각 release 전체의 수량, label, mask, hash, 계보, 배치, 조명·shadow spill 및 paired-clean QC를 batch replay합니다. 이는 annotation/data integrity 검증이며 새 현장 사진을 분류하거나 결함을 검출하는 model inference가 아닙니다.
