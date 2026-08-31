# Label policy R4

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

`synthetic-v4-conveyor`는 실제 사진 label과 분리된 두 annotation namespace를 사용합니다.

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

### Batch validation의 의미

`validate_synthetic_v4_conveyor.py`는 release 전체의 수량, label, mask, hash, 계보, 배치, 조명 spill 및 paired-clean QC를 batch replay합니다. 이는 annotation/data integrity 검증이며 새 현장 사진을 분류하거나 결함을 검출하는 model inference가 아닙니다.
