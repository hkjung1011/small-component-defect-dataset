# Label policy R3

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
