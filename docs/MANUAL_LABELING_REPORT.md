# 수동 라벨링 결과 v4

검토일: 2026-08-31  
검토자: OpenAI Codex  
검토 방식: 원본 17장, 전체 contact sheet, 단일 부품 crop을 직접 시각 비교

## 판정 원칙

- 사진에서 보이는 결함과 실물 specimen 상태를 분리했습니다.
- 동일 위치의 흔적이 연속 frame에서 재현될 때만 scratch를 확정했습니다.
- 한 면에 결함이 보이지 않아도 동일 specimen의 다른 면에서 결함이 확인되면 정상품으로 사용하지 않았습니다.
- 원인이 구분되지 않는 점상 이상은 `surface_spot_unknown`으로 유지했습니다.
- 파단면이 보이지 않는 lead 이상은 `lead_deformation_review`로 두고 breakage로 확정하지 않았습니다.
- blur, 과노출, 원거리 다중부품 장면은 학습에서 제외했습니다.

## 사진별 최종 라벨

| 이미지 | 수동 image label | specimen | 학습 처리 |
|---|---|---|---|
| 171517 | HOLD_QUALITY | NG | 화질 제외 |
| 171519 | HOLD_QUALITY | NG | 화질 제외 |
| 171532 | HOLD_QUALITY | NG | 화질 제외 |
| 171544 | VIEW_OK_NOT_SPECIMEN_OK | NG | 정상 학습 제외 |
| 171549 | VIEW_OK_NOT_SPECIMEN_OK | NG | 정상 학습 제외 |
| 171555 | VIEW_OK_NOT_SPECIMEN_OK | NG | 정상 학습 제외 |
| 171556 | VIEW_OK_NOT_SPECIMEN_OK | NG | 정상 학습 제외 |
| 171602 | NG_SCRATCH | NG | specimen-group split 필수 |
| 171603 | NG_SCRATCH | NG | specimen-group split 필수 |
| 171611 | NG_SCRATCH_SPOT | NG | spot subtype 미확정 |
| 171640 | HOLD_MULTI_PART | HOLD | 개별 판독 불가 |
| 171641 | HOLD_MULTI_PART | HOLD | 개별 판독 불가 |
| 171652 | HOLD_MULTI_PART | HOLD | 개별 판독 불가 |
| 171925 | NG_SCRATCH_SPOT_LEAD_REVIEW | NG | lead는 review만 허용 |
| 171929 | NG_SCRATCH | NG | specimen-group split 필수 |
| 171931 | NG_SCRATCH | NG | specimen-group split 필수 |
| 171936 | HOLD_QUALITY | HOLD | 재촬영 필요 |

## 결론

- 확정 scratch: 6장
- 확정 surface spot: 2장, 단 변색/오염 subtype은 미확정
- lead deformation review: 1장
- breakage/discoloration/contamination confirmed: 각 0장
- 확정 정상품: 0장

현재 확정 결함 라벨은 annotation seed로 사용할 수 있지만, 독립 specimen 수가 2개 수준이므로 성능 평가나 train/test 분할 근거로는 부족합니다. 같은 specimen의 모든 frame은 반드시 같은 split에 둬야 합니다.
