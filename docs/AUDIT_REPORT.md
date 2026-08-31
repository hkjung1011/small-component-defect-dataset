# 결함 재감사 보고서 R3

## 결론

엄격한 specimen gate 기준으로 `OK 0 / NG 13 / HOLD 4`입니다. 모든 17장은 `normal_training_eligible=NO`입니다.

## 변경 사항

| 항목 | R2 | R3 | 근거 |
|---|---|---|---|
| 171929 | scratch review | scratch positive | 다른 frame에서도 부품상의 동일 좌표에 곡선형 흔적 유지 |
| 171931 | scratch review | scratch positive | 171929와 동일 위치와 형상 재현 |
| 171611 dark spot | 변색/오염 ambiguous | spot positive, subtype ambiguous | 촬영 조건이 달라도 국부 spot 자체는 유지 |
| 171925 dark spot | 변색/오염 ambiguous | spot positive, subtype ambiguous | 171611과 대응되는 국부 표면 이상 |
| 171925 lead | breakage 후보 | lead deformation review, breakage ambiguous | 끝단 이상은 보이나 원근과 mat 홈 영향 배제 불가 |

## 정상과 비정상 구분

- `OK_confirmed`: 필수 표면과 lead가 판독 가능하고 결함 없음이 확인된 별도 specimen만 허용합니다. 현재 0장입니다.
- `NG_confirmed`: visible defect가 확정됐거나 source metadata상 같은 specimen의 다른 view에서 결함이 확정된 사진입니다. 13장입니다.
- `HOLD_unverified`: 결함이 없다는 뜻이 아니라 판정 근거가 부족하다는 뜻입니다. 4장입니다.
- `no_visible_defect_on_view`: 촬영된 면만 깨끗한 상태입니다. specimen-level `OK`와 동일하지 않습니다.

## 결함별 최종 상태

| 결함 | 확정 사진 | 검토/미확정 |
|---|---|---|
| Scratch | 171602, 171603, 171611, 171925, 171929, 171931 | 171936 suspected but unreadable |
| Surface spot anomaly | 171611, 171925 | discoloration vs contamination subtype unresolved |
| Breakage | 없음 | 171925 outer lead deformation review |
| Discoloration | 없음 | 171611, 171925 ambiguous |
| Contamination | 없음 | 171611, 171925 ambiguous |

## 잔여 위험과 필요한 확인

- `171925` lead: 평면 datum 위 정면/측면 macro 촬영 후 양쪽 outer lead 길이, coplanarity, 끝단 fracture surface 비교가 필요합니다.
- Spot subtype: 균일한 diffuse lighting, white balance 고정, 세척 전후 macro 또는 현미경 비교가 필요합니다.
- 컨베이어 3장: 동일 instance ID를 유지한 crop과 더 높은 표면 해상도의 재촬영이 필요합니다.
- 정상 class: 별도 기준 정상품을 전면, 후면, 양 측면, lead 끝단까지 촬영해야 합니다.
- 사진 판정은 전기적 기능과 진품 여부를 검증하지 않습니다.
