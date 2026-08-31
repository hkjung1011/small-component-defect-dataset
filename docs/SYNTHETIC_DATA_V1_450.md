# Synthetic defect dataset v1-450

## 선정 구성

기존 `synthetic-v1` 900장을 복사하거나 잘라낸 것이 아니라, 동일한 `synthetic_restored` clean base와 defect recipe에 새 seed를 적용해 클래스별 50장을 새로 생성했습니다. 기존 release의 image, mask, manifest, generator는 변경하지 않았습니다.

| 항목 | 값 | 상태 |
|---|---|---|
| Release | `synthetic-v1-450` | CONFIRMED |
| Generator | `scripts/generate_synthetic_v1_450.py` v1.1.1 | CONFIRMED |
| Config | `configs/synthetic_v1_450.json` | CONFIRMED |
| Global seed | `7809250` | CONFIRMED |
| ID prefix | `syn-v1-450-*` | CONFIRMED |
| Resolution | 512×512 | CONFIRMED |
| Primary classes | 9개, 각 50장 | CONFIRMED |
| Images / semantic masks | 450 / 450 | CONFIRMED |
| Split | train only | CONFIRMED |
| Real validation/test | 없음 | NOT VERIFIED |

## 클래스와 라벨

| Primary class | 수량 | Semantic mask ID |
|---|---:|---:|
| `normal_proxy` | 50 | 0 |
| `scratch` | 50 | 1 |
| `surface_spot` | 50 | 2 |
| `discoloration` | 50 | 3 |
| `contamination` | 50 | 4 |
| `lead_breakage` | 50 | 5 |
| `body_chip` | 50 | 6 |
| `body_crack` | 50 | 7 |
| `multi_defect` | 50 | 1–7 중 복수 |

`synthetic/v1_450/annotations/manifest.csv`에는 sample별 image/mask 경로, primary class, visible multi-label, severity, seed, generator/config/base/image/mask SHA-256, defect pixel 수, bbox, ROI containment와 생성 파라미터가 있습니다. `instances.jsonl`은 mask에서 계산한 class별 area와 bbox를 제공합니다.

## 생성과 검증

```powershell
python -m pip install -r requirements-synthetic.txt
python scripts/generate_synthetic_v1_450.py --force
python scripts/validate_synthetic.py --config configs/synthetic_v1_450.json --release synthetic/v1_450
```

검증 결과:

- 9개 class × 50장, image 450개와 mask 450개: PASS
- image/mask decode, 크기, SHA-256, semantic class, multi-label, instance area/bbox: PASS
- 최종 geometry를 재적용한 mask ROI containment 99.9% 이상: PASS
- `synthetic-v1`과 sample ID, sample seed, image SHA-256 중복 0: PASS
- 동일 seed 9-class 최소 세트 2회 생성 tree SHA-256 `9685C75445F5C08A045EF7DBE64210494E387182743C557E5AEBF619405101CC`: PASS
- contact sheet 27장과 severe 대표 8장 수동 QA: `PASS_POC_VISUAL`

수동 QA 결과는 `annotations/synthetic_v1_450_human_qa.csv`에 기록했습니다.

## 잔여 위험

- 두 release가 같은 restored base group에서 파생되므로 1,350장이 독립 실물 1,350개를 의미하지 않습니다.
- `normal_proxy`는 생성형 복원본이며 실제 정상품이 아닙니다.
- surface spot, discoloration, contamination은 합성 recipe상 분리했지만 실제 사진에서 원인별 구분 성능은 검증되지 않았습니다.
- lead breakage, body chip, body crack은 2D proxy이고 실제 파단면, 그림자, coplanarity 재현은 `NOT VERIFIED`입니다.
- 모든 synthetic 자료는 train에만 사용하고 threshold와 최종 성능은 신규 실제 specimen으로 검증해야 합니다.
