# Synthetic defect dataset v1

## 선정 구성

`20260827_171603`의 후면 사진에서 scratch를 제거한 clean-back 이미지를 base로 사용하고, 2D procedural defect와 domain randomization을 적용했습니다. 원본 specimen이 NG였고 clean base도 생성형 복원본이므로 `real normal`이 아니라 `synthetic_restored`로 기록했습니다.

| 항목 | 값 | 상태 |
|---|---|---|
| Release | `synthetic-v1` | CONFIRMED |
| Generator | `scripts/generate_synthetic.py` v1.1.0 | CONFIRMED |
| Base | `synthetic/sources/clean_back_crop_v1.png` | CONFIRMED |
| Base domain | `synthetic_restored` | CONFIRMED |
| Resolution | 512×512 | CONFIRMED |
| Samples | 900장 | CONFIRMED |
| Primary classes | 9개, 각 100장 | CONFIRMED |
| Pixel masks | 900개 lossless PNG | CONFIRMED |
| Split | train only | CONFIRMED |
| Real validation/test | 없음 | NOT VERIFIED |

## 클래스 및 mask 값

| Primary class | Semantic mask ID | 의미 |
|---|---:|---|
| `normal_proxy` | 0 | 복원 clean base의 조건 변화. 실제 정상품이 아님 |
| `scratch` | 1 | metal tab 선형 scratch |
| `surface_spot` | 2 | 원인 미확정 점상 표면 이상 proxy |
| `discoloration` | 3 | 넓고 부드러운 색 변화 recipe |
| `contamination` | 4 | smear와 particle 조합 recipe |
| `lead_breakage` | 5 | outer lead 끝단 material loss proxy |
| `body_chip` | 6 | black body edge의 material loss proxy |
| `body_crack` | 7 | body edge에서 시작되는 crack proxy |
| `multi_defect` | 1–7 중 복수 | 충돌하지 않는 2종 결함 조합 |

실데이터의 `surface_spot_unknown`과 연결할 때는 synthetic `surface_spot`, `discoloration`, `contamination`을 원인 라벨이 아닌 관찰형상 proxy로 취급해야 합니다. 실제 데이터에서는 세 종류가 아직 구분 검증되지 않았습니다.

## 자동 라벨

`synthetic/v1/annotations/manifest.csv`에는 다음 항목을 포함합니다.

- image와 semantic mask 경로
- primary class와 visible multi-label
- severity, recipe, domain-randomization 파라미터
- global/sample seed와 generator/config version
- base/image/mask SHA-256
- mask pixel 수, bbox, ROI containment, 합성 전후 visibility delta
- `TRAIN_ONLY`와 `evaluation_eligible=NO`

`instances.jsonl`은 sample별 class ID, mask area, bbox를 제공합니다. Semantic PNG는 배경 0과 위 표의 class ID를 저장하며, 화면에서 그대로 열면 거의 검게 보이는 것이 정상입니다. 육안 확인은 `contact_sheet.jpg`의 color overlay를 사용합니다.

## 생성 및 검증

```powershell
python -m pip install -r requirements-synthetic.txt
python scripts/generate_synthetic.py --force
python scripts/validate_synthetic.py
```

기본 seed는 `7809208`이고 config와 generator script, manifest, instance label, summary의 SHA-256을 `release.json`에 기록합니다. 출력 폴더를 제거하는 `--force`는 `.synthetic_release_marker`가 있는 repository 내부 `synthetic/` 하위 경로에만 허용됩니다.

## QA 결과

- 이미지 900개와 mask 900개의 decode, 크기, 경로, SHA-256, class ID, bbox 일치: PASS
- semantic mask class·`visible_multilabel`·`instances.jsonl`의 class/area/bbox 직접 대조: PASS
- class balance: 9개 class 각각 100장: PASS
- synthetic split이 모두 train-only이며 평가 사용 불가: PASS
- validator가 각 샘플의 geometry와 config ROI를 재적용한 독립 결함 mask containment 99.9% 이상: PASS
- generator가 합성 직후 기록한 defect 영역 visibility delta gate: PASS
- contact sheet 27장과 severe 대표 8장 수동 image/mask 비교: `PASS_POC_VISUAL`
- 동일 seed로 9-class 최소 세트를 두 번 재생성한 전체 파일 tree SHA-256 `EE7DC3FFCE8014957E0CE5DD21335DDC48ADB15AEB3FAE34FE8AC61C2CAF1ABB` 일치: PASS

수동 검토 표본은 `annotations/synthetic_v1_human_qa.csv`에 기록했습니다. 이 파일은 release 재생성 시 삭제되지 않습니다. 이는 generator PoC의 시각적 타당성 확인이며 실제 현장과의 domain realism 또는 분류 성능 검증은 아닙니다.

## 잔여 위험

- base group이 하나뿐이므로 900장은 독립 실물 900개가 아닙니다.
- `normal_proxy`는 생성형 복원본이므로 실제 정상품 학습자료를 완전히 대체하지 않습니다.
- lead breakage와 body damage는 2D proxy이며 실제 fracture surface, shadow, coplanarity를 완전히 재현하지 않습니다.
- visibility delta는 generator 내부 합성 직후 값이며, 실제 촬영 domain에서의 지각 가능성과 동등하다고 검증된 값은 아닙니다.
- synthetic 자료는 train에만 사용하고, threshold·accuracy·false reject/miss 평가는 신규 실제 specimen으로 수행해야 합니다.
