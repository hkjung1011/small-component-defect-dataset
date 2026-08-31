# Small Component Defect Dataset

KEC `KIA7809AF`로 식별된 소형 전원 반도체 사진 17장을 직접 수동 검수해, specimen 상태와 사진에서 보이는 결함을 분리한 dataset입니다. 최신 라벨은 `annotations/image_labels_v4.csv`입니다.

실데이터와 별도로, clean-back 복원본에서 자동 생성한 train-only synthetic 이미지 2,050장과 pixel mask 2,050장을 포함합니다.

## 최종 판정

| 구분 | 사진 수 | 의미 |
|---|---:|---|
| `OK_confirmed` | 0 | 모든 필수 면에서 결함 없음이 확인된 정상품 |
| `NG_confirmed` | 13 | 사진 자체 결함 또는 같은 specimen의 반복 관찰로 NG 근거가 있는 사진 |
| `HOLD_unverified` | 4 | 화질 또는 다중 부품 장면 때문에 OK/NG 확정 불가 |

따라서 현재 17장 중 정상품 학습 negative로 사용할 수 있는 사진은 없습니다. `HOLD`도 정상품이 아니며, 재촬영 또는 instance-level 판정 전까지 학습에서 제외해야 합니다.

## Synthetic releases

| Release | 클래스별 수량 | 총 이미지 / mask | 조건 |
|---|---:|---:|---|
| `synthetic-v1` | 100장 | 900 / 900 | seed `7809208`, ID `syn-v1-*` |
| `synthetic-v1-450` | 50장 | 450 / 450 | 신규 random condition, seed `7809250`, ID `syn-v1-450-*` |
| `synthetic-v2-700` | 100장 | 700 / 700 | 7개 결함 class, post-JPEG 512/224 QC, ID `syn-v2-700-*` |

`synthetic-v1`과 `synthetic-v1-450`은 `normal_proxy`, scratch, surface spot, discoloration, contamination, lead breakage, body chip, body crack, multi-defect의 9개 primary class로 구성됩니다. `synthetic-v1-450`은 기존 900장의 subset이나 복사본이 아니라 다른 seed로 새로 생성한 450장입니다.

`synthetic-v2-700`은 scratch, surface spot, discoloration, contamination, lead breakage, body chip, body crack의 7개 결함 class만 포함합니다. 각 class는 `mild 40 / moderate 40 / severe 20`이고, paired-clean 비교로 JPEG 저장 후 512 px와 학습 입력 224 px에서 모두 가시성 gate를 통과했습니다. 정상 class는 포함하지 않으므로 이 release만 학습한 모델은 OK/NG 판정을 할 수 없습니다.

Synthetic 자료는 모두 `synthetic_restored / TRAIN_ONLY / evaluation_eligible=NO`입니다. 실제 정상품이나 독립 실물 specimen 수로 집계하면 안 됩니다. 자세한 생성 조건·라벨·QA는 [synthetic-v1 문서](docs/SYNTHETIC_DATA.md), [synthetic-v1-450 문서](docs/SYNTHETIC_DATA_V1_450.md), [synthetic-v2-700 실험 보고서](docs/SYNTHETIC_DATA_V2_700.md)에 있습니다.

## 재감사에서 바뀐 항목

- `20260827_171929`, `20260827_171931`: 두 frame의 동일 위치에서 곡선형 손상이 반복되어 `scratch confirmed`로 승격했습니다.
- `20260827_171611`, `20260827_171925`: 점상 표면 이상은 확정했지만 `discoloration`과 `contamination` 중 어느 것인지는 보류했습니다.
- `20260827_171925`: outer lead 변형은 검토 대상으로 유지하고 `breakage confirmed`로 확정하지 않았습니다.
- 전면이 깨끗한 4장은 `no_visible_defect_on_view`일 뿐, specimen 기준 정상품으로 사용하지 않습니다.

## 폴더

```text
annotations/                    사진별 라벨과 source hash
configs/synthetic_v1.json       합성 class, ROI, seed, domain-randomization 설정
configs/synthetic_v1_450.json   클래스당 50장 신규 release 설정
configs/synthetic_v2_700.json   7개 결함 class, severity, dual-resolution QC 설정
data/by_specimen_status/        OK / NG / HOLD 상호배타 분류
data/by_visible_class/          사진에서 보이는 결함의 multi-label export
data/crops_by_specimen_status/  단일 부품 crop 보조본
docs/                           재감사, 라벨 정책, 부품 식별, 공식 출처
scripts/validate_dataset.py     수량, 경로, SHA-256, 라벨 불변조건 검사
scripts/generate_synthetic.py   image, mask, manifest 동시 생성
scripts/generate_synthetic_v1_450.py  신규 450장 release 생성
scripts/validate_synthetic.py   synthetic image/mask/label/hash 검사
scripts/generate_synthetic_v2_700.py  700장 paired-clean post-JPEG QC 생성
scripts/validate_synthetic_v2_700.py  700장 512/224 QC 독립 replay 검증
synthetic/sources/              clean base와 생성 provenance
synthetic/v1/                   900 images, 900 masks, 자동 라벨과 QA
synthetic/v1_450/               450 images, 450 masks, 자동 라벨과 QA
synthetic/v2_700/               700 images, 700 masks, full contact sheets
training/                       고정 split, ResNet-18 학습·평가·3-seed 집계
```

## 부품 식별

사진 마킹은 회전 후 `KIA / 7809AF / 208`로 읽히며, KEC 공식 marking specification과 일치합니다. Base model은 `KIA7809AF`, 기능은 `9 V / 1 A three-terminal positive standard voltage regulator`, package family는 `DPAK / DPAK(3)`입니다. `208`은 모델 suffix가 아니라 `2022년 08주차` lot code입니다. 정확한 ordering suffix와 DPAK 세부 variant는 사진만으로 확정하지 않았습니다.

자세한 근거는 [docs/PART_IDENTIFICATION.md](docs/PART_IDENTIFICATION.md), 공식 문서는 [docs/SOURCES.md](docs/SOURCES.md), 라벨 한계는 [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)를 참고하십시오.

사진별 수동 판정과 학습 포함/제외 결정은 [docs/MANUAL_LABELING_REPORT.md](docs/MANUAL_LABELING_REPORT.md)에 기록했습니다.

## 검증

```powershell
python scripts/validate_dataset.py
python scripts/validate_synthetic.py
python scripts/validate_synthetic.py --config configs/synthetic_v1_450.json --release synthetic/v1_450
python scripts/validate_synthetic_v2_700.py
python training/scripts/train_eval_classifier.py --check-only
```

성공 시 real v4는 `OK=0, NG=13, HOLD=4`, synthetic release는 각각 `synthetic=900`, `synthetic=450`, `synthetic=700` PASS를 출력합니다.

## 사용 제한

이 저장소에는 공개 라이선스가 부여되지 않았습니다. 사진 기반 외관 판정은 전기적 정상성, 진품 여부, 안전성 또는 양산 합격을 보증하지 않습니다.
