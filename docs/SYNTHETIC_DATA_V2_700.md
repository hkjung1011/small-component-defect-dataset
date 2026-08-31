# Synthetic v2 700 생성·학습 보고서

## 결론

`synthetic-v2-700`은 7개 결함 class별 100장, 총 700장의 image-level synthetic classification release입니다. 모든 image/mask는 자동 무결성 검사와 224 px 전체 contact-sheet 수동 감사를 통과했습니다.

ResNet-18을 196장 개발 pool로 구성해 3개 training seed로 반복한 결과, 동일한 504장 synthetic holdout에서 평균 Accuracy는 `97.09 ± 0.41%p`, Macro-F1은 `97.11 ± 0.39%p`였습니다. 이 수치는 단일 restored base와 동일 generator를 공유한 **synthetic same-base sanity 결과**이며, 실물 또는 신규 specimen 검출 성능이 아닙니다.

## Release 구성

| Class | 이미지 | Mild | Moderate | Severe | Mask ID |
|---|---:|---:|---:|---:|---:|
| `scratch` | 100 | 40 | 40 | 20 | 1 |
| `surface_spot` | 100 | 40 | 40 | 20 | 2 |
| `discoloration` | 100 | 40 | 40 | 20 | 3 |
| `contamination` | 100 | 40 | 40 | 20 | 4 |
| `lead_breakage` | 100 | 40 | 40 | 20 | 5 |
| `body_chip` | 100 | 40 | 40 | 20 | 6 |
| `body_crack` | 100 | 40 | 40 | 20 | 7 |

이 release에는 `normal/OK`, `normal_proxy`, `multi_defect`가 없습니다. 따라서 이 데이터만 학습한 7-way classifier는 정상 이미지를 입력해도 결함 class 중 하나를 강제로 출력합니다.

주요 파일:

- 설정: `configs/synthetic_v2_700.json`
- 생성기: `scripts/generate_synthetic_v2_700.py`
- 독립 검증기: `scripts/validate_synthetic_v2_700.py`
- manifest: `synthetic/v2_700/annotations/manifest.csv`
- instance label: `synthetic/v2_700/annotations/instances.jsonl`
- 전체 raw/overlay contact sheet: `synthetic/v2_700/annotations/contact_sheets/`

## 생성 및 QC

각 sample은 결함 이미지와 동일 geometry, brightness, contrast, saturation, channel gain, blur, noise seed를 적용한 paired clean control을 만듭니다. 두 이미지를 같은 JPEG 조건으로 encode/decode한 뒤, semantic mask 내부 차이를 512 px와 model input 224 px에서 각각 계산합니다.

검사 항목은 class/severity별 mask area·shape, mean absolute RGB delta, masked CIE76 median(`delta_e76_p50`), changed-pixel fraction, ROI 포함률입니다. `delta_e76_p50`은 CIEDE2000이 아닙니다.

최종 전수 검사 결과:

- image 700 / mask 700 / class별 100장 / severity quota: PASS
- JPEG format, 512×512, path, SHA-256, mask class, bbox/area, instance record: PASS
- paired-clean post-JPEG 512/224 visibility replay: PASS 700 / FAIL 0
- 100장/class raw·overlay 224 px 수동 감사: PASS 700 / REVIEW 0 / FAIL 0
- 기존 `synthetic/v1`, `synthetic/v1_450` 회귀 검증: PASS
- Seed `2700701` checkpoint 독립 재평가: prediction, confusion matrix, class metrics exact match

| Class | 최소 mask area 224 px | 최소 MAD 224 | 최소 dE76 p50 224 |
|---|---:|---:|---:|
| `scratch` | 19 | 52.936 | 19.820 |
| `surface_spot` | 63 | 39.596 | 15.272 |
| `discoloration` | 465 | 18.871 | 10.036 |
| `contamination` | 127 | 31.466 | 14.979 |
| `lead_breakage` | 77 | 34.909 | 14.136 |
| `body_chip` | 25 | 20.627 | 8.020 |
| `body_crack` | 25 | 11.212 | 8.705 |

최초 생성본의 `scratch-0056`은 자동 gate를 통과했지만 224 px 수동 감사에서 금속 texture와 구분하기 어려워 FAIL 처리했습니다. 스크래치 최소 폭·길이·대비와 gate를 강화하고 release 전체를 다시 생성한 뒤 최종 700장을 재검증했습니다.

## 학습·평가 설계

| 항목 | 값 |
|---|---|
| Model | ImageNet pretrained `ResNet-18`, local cached weights only |
| Input | 전체 sample 공통 fixed ROI `[96, 64, 384, 416]` → 224×224 |
| Outer split | train-development 196 / locked test 504 |
| Model split | gradient train 168 / validation 28 / test 504 |
| Class별 outer split | train 28 / test 72 |
| Class별 test severity | mild 29 / moderate 29 / severe 14 |
| Training | 최대 30 epochs, backbone freeze 4 epochs, AdamW |
| Selection | validation loss; test는 checkpoint 선택 완료 후 1회 평가 |
| Repeats | training seed `2700701`, `2700711`, `2700721` |
| Split fingerprint | `38edccbcdd706764a63e7d48a90a657f512712d5b28faeef788d5fd5009479c8` |

Split은 class×severity 내부 SHA-256 rank로 고정했습니다. Train/validation/test 사이 sample ID와 exact image SHA-256 중복은 0입니다. 다만 모든 partition이 같은 `base_group_id`와 `source_specimen_group`을 공유하므로 specimen leakage는 구조적으로 존재합니다.

## 3-seed 클래스별 결과

`Recall`을 요청한 클래스별 검출률로 해석했습니다. 각 반복의 class별 test support는 72장입니다. `±`는 3개 training seed의 sample standard deviation입니다.

| Class | Precision | Recall(검출률) | F1 |
|---|---:|---:|---:|
| `scratch` | 100.00 ± 0.00%p | **94.44 ± 2.41%p** | 97.13 ± 1.28%p |
| `surface_spot` | 98.11 ± 0.80%p | **95.83 ± 0.00%p** | 96.96 ± 0.39%p |
| `discoloration` | 99.54 ± 0.79%p | **99.54 ± 0.80%p** | 99.54 ± 0.40%p |
| `contamination` | 94.54 ± 1.22%p | **95.83 ± 1.39%p** | 95.17 ± 0.07%p |
| `lead_breakage` | 97.31 ± 1.32%p | **100.00 ± 0.00%p** | 98.63 ± 0.68%p |
| `body_chip` | 91.11 ± 3.92%p | **98.15 ± 0.80%p** | 94.46 ± 1.99%p |
| `body_crack` | 100.00 ± 0.00%p | **95.83 ± 1.39%p** | 97.87 ± 0.72%p |

주요 오분류 방향은 `scratch → body_chip`, `surface_spot → contamination`, `contamination → surface_spot/body_chip`, `body_crack → body_chip`입니다. `body_chip`은 Recall은 높지만 다른 class를 chip으로 잘못 받는 경우가 있어 Precision이 가장 낮습니다.

Severity 전체 평균 Recall은 `mild 93.76 ± 0.28%p`, `moderate 99.01 ± 0.85%p`, `severe 100.00 ± 0.00%p`였습니다. Mild class 중 `scratch 89.66%`, `surface_spot 89.66%`, `body_crack 90.80%`, `contamination 91.95%`가 특히 낮았습니다. Moderate/severe synthetic 결함만으로 전체 성능을 판단하면 안 됩니다.

결과 파일:

- 3-seed aggregate: `training/results/final-stratified-aggregate-3seeds/`
- seed별 confusion matrix, prediction, metrics: `training/results/final-stratified-seed-*/`
- checkpoint: 각 seed directory의 `model_final.pt` (최종 3개만 저장소에 명시적으로 포함)

## 재현 명령

```powershell
$py = 'python'

& $py -B scripts\validate_synthetic_v2_700.py
& $py -B training\scripts\train_eval_classifier.py --check-only

& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700701 --output training\results\final-stratified-seed-2700701
& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700711 --output training\results\final-stratified-seed-2700711
& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700721 --output training\results\final-stratified-seed-2700721

& $py -B training\scripts\aggregate_seed_results.py --runs `
  training\results\final-stratified-seed-2700701 `
  training\results\final-stratified-seed-2700711 `
  training\results\final-stratified-seed-2700721 `
  --output training\results\final-stratified-aggregate-3seeds
```

## 사용 제한과 다음 검증

현재 결과로 확인된 것은 generator가 만든 7개 결함 형태를 같은 generator domain에서 구분할 수 있다는 점입니다. 다음 항목은 검증되지 않았습니다.

- 실제 정상품과 결함품의 OK/NG 판정
- 새로운 실물 specimen, 촬영기기, 조명, 배경에 대한 일반화
- 한 이미지의 여러 부품 또는 여러 결함 검출
- 결함 위치를 출력하는 object detection/segmentation 성능
- 실제 생산 불량률 조건에서의 false-positive rate, calibration, latency

실사용 평가는 최소한 독립 실물 specimen 단위로 split한 normal/7-class test set이 필요합니다. 같은 부품을 여러 장 촬영했다면 모든 frame을 같은 split에 묶어야 합니다. 정상 class를 추가하기 전에는 현재 모델을 양산 OK/NG 판정에 사용하면 안 됩니다.
