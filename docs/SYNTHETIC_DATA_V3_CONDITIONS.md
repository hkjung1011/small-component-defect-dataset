# Synthetic v3 condition augmentation

`synthetic-v3-conditions`는 `synthetic-v2-700`의 결함 형상과 semantic mask를 그대로 유지하고 조명·노출·색온도·그림자·반사·초점·센서 노이즈·JPEG 조건만 바꾼 auxiliary train release입니다.

## 구성

| 항목 | 값 | 상태 |
|---|---:|---|
| Parent | v2 고정 split의 `gradient_train` 168장 | CONFIRMED |
| Parent/class | 24장 | CONFIRMED |
| Variant/parent | 6장 | CONFIRMED |
| 신규 image/mask | 1,008 / 1,008 | CONFIRMED |
| 신규 image/class | 144장 | CONFIRMED |
| Split | `gradient_train_auxiliary` | TRAIN_ONLY |
| Evaluation eligible | `NO` | CONFIRMED |
| 신규 실물 specimen | 0 | CONFIRMED |

적용 profile은 다음 6종입니다.

- `underexposure`
- `overexposure`
- `warm_directional`
- `cool_directional`
- `soft_shadow_vignette`
- `specular_sensor`

변환 범위는 [config](../configs/synthetic_v3_conditions.json)에 기록했습니다. `discoloration`에는 강한 white-balance와 hotspot을 제한하고, `lead_breakage`에는 512/224 가시성 gate를 유지하기 위한 보수적 노출·blur 제한을 별도로 적용합니다.

## 계보와 split 누수 방지

각 manifest 행에는 `parent_sample_id`, parent image/mask SHA-256, `lineage_group_id`, `family_split_id`, `augmentation_family_id`, condition seed·profile·parameter가 포함됩니다.

새 release는 기존 split에서 이미 `gradient_train`으로 확정된 parent만 사용합니다. validation parent 28장과 test parent 504장은 파생 source로 사용하지 않습니다. 따라서 v3 variant는 해당 parent와 함께 train에서만 사용하고, validation/test 또는 최종 성능 계산에 넣으면 안 됩니다.

## QC

- v2 parent image와 mask를 seed·recipe에서 독립 replay하고 published SHA-256과 일치 확인
- defect와 paired-clean에 동일 condition transform 적용
- semantic mask는 geometry 변경 없이 픽셀 단위로 보존
- post-JPEG 512 px 및 224 px에서 v2 class/severity visibility gate 재적용
- 평균 휘도, 휘도 표준편차, black/white clipping gate 적용
- image/mask/config/generator/manifest SHA-256과 sample·seed·path inventory 검증
- class×profile 균형 및 parent당 6개 계보 검증
- raw/overlay contact sheet 수동 시각 감사

Overview 1장과 class별 raw/overlay 14장, 총 15개 sheet의 전수 시각 감사 결과는 [`annotations/synthetic_v3_conditions_human_qa.csv`](../annotations/synthetic_v3_conditions_human_qa.csv)에 기록했습니다.

최종 1,008장 중 1,006장은 profile full strength에서 통과했습니다. 낮은 색차 margin을 가진 `body_chip/underexposure` 2장만 결함 가시성 gate를 만족할 때까지 strength를 감쇠했으며 최저 적용 scale은 `0.418947`입니다. 모든 값은 sample별 `parameters_json`에 기록됩니다. 신규 image SHA-256은 1,008개 모두 고유하고 v2 image와 exact hash 중복은 0입니다.

독립 validator 최종 결과:

```text
PASS: synthetic=1008, parents=168, profiles=6, classes=7,
gradient_train_only=YES, deterministic_replay=PASS,
post_jpeg_512_224=PASS, evaluation_eligible=NO
```

`ImageGen` 조명 편집도 사전 시험했지만 원본의 표면 무늬와 부품 형상이 변해 label invariant를 만족하지 못했습니다. 해당 시험 출력은 release에 포함하지 않았으며, 최종 자료는 deterministic camera/illumination transform만 사용합니다.

## 재현

```powershell
py -3.14 -B scripts\generate_synthetic_v3_conditions.py --force
py -3.14 -B scripts\validate_synthetic_v3_conditions.py
py -3.14 -B training\scripts\train_eval_classifier.py --check-only --auxiliary-condition-manifest synthetic\v3_conditions\annotations\manifest.csv
```

이 release는 촬영조건 robustness를 늘리지만 새로운 결함 morphology, 새 부품, 새 생산 lot 또는 실물 normal을 추가하지 않습니다. 실물 성능은 독립 specimen의 실제 normal/결함 validation·test로 별도 검증해야 합니다.

저장소에는 v2-only C0 checkpoint 3개와 v3 condition을 gradient-train에
append한 C2 checkpoint 3개가 있습니다. C2의 3-seed synthetic same-base
macro-F1은 `0.984178`, mild recall은 `0.962233`이지만 C0보다 optimizer
update가 많아 condition 자체의 순효과로 확정할 수 없습니다. Parent당
variant 하나만 선택하고 C0와 같은 180 updates를 사용한 C3 control은
macro-F1 `0.964601`, mild recall `0.919540`으로 현재 recipe를 기각했습니다.
두 결과 모두 base validation/test를 그대로 사용한 합성 sanity 결과이며
실물·독립 specimen 성능은 `NOT VERIFIED`입니다.
