# Synthetic v5 paired multi-light illumination release

`synthetic-v5-illumination`은 검정 컨베이어 다중 부품 장면에 여러 방향의 광원과 그림자 조건을 추가한 detection/segmentation용 보조 학습 release입니다. `synthetic-v4-conveyor`의 384개 composition을 중립 상태로 deterministic replay하고, composition마다 서로 다른 조명 조건 2개를 적용해 768개 장면을 만듭니다.

이 release는 새 실물 사진이나 새 specimen을 추가하지 않습니다. 모든 장면은 `TRAIN_ONLY / evaluation_eligible=NO / classification_eligible=NO`이며 실제 성능 평가에 사용할 수 없습니다.

## 구성

| 항목 | 값 | 해석 |
|---|---:|---|
| Source composition | 384 | v4 장면의 geometry·placement·label 재사용 |
| Variant/source | 2 | 서로 다른 rig·proxy bin·shadow regime |
| Scene | 768, 1280×720 | train-only auxiliary |
| Component/scene | 5 | v4와 동일 |
| Component instance | 3,840 | 8 status, class당 480 |
| Multi-light rig | 4 | 한 장면에 2–3개 광원 동시 적용 |
| Synthetic illuminance proxy | 6 | `P0`–`P5`, 실측 lux 아님 |
| Shadow regime | 2 | contact + directional shadow |
| Full condition cell | 48 | 4 rigs × 6 proxies × 2 shadows |
| Scene/cell | 16 | 균형 고정 |
| Class/cell | 9–11 instances | 허용 균형 범위 |
| 신규 실물 image/specimen | 0 | 실제 domain 다양성 증가 없음 |

Component status는 다음 8개입니다.

- `normal_proxy`
- `scratch`
- `surface_spot`
- `discoloration`
- `contamination`
- `lead_breakage`
- `body_chip`
- `body_crack`

각 status는 480개입니다. `normal_proxy`는 합성 paired-clean 상태일 뿐 실제 `OK_confirmed`, 전기적 정상 또는 출하 합격을 뜻하지 않습니다.

## 조도 proxy와 실측 lux의 구분

이 release의 photometry domain은 `SYNTHETIC_PROXY`입니다. `relative_light_power`와 노출·noise·blur·JPEG parameter를 함께 바꿔 밝기 조건을 상대적으로 구성하며, 광학계나 lux meter로 radiometric calibration한 값이 아닙니다.

| Proxy | `capture_plan_target_lux` | 의미 |
|---|---:|---|
| `P0` | 50 | 향후 실물 촬영 계획의 목표점 |
| `P1` | 100 | 향후 실물 촬영 계획의 목표점 |
| `P2` | 200 | 향후 실물 촬영 계획의 목표점 |
| `P3` | 400 | 향후 실물 촬영 계획의 목표점 |
| `P4` | 800 | 향후 실물 촬영 계획의 목표점 |
| `P5` | 1600 | 향후 실물 촬영 계획의 목표점 |

위 숫자는 synthetic image에서 측정된 lux가 아닙니다. 모든 합성 row는 다음 상태를 유지합니다.

- `measured_illuminance_lux=null`
- `absolute_lux_eligible=NO`
- `photometric_calibration_status=NOT_CALIBRATED_SYNTHETIC_PROXY`
- `calibrated_to_lux=false`

따라서 `P0`–`P5`별 모델 점수를 “50–1600 lux 실성능”으로 보고하면 안 됩니다. 실측 lux 성능은 별도 실물 촬영에서 meter 위치·교정 상태·camera exposure를 기록한 뒤 평가해야 합니다.

## 한 화면의 다중 광원과 그림자

네 rig는 서로 다른 방향·상대 세기·CCT proxy를 가진 광원 2–3개를 한 장면에 동시에 적용합니다.

- `dual_opposed_neutral`: 좌우 대향 2광원
- `dual_cross_mixed_cct`: 대각선 warm/cool 2광원
- `triple_asymmetric`: 상부 key, 좌하단 fill, 우측 rim의 비대칭 3광원
- `triple_raking`: 좌우 저각 rake와 상부 soft light의 3광원

광원 각도는 `image_xy_clockwise_degrees`라는 2D image-plane 좌표계의 rendering proxy입니다. 실제 fixture의 3D 방위각·고도각 또는 광도 측정값으로 취급하지 않습니다.

양의 조명 증가는 component 영역에만 적용하고 검정 belt에는 별도 양의 광원을 더하지 않습니다. 반면 물체가 belt에 만드는 그림자는 물리적 위치 관계를 표현하기 위해 component 외부에 존재합니다. 두 shadow regime 모두 contact shadow와 directional cast shadow를 조합합니다.

- `soft_contact_dominant`: 부드러운 접촉 그림자 중심
- `defined_directional`: 방향성이 더 뚜렷한 투영 그림자 중심

각 장면에는 합성된 총 shadow attenuation mask도 저장합니다.

## Paired condition과 split 계보

동일한 v4 composition에서 만든 두 variant는 rig, proxy bin, shadow regime이 모두 다르지만 같은 `composition_family_id`를 공유합니다. source v4 장면과 두 variant는 항상 같은 split에 있어야 합니다.

- v5 전체를 train에만 사용합니다.
- 동일 `composition_family_id`의 두 variant를 train/validation/test로 나누지 않습니다.
- v4 source를 validation/test에 두고 v5 variant를 train에 두는 식의 교차 분할도 금지합니다.
- instance crop, mask, COCO/YOLO export를 별도 sample처럼 다시 분할하지 않습니다.

모든 v5 장면은 동일한 synthetic-restored physical base 계보를 상속하므로, v5 내부 random split은 independent-specimen 평가가 아닙니다.

## Paired-clean defect visibility

각 장면은 동일한 composition·lighting·shadow·sensor 조건의 defect/clean pair로 내부 렌더링됩니다. 저장된 defect image와 paired-clean reference의 결함 semantic 영역 차이를 비교해 detector 입력 크기에서 defect mean absolute delta와 changed fraction gate를 검사합니다. 이 비교는 결함이 조명 변화 때문에 사라지지 않았는지 확인하는 합성 QA이며, clean reference를 실제 정상품으로 승격시키는 절차가 아닙니다.

검정 background luma, component dark/saturation, component 밖 positive light spill, paired-clean post-JPEG spill, shadow coverage/attenuation도 자동 gate에 포함됩니다. 정확한 gate와 실행 결과, file hash, payload 크기는 release의 `annotations/release.json`을 기준으로 확인합니다.

## Annotation 형식

v4와 동일하게 component status와 defect localization을 분리합니다. 두 namespace의 class ID를 합치면 안 됩니다.

| 경로 | 내용 |
|---|---|
| `images/train/*.jpg` | 768개 multi-light 장면 |
| `annotations/coco/component_status_train.json` | 8-class component status COCO annotation |
| `annotations/coco/defects_train.json` | 7-class defect localization COCO annotation |
| `labels/yolo_component_status/train/*.txt` | component status YOLO bbox |
| `labels/yolo_defects/train/*.txt` | defect localization YOLO bbox |
| `masks/component_visible_instances/train/*.png` | 16-bit component instance-ID mask |
| `masks/defect_semantic/train/*.png` | 8-bit defect semantic mask |
| `masks/shadow_attenuation/train/*.png` | combined contact/directional shadow attenuation mask |
| `annotations/manifest.csv` | 장면별 경로·hash·condition·QC·family |
| `annotations/instances.jsonl` | 3,840개 instance별 status·lineage·lighting·visibility |
| `annotations/lighting_scenes.jsonl` | 장면별 rig·proxy·shadow·sensor·QC |
| `annotations/light_sources.jsonl` | 개별 light source의 2D 방향·상대 세기·CCT proxy |
| `annotations/condition_matrix.csv` | 48개 condition cell과 class 균형 |
| `annotations/release.json` | release 수량·hash·제약·payload summary |
| `../../annotations/synthetic_v5_illumination_human_qa.csv` | 조건 grid·overlay·paired 비교·대표 P0/P5 및 shadow mask 시각 QA 기록 |

## 미리보기

- [원본 조건 grid](../synthetic/v5_illumination/contact_sheet.jpg)
- [Annotation overlay grid](../synthetic/v5_illumination/contact_sheet_overlay.jpg)
- [동일 composition의 paired 조건 비교](../synthetic/v5_illumination/paired_condition_comparison.jpg)

## 생성과 검증

```powershell
py -3.14 -B scripts\generate_synthetic_v5_illumination.py --force
py -3.14 -B scripts\validate_synthetic_v5_illumination.py
```

Validator는 전체 release의 수량·condition balance·계보·COCO/YOLO/mask·조명 source·shadow·paired-clean 가시성·hash·deterministic replay를 검사합니다. 이는 dataset integrity 검증이며 학습된 모델의 검출 정확도 시험이 아닙니다. 개별 장면의 수동 판독 여부는 manifest의 `human_verified` 필드를 따르며, 자동 gate 통과를 사람의 결함 판독으로 표현하면 안 됩니다.

사람이 수행한 표본·contact-sheet 시각 검토 범위와 결과는 [`annotations/synthetic_v5_illumination_human_qa.csv`](../annotations/synthetic_v5_illumination_human_qa.csv)에 별도로 기록합니다. 이 기록도 합성 장면의 시각 무결성 확인이지 실제 specimen의 결함 판정 또는 실환경 성능 검증은 아닙니다.

## 실물 조명 데이터로 이어가기

50/100/200/400/800/1600 lux는 [실물 조명 촬영 프로토콜](REAL_LIGHTING_CAPTURE_PROTOCOL.md)의 계획 목표점입니다. 실제 촬영값을 해당 수치로 간주하지 말고 [`real_lighting_capture_template.csv`](../annotations/real_lighting_capture_template.csv)에 lux meter 실측값과 측정 위치·교정 정보, camera exposure, 광원별 방향·거리·CCT, 그림자 상태, specimen/session/family를 기록해야 합니다. 현재 repository에 이 프로토콜로 추가된 실물 이미지는 0장입니다.

권장 사용법은 v4/v5를 detector의 auxiliary train data로만 사용하고, 동일 family에 과도한 sampling weight가 실리지 않도록 source family 단위로 sampling하는 것입니다. 최종 threshold와 성능은 실제 normal·defect specimen을 분리한 locked validation/test에서 class별 precision, recall, AP와 false accept/reject로 결정해야 합니다.

## 이용 제한

이 release는 공개 열람용이며 open source 또는 open data가 아닙니다. 권리가 성립하고 `hkjung1011`이 보유하는 범위에서 © 2026 hkjung1011. All Rights Reserved. 사전 서면 허가 없는 복제·수정·외부 재배포·재호스팅·상업 이용·다른 dataset 편입·AI/ML 학습/평가/fine-tuning·파생물 제작은 허가되지 않습니다. 세부 조건은 [LICENSE_STATUS.md](../LICENSE_STATUS.md)를 따릅니다.
