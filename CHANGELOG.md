# Changelog

## v8 / synthetic-v5-illumination - 2026-09-01

- v4의 384개 multi-instance composition을 중립 replay하고 composition별 서로 다른 조명 variant 2개, 총 768개 train-only 장면 추가
- 장면당 component 5개, 총 3,840개 instance와 8개 status class 각 480개 구성
- 2–3개 광원을 동시에 쓰는 4개 rig × synthetic illuminance proxy `P0`–`P5` × 2개 shadow regime의 48개 full condition cell을 cell당 16 scenes로 균형화
- `capture_plan_target_lux` 50/100/200/400/800/1600은 향후 실물 촬영 목표이며 실측 lux가 아님을 명시하고 `measured_illuminance_lux=null`, `absolute_lux_eligible=NO` 고정
- 검정 belt에는 양의 조명을 더하지 않고 component-only positive light, contact shadow와 directional cast shadow를 합성하며 shadow attenuation mask 추가
- 동일 composition의 두 variant와 source v4 장면이 같은 `composition_family_id`를 공유하도록 해 split leakage 차단
- 동일 lighting·shadow·sensor의 paired-clean reference로 class별 결함 가시성과 post-JPEG spill 자동 gate 추가
- COCO/YOLO, component instance mask, defect semantic mask 외에 장면별 lighting, 개별 light source, condition matrix metadata 추가
- 실측 조명·그림자·camera 설정을 기록하는 99-column capture template, 촬영 프로토콜과 manifest validator 추가; 해당 프로토콜로 추가된 실물 이미지는 0장
- 전체 synthetic image 수를 4,210장으로 갱신하고 All Rights Reserved 공개 열람 조건 유지

## v7 / synthetic-v4-conveyor - 2026-09-01

- 1280×720 검정 컨베이어 배경의 train-only 다중 부품 장면 384장 추가
- 장면당 겹침·잘림 없는 component 5개, 총 1,920개 instance와 8개 status class 각 240개 균형화
- `normal_proxy`와 7개 결함 class를 분리하고, `normal_proxy`는 실제 정상품이 아닌 paired-clean 합성본으로 명시
- neutral/warm/cool/side 4개 component-only 조명 profile을 각 96개 장면에 적용하고 background light-spill gate 추가
- component status와 defect localization을 분리한 COCO/YOLO label, component instance-ID mask, defect semantic mask 제공
- v2 `gradient_train` parent 168개만 사용하고 family 계보와 train-only 고정으로 validation/test 파생 누수 차단
- 전체 384개 장면 deterministic batch replay, label/hash/QC/payload 검증 추가
- Python 3.14.6 / NumPy 2.5.1 / Pillow 12.3.0 / libjpeg 8.0 runtime과 helper script·requirements SHA를 고정
- 기존 v2-only `model_final.pt` 3개는 다중 부품 전체 장면 detector가 아니며 v4 성능은 실물 데이터에서 검증되지 않았음을 명시
- release 명칭은 과거 실제 사진 label audit `v4`와 혼동을 피하기 위해 changelog version `v7`로 기록

## v6 / synthetic-v3-conditions - 2026-08-31

- v2 고정 split의 `gradient_train` parent 168장만 source로 사용
- parent별 6개 조명·촬영조건, 총 1,008장의 train-only image/mask 추가
- under/over exposure, warm/cool directional light, shadow/vignette, specular/sensor condition 균형 적용
- parent image/mask SHA-256, lineage/family ID, profile/seed/parameter manifest 추가
- validation/test parent 파생 0 및 parent-family leakage gate 추가
- post-JPEG 512/224 defect visibility와 luma/clipping deterministic replay 검증
- ImageGen 조명 편집 시험본은 geometry/표면 invariant 미충족으로 release에서 제외

## v5.1 / synthetic-v1-450 - 2026-08-31

- 기존 synthetic-v1을 변경하지 않고 별도 random seed `7809250` release 추가
- 9개 primary class 각각 50장, 총 450장의 512×512 train-only 이미지 생성
- 고유 ID prefix `syn-v1-450-*`와 별도 generator/config/provenance 사용
- 이미지별 semantic mask, bbox, multi-label, severity, seed, recipe, SHA-256 자동 라벨 포함
- 기존 release와 sample ID, sample seed, image SHA-256 중복 0 검증
- contact sheet 27장과 severe 대표 8장 수동 visual QA 기록
- 실제 normal과 real validation/test 부재는 계속 `NOT VERIFIED`

## v5 / synthetic-v1 - 2026-08-31

- built-in image generation edit로 `171603` 후면 scratch를 제거한 `synthetic_restored` clean base 추가
- 9개 primary class 각 100장, 총 900장의 512×512 train-only synthetic dataset 생성
- scratch, surface spot, discoloration, contamination, lead breakage, body chip, body crack와 multi-defect 지원
- 이미지별 semantic PNG mask, bbox, multi-label, seed, recipe, severity, SHA-256 자동 생성
- ROI containment, visibility delta, class balance, image/mask/hash 검증 script 추가
- contact sheet 27장과 severe 8장에 대한 OpenAI Codex 수동 visual QA 기록
- 실제 normal/real validation/test 부재는 `NOT VERIFIED`로 유지

## v4 - 2026-08-31

- 원본 17장과 단일 부품 crop을 OpenAI Codex가 직접 수동 비교 판독
- 사진별 `manual_image_label`, 검토자, 검토일, 학습 포함/제외 결정을 추가
- 확정 scratch 6장, surface spot 2장, lead deformation review 1장 유지
- 확정 정상품 0장과 breakage/discoloration/contamination confirmed 0장을 재확인
- validator 기준을 `image_labels_v4.csv`로 전환

## v3 - 2026-08-30

- specimen 기준 `OK 0 / NG 13 / HOLD 4` 분리
- `171929`, `171931` scratch confirmed 승격
- `171611`, `171925`에 `surface_spot_unknown=positive` 추가
- `171925`를 `lead_deformation=review`, `breakage=ambiguous`로 보수 판정
- KEC `KIA7809AF` 공식 marking, datasheet, DPAK source 추가
- 재현 가능한 hash 및 label validation script 추가
