# Changelog

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
