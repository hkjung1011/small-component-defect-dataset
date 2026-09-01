# Synthetic v4 black-conveyor multi-instance release

`synthetic-v4-conveyor`는 검정 컨베이어 위의 여러 부품을 한 장에서 찾고 상태를 구분하는 detector/segmenter 학습용 합성 release입니다. 1280×720 장면 384개에 부품을 5개씩 배치해 총 1,920개 component instance를 제공합니다.

이 release는 changelog에서 `v7`입니다. `v4`라는 번호는 이미 실제 사진의 수동 label audit에 사용되었으므로, dataset 경로와 ID는 `synthetic-v4-conveyor`, 변경 이력은 `v7`로 구분합니다.

## 구성

| 항목 | 값 | 상태 |
|---|---:|---|
| Scene | 384장, 1280×720 | TRAIN_ONLY |
| Component/scene | 5개 | CONFIRMED |
| Component instance | 1,920개 | CONFIRMED |
| Status class | 8개, class당 240개 | CONFIRMED |
| Defect localization | 7개 class, 1,680개 instance | CONFIRMED |
| `normal_proxy` | 240개, defect target 없음 | SYNTHETIC PROXY |
| Lighting profile | 4개, profile당 96 scenes | CONFIRMED |
| Split | `train` | FIXED |
| Evaluation eligible | `NO` | CONFIRMED |
| Classification eligible | `NO` | CONFIRMED |
| 신규 실물 specimen | 0 | CONFIRMED |
| Release inventory | 1,936 files / 143,362,066 bytes (136.72 MiB) | VALIDATED |

8개 component status는 다음과 같습니다.

- `normal_proxy`
- `scratch`
- `surface_spot`
- `discoloration`
- `contamination`
- `lead_breakage`
- `body_chip`
- `body_crack`

각 status는 전체 release에서 240개, 각 조명 profile에서 60개입니다. `normal_proxy`는 결함 parent와 대응하는 paired-clean 합성 부품입니다. 필수 면의 실물 검사나 전기 검사를 통과한 실제 `OK_confirmed`가 아니며, 정상품 specimen 수로 집계하면 안 됩니다.

## 검정 belt와 component-only 조명

배경은 OpenAI built-in image generation으로 만든 빈 matte-black conveyor texture입니다. 생성 prompt와 SHA-256은 `synthetic/sources`와 [SOURCES.md](SOURCES.md)에 보존합니다. 부품과 결함 geometry는 ImageGen으로 다시 그리지 않고, 고정 source·mask에서 deterministic composition합니다.

국소 조명은 다음 네 profile입니다.

- `neutral_component_spot`
- `warm_component_spot`
- `cool_component_spot`
- `side_component_spot`

노출·색온도·gradient·hotspot 변환은 component alpha 내부에만 적용합니다. 검정 belt에는 별도의 국소 광원을 합성하지 않으며, 배경에는 어두운 belt texture randomization과 최종 scene sensor noise/blur/JPEG만 적용됩니다. component 외부의 양의 light spill은 별도 QC gate로 제한합니다. 접촉 그림자는 component 위치에만 국소적으로 추가합니다.

장면은 4×2 배치 grid에서 5개 cell을 선택하고, 회전·scale·위치 jitter를 적용합니다. component overlap과 frame truncation은 허용하지 않습니다.

현재 release는 class·조명·위치 균형을 확인하기 위한 balanced pilot이므로 한 장면의 5개 status가 모두 서로 다릅니다. 실제 생산 prevalence처럼 정상 다수, 동일 status 반복, 결함 희소 장면을 모사하지는 않습니다. detector 배포 전에는 이 구성을 별도 `production_mix` train release로 추가하고, 실제 독립 사진에서 오검출률을 검증해야 합니다.

## Annotation 형식

한 장면에는 status detection과 defect localization이라는 두 label namespace가 있습니다. 두 namespace의 class ID를 섞으면 안 됩니다.

| 경로 | 내용 |
|---|---|
| `images/train/*.jpg` | 384개 합성 장면 |
| `annotations/coco/component_status_train.json` | 8-class component status COCO bbox |
| `annotations/coco/defects_train.json` | 7-class defect COCO bbox; `normal_proxy` 제외 |
| `labels/yolo_component_status/train/*.txt` | COCO component status에서 파생한 YOLO bbox |
| `labels/yolo_defects/train/*.txt` | COCO defect localization에서 파생한 YOLO bbox |
| `masks/component_visible_instances/train/*.png` | 16-bit component instance-ID mask |
| `masks/defect_semantic/train/*.png` | 8-bit defect semantic mask, background 0 |
| `annotations/manifest.csv` | 장면별 경로·hash·profile·QC·계보 |
| `annotations/instances.jsonl` | 1,920개 instance별 status·bbox·source·placement·visibility |
| `annotations/release.json` | release 수량·hash·제약·payload summary |
| `annotations/synthetic_v4_conveyor_human_qa.csv` | overview와 profile별 raw/overlay contact sheet 시각 QA 기록 |

COCO annotation에는 full-image uncompressed RLE `segmentation`을 넣고, 동일한 canonical raster mask 경로도 annotation attribute에 연결합니다. 독립 validator가 RLE decode 결과와 PNG mask를 pixel-exact로 대조합니다. YOLO text는 편의용 bbox derived export입니다.

Component status ID는 YOLO `0..7`, COCO `1..8`입니다. Defect localization ID는 YOLO `0..6`, semantic/COCO `1..7`입니다. `normal_proxy`는 component status target만 가지며 defect YOLO/semantic target은 없습니다.

## Source 계보와 split

Source는 `synthetic-v2-700` 고정 split의 `gradient_train` parent 168개, 결함 class당 24개입니다. validation/test parent는 사용하지 않습니다. 각 결함 parent는 v4에서 10회 사용됩니다. `normal_proxy` 240개도 같은 168개 parent의 paired-clean으로 만들며 parent당 1~2회만 사용합니다.

모든 장면은 하나의 synthetic-restored physical base family에서 파생되므로 독립 specimen이 아닙니다. 다음 규칙을 지켜야 합니다.

- release 전체를 train에만 둡니다.
- scene, crop, instance, parent, paired-clean 파생본을 val/test로 재분할하지 않습니다.
- 동일 `family_split_id`와 `composition_family_id`의 모든 파생물을 같은 train family로 유지합니다.
- v4 장면으로 validation/test metric을 계산하지 않습니다.

## 전체 batch 검증

생성기는 각 장면의 component와 paired-clean을 동일 placement·lighting·sensor 조건으로 렌더링하고, detector 입력 1024 기준 결함 가시성을 검사합니다. 독립 validator는 384개 장면 전체를 deterministic replay하여 다음을 일괄 검증합니다.

- scene 384개, instance 1,920개, 장면당 instance 5개
- 8 status × 240개, 4 profile × 96 scenes, status×profile 균형
- COCO/YOLO bbox, component instance mask, defect semantic mask의 상호 일치
- source SHA-256, parent split·재사용 횟수, family 계보, validation/test 파생 0
- component 겹침·잘림·가시 fraction, defect 크기·paired-clean 변화량
- 검정 background luma, component/background 대비, component 외부 light spill
- image/label/mask/config/generator hash와 payload 제한

Byte-exact replay 환경은 Python `3.14.6`, NumPy `2.5.1`, Pillow `12.3.0`, libjpeg `8.0`, zlib `1.3.1.zlib-ng`로 고정합니다. Python RNG와 JPEG runtime이 달라지면 동일 seed라도 byte 결과가 달라질 수 있으므로 validator는 다른 runtime을 setup failure로 거부합니다. Python을 준비한 뒤 `requirements-synthetic.txt`의 exact dependency를 설치해야 합니다. Generator가 import하는 세 helper script와 requirements lock도 SHA-256으로 고정됩니다.

```powershell
py -3.14 -B scripts\generate_synthetic_v4_conveyor.py --force
py -3.14 -B scripts\validate_synthetic_v4_conveyor.py
```

이 명령은 dataset 전체 파일과 label을 batch 검증합니다. 학습된 모델로 여러 새 사진의 결함을 추론하는 명령은 아닙니다. 새 사진 batch inference에는 v4 label로 학습한 detector/segmenter와 별도 inference pipeline이 필요합니다.

## 모델 사용 한계

Repository의 C0/C2 `model_final.pt` 6개는 `synthetic-v2-700` 단일 부품
crop을 판정하는 7-class ResNet-18 classifier입니다. C2 세 모델은 v3
조명·카메라 variant를 학습에 추가했지만, 어느 모델도 전체 장면에서
여러 component를 찾거나 `normal_proxy`를 판정하도록 학습되지 않았으므로
v4 장면 detector로 사용할 수 없습니다.

권장 구조는 전체 장면에서 component/defect를 찾는 detector 또는 segmenter를 먼저 학습하고, 필요하면 검출 crop에 별도 classifier를 적용하는 2-stage pipeline입니다. 기존 checkpoint를 crop 보조 분류기로 재사용하는 경우에도 v4 domain과 실제 컨베이어 사진에서 별도 검증해야 합니다.

현재 다음 항목은 `NOT VERIFIED`입니다.

- 실제 검정 컨베이어 사진의 detection/segmentation 성능
- 실제 정상품과 실제 결함 specimen에 대한 precision, recall, mAP, false reject/accept rate
- 조명·카메라·생산 lot·부품 방향 변화에 대한 실물 domain generalization
- 전기적 정상성, 진품 여부, 출하 합격 여부

따라서 synthetic validation score를 실제 성능이나 양산 수율로 표현하면 안 됩니다. 독립 실물 specimen 단위의 locked validation/test set을 확보한 뒤 class별 metric과 현장 threshold를 결정해야 합니다.

## 이용 제한

이 release는 공개 열람용이며 open source 또는 open data가 아닙니다. 권리가 성립하고 `hkjung1011`이 보유하는 범위에서 © 2026 hkjung1011. All Rights Reserved. 사전 서면 허가 없는 복제·수정·외부 재배포·재호스팅·상업 이용·다른 dataset 편입·AI/ML 학습/평가/fine-tuning·파생물 제작은 허가되지 않습니다. 세부 조건은 [LICENSE_STATUS.md](../LICENSE_STATUS.md)를 따릅니다.
