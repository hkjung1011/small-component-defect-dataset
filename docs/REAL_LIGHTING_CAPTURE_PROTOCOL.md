# 실물 조명 조건 촬영 protocol v2

이 문서는 검정 conveyor belt 위 소형 부품을 실제 광원으로 촬영할 때 조도, CCT, 광원 기하, camera 설정, shadow, specimen 계보를 재현 가능하게 기록하는 절차입니다. 모든 행은 `source_domain=REAL_CAPTURE`, `illuminance_value_kind=MEASURED`로 고정하며 합성 데이터의 `synthetic_illuminance_proxy_bin`과 실측 `illuminance_lux_measured`를 혼용하지 않습니다.

## 1. 적용 범위와 판정 단위

- manifest 1행은 저장된 실물 이미지 1장과 primary physical specimen 1개입니다. 현재 schema는 **single-primary-specimen capture 전용**입니다. 한 frame에 여러 specimen이 있으면 이 manifest만으로 instance별 specimen/status/split 계보를 표현할 수 없으므로 evaluation에 사용하지 말고 별도의 per-instance manifest와 validator를 먼저 정의해야 합니다.
- 동일 이미지에 동시에 켜진 광원은 `lights_json` 배열 한 개에 모두 기록합니다. 광원별 행 복제는 금지합니다.
- 사진 수와 독립 specimen 수를 분리합니다. 같은 physical part의 면·회전·조명·노출 변형은 모두 같은 `specimen_group_id`, `split_group_id`, `split`을 사용합니다.
- `session_group_id`도 하나의 split만 사용합니다. 같은 날 같은 rig에서 연속 촬영한 frame이 train과 test에 동시에 들어가는 것을 막기 위한 규칙입니다.
- `OK_confirmed`는 필수 면 검사와 적용 가능한 기능 검사를 통과한 specimen에만 사용합니다. 한 관찰면에서 결함이 보이지 않는다는 이유만으로 OK를 부여하지 않습니다.
- `HOLD_unverified`, provisional label, QC FAIL/HOLD 이미지는 evaluation에 사용하지 않습니다.
- lux/CCT meter 표시 화면, logger export 또는 원본 측정 record에 evidence ID를 부여하고 원본 byte SHA-256을 각각 기록합니다. 합성 proxy, 목표 lux, 조명 controller 설정값은 measurement evidence로 사용할 수 없습니다.

## 2. 좌표계와 각도

촬영 전에 station 좌표계를 고정하고 사진으로 남깁니다.

- 원점: specimen 중심을 belt 평면에 투영한 점
- `+X`: belt 이송 방향
- `+Y`: 위에서 봤을 때 `+X`의 왼쪽
- `+Z`: belt에서 위쪽
- azimuth: `+X=0°`, `+Y=90°`, `-X=180°`, `-Y=270°`; `+Z`에서 내려다볼 때 반시계 방향
- elevation: belt 평행 방향 `0°`, 정수직 상부 `90°`
- 광원 distance: 발광면 중심에서 원점까지의 직선거리
- camera distance: lens entrance pupil 기준점에서 원점까지의 직선거리. 기준점이 불명확하면 제조사 기준면과 보정 offset을 station 기록에 남깁니다.

좌표계가 바뀌면 기존 ID를 재사용하지 말고 새로운 `station_id`, `lighting_profile_id` 또는 `camera_profile_id`를 발급합니다.

## 3. 검정 belt와 component-only 조명

1. 실제 사용할 matte-black belt 또는 동일 재질 coupon을 설치하고 `background_id`, 재질, 오염·마모·주름 상태를 기록합니다.
2. room light, window light, indicator LED 등 비제어 광원을 차단합니다.
3. hood, flag, snoot 또는 baffle로 직접광 footprint를 부품에 제한합니다. 이 프로젝트의 모든 행은 `background_color=black`, `component_only_illumination=YES`, `direct_belt_illumination=NO`여야 하며 validator가 고정값으로 강제합니다.
4. 검정 바닥을 만들기 위해 belt pixel을 디지털로 0에 clamp하거나 합성 배경으로 교체하지 않습니다. 실제 stray light와 sensor black level이 이미지에 남아야 합니다.
5. belt와 component 사이의 contact shadow, 광원 방향에 따른 directional shadow를 제거 대상으로 간주하지 않고 별도 측정합니다.

## 4. 계측기와 교정

### 4.1 lux 측정

1. lux meter의 자산 ID, serial, 최근 교정일, 교정 만료일, certificate ID를 기록합니다.
2. 교정 유효기간이 촬영일과 측정 시각을 모두 포함하는지 확인합니다. 둘 중 하나라도 교정 전이거나 만료 후이면 validator가 FAIL합니다.
3. 실제 촬영 광원을 모두 켜고 안정화한 뒤 측정합니다. 여러 광원의 값을 산술 합산해 대신하지 않습니다.
4. sensor head 위치는 다음 중 하나로 고정합니다.
   - `specimen_removed_at_origin`: specimen을 잠시 제거하고 원점에서 측정
   - `adjacent_in_plane`: 원점 옆의 고정 offset에서 측정
   - `in_situ_probe`: specimen을 둔 상태에서 고정 probe로 측정
5. 위치 `x/y/z`, sensor normal azimuth/elevation, 좌표 frame을 mm/degree로 기록합니다.
6. 동일 위치에서 반복 측정하고 실제 평균을 `illuminance_lux_measured`, 반복 수를 `illuminance_repeat_count`, sample standard deviation을 `illuminance_std_lux`에 기록합니다. 1회 측정이면 표준편차는 0입니다.
7. 측정 시각을 UTC ISO-8601 형식으로 기록합니다. 표시값을 목표값으로 치환하거나 반올림한 등급명만 남기지 않습니다.
8. 촬영을 시작하기 전에 해당 session에 허용할 최대 측정-촬영 시각 차이를 `measurement_capture_offset_limit_seconds`로 정합니다. validator에는 default가 없으며 각 행에 실제 승인값을 기록해야 합니다. lux와 CCT 측정 시각 모두 이 한계 이내여야 하고 같은 `session_id`/`session_group_id`에서 값을 바꿀 수 없습니다.
9. lux 측정 원본 evidence의 `lux_measurement_evidence_id`와 `lux_measurement_evidence_sha256`을 기록합니다. 같은 evidence ID를 재사용할 때 meter, 시각, 값, 반복성, 위치가 모두 같아야 합니다.

### 4.2 CCT 측정

- lux meter가 CCT를 지원하더라도 CCT sensor의 ID·serial·교정 이력을 별도 필드에 반복 기록합니다.
- 모든 활성 광원이 켜진 상태에서 specimen plane의 combined CCT를 `cct_measured_k`로 기록합니다. `meter_location_frame`과 위치·sensor normal field는 lux와 CCT 측정에 공통으로 적용하며, 두 sensor 위치가 다르면 동일 위치로 재측정하거나 별도 capture row/profile로 분리합니다.
- 광원 catalog 값은 `lights_json.nominal_cct_k`, 실측값은 `cct_measured_k`입니다. 둘을 대체 사용하지 않습니다.
- CCT 측정 원본 evidence의 `cct_measurement_evidence_id`와 `cct_measurement_evidence_sha256`을 기록합니다. CCT 측정 시각도 CCT meter 교정 window와 session offset limit를 모두 만족해야 합니다.

## 5. camera 설정

각 촬영에서 다음 값을 camera metadata 또는 제어 software에서 직접 읽어 기록합니다.

- camera make/model/serial, lens model, focal length
- 해상도, bit depth, file format
- camera azimuth/elevation/distance
- `exposure_mode`, 실제 `exposure_time_us`, ISO 또는 industrial-camera `analog_gain_db`
- aperture f-number, exposure compensation
- WB mode, WB Kelvin, 가능하면 red/blue gain
- focus mode

재현용 baseline과 locked validation/test에는 manual exposure, manual/custom WB, manual/fixed focus를 권장합니다. Auto mode를 사용한 조건도 기록할 수 있지만 별도 `camera_profile_id`와 `condition_id`로 분리합니다. 광량 변화와 exposure 변화의 효과를 동시에 해석하지 않도록 한 profile에서는 설정을 고정합니다.

## 6. 광원 정의와 multi-light

`light_count`는 동시에 활성화된 광원의 수이며 최대 8개입니다. `multi_light_mode`는 한 개면 `single`, 둘 이상이면 `simultaneous`입니다. 순차 조명을 software로 합친 composite는 원본 capture로 사용하지 않습니다. 단일광원 baseline은 허용하지만 완성 manifest 전체에는 simultaneous multi-light 행이 최소 1개 있어야 합니다.

`lights_json`은 아래 key를 정확히 포함한 JSON array입니다. key와 광원 배열의 표기 순서는 무관하며 validator가 의미 기준으로 canonicalize합니다.

```json
[
  {
    "light_id": "LIGHT-KEY-01",
    "role": "key",
    "source_type": "LED_PANEL",
    "azimuth_deg": 45.0,
    "elevation_deg": 35.0,
    "distance_mm": 320.0,
    "nominal_cct_k": 5000.0,
    "power_setting_pct": 70.0,
    "diffuser_id": "DIFFUSER-A",
    "polarizer_angle_deg": null
  },
  {
    "light_id": "LIGHT-FILL-01",
    "role": "fill",
    "source_type": "LED_BAR",
    "azimuth_deg": 225.0,
    "elevation_deg": 25.0,
    "distance_mm": 410.0,
    "nominal_cct_k": 5000.0,
    "power_setting_pct": 25.0,
    "diffuser_id": "DIFFUSER-B",
    "polarizer_angle_deg": null
  }
]
```

허용 role은 `key`, `fill`, `rim`, `auxiliary`입니다. azimuth는 `[0,360)`, elevation은 `[0,90]`, polarizer angle은 `null` 또는 `[0,180)`입니다. 광원의 mount 위치, diffuser 또는 polarizer가 바뀌면 같은 `lighting_profile_id`를 재사용하지 않습니다.

촬영 전에 의도한 최소 active-light 입사각 분리를 `minimum_active_light_angular_separation_deg`에 `0 < value <= 180`으로 지정합니다. validator는 azimuth/elevation을 3-D unit vector로 변환해 simultaneous 행의 모든 pair 중 최대 구면각이 지정 threshold 이상인지 확인합니다. 이 값에는 default가 없으며 같은 lighting profile, session, session group에서 변경할 수 없습니다.

## 7. shadow 기록

- 완성 manifest 전체에는 `shadow_present=YES`인 행이 최소 1개 있어야 합니다.
- `shadow_present=NO`이면 `shadow_type=none`, `shadows_json=[]`입니다.
- shadow가 있으면 summary `shadow_type`은 `contact`, `directional`, `contact_and_directional`, `uncontrolled` 중 하나이고 `shadows_json`에는 실제 관찰 shadow를 각각 기록합니다.
- directional shadow는 반드시 해당 capture의 active `light_id`를 참조합니다. 동시 다중광원에서 서로 다른 광원이 만든 directional shadow는 별도 object로 기록합니다.
- contact/uncontrolled shadow는 광원 하나에 임의 귀속하지 않고 `source_light_id`와 `direction_azimuth_deg`를 `null`로 둡니다.
- contrast ratio의 권장 정의는 동일 exposure 이미지의 인접 belt ROI를 사용한 `(L_belt - L_shadow) / max(L_belt, epsilon)`입니다. 계산에 사용한 linearization/ROI 방법을 각 `measurement_method`에 기록하고 profile 간 동일하게 유지합니다.

```json
[
  {
    "shadow_id": "SHADOW-CONTACT-01",
    "source_light_id": null,
    "shadow_type": "contact",
    "direction_azimuth_deg": null,
    "length_mm": 1.8,
    "contrast_ratio": 0.32,
    "measurement_method": "image_roi_linear_luma"
  },
  {
    "shadow_id": "SHADOW-KEY-01",
    "source_light_id": "LIGHT-KEY-01",
    "shadow_type": "directional",
    "direction_azimuth_deg": 225.0,
    "length_mm": 6.4,
    "contrast_ratio": 0.41,
    "measurement_method": "image_roi_linear_luma"
  }
]
```

위 예시의 summary는 `shadow_type=contact_and_directional`입니다. 동일 active light를 source로 하는 directional shadow object는 capture당 하나만 허용합니다.

## 8. 촬영 순서

1. specimen에 영구 `specimen_id`를 부여하고 lot, 상태, 확인 근거를 등록합니다.
2. station 좌표계, black belt 상태, camera와 광원 mount를 고정합니다.
3. 계측기 교정 유효기간을 확인하고 dark/ambient stray light를 점검합니다. 촬영 전에 session의 측정-촬영 offset limit와 최소 광원 각도 분리를 승인·고정합니다.
4. camera 설정을 고정한 뒤 camera profile을 발급합니다.
5. 각 lighting profile에서 모든 광원을 안정화하고 combined lux/CCT를 실측한 뒤 meter record/screenshot/export의 evidence ID와 SHA-256을 등록합니다.
6. specimen의 필요한 view를 촬영합니다. 동일 specimen의 lighting variant는 동일 split에 둡니다.
7. component ROI의 high/low clipping 비율과 선택한 focus metric을 기록합니다. 허용값은 camera와 결함 크기에 맞춰 연구 시작 전에 별도 acceptance criterion으로 고정합니다.
8. SHA-256을 원본 저장 직후 계산하고 manifest에 기록합니다. 저장 후 재압축한 preview의 hash를 쓰지 않습니다.
9. label review와 capture QC가 끝난 후에만 validation/test의 `evaluation_eligible=YES`를 허용합니다.

## 9. 조건 matrix와 split

실제 lux 절대값은 계측 전 임의로 만들지 않습니다. 먼저 nominal setting을 정한 뒤 실제 측정값으로 조건을 구분합니다. 최소한 다음 변수를 독립적으로 포함합니다.

| 변수 | 권장 구성 | 통제 규칙 |
|---|---|---|
| 광량 | low / nominal / high nominal setting | 각 capture에 measured lux 저장 |
| 단일 광원 azimuth | 서로 다른 최소 4방향 | elevation/distance 고정 |
| elevation | 낮음 / 중간 / 높음 | azimuth/distance 고정 |
| multi-light | key only / key+fill / opposed pair | 최소 1 simultaneous row, 지정 구면각 이상 |
| shadow | contact / directional / 최소화 | 최소 1 shadow row, `shadows_json` 기록 |
| camera | baseline + 의도한 exposure variant | profile ID 분리 |
| specimen | OK/각 결함 class·severity | physical specimen ID 우선 집계 |

balanced challenge split과 실제 prevalence를 유지한 `production_mix` split을 별도로 관리합니다. 학습 시 조건을 oversampling할 수 있지만 locked real test의 class prior를 바꾸지 않습니다. `split_group_id`, `specimen_group_id`, `session_group_id` 어느 것도 서로 다른 split에 걸치면 validator가 FAIL합니다.

## 10. manifest schema

정확한 header는 `annotations/real_lighting_capture_template.csv`가 기준입니다.

| 그룹 | 주요 field | 의미 |
|---|---|---|
| domain | `source_domain`, `illuminance_value_kind` | `REAL_CAPTURE`·`MEASURED` 고정, proxy 혼입 차단 |
| 계보·split | `capture_id` … `evaluation_eligible` | specimen/session/split 누수 방지와 OK/NG 상태 |
| 배경 | `background_*` | 실제 black belt 재질과 상태 |
| camera | `camera_*`, exposure/ISO/gain/aperture/WB/focus | 실제 촬영 설정과 기하 |
| timing policy | `measurement_capture_offset_limit_seconds` | 사용자가 사전 지정한 session 측정-촬영 허용 간격 |
| lux | `lux_meter_*`, `meter_*`, `illuminance_*`, `lux_measurement_evidence_*` | 교정, 위치, 실측 lux·반복성·원본 evidence |
| CCT | `cct_meter_*`, `cct_measured_k`, `cct_measurement_evidence_*` | combined-light 실측 CCT와 원본 evidence |
| 광원 | `light_count`, `multi_light_mode`, `minimum_active_light_angular_separation_deg`, `lights_json` | 광원별 입사각·거리·nominal CCT·출력 |
| shadow | `shadow_present`, `shadow_type`, `shadows_json` | 광원별 directional/contact shadow 계보와 metric |
| QC·label | clipping/focus/QC/reviewer field | evaluation eligibility 근거 |

`visible_defect_classes_json`은 JSON list이며 허용 class는 `scratch`, `surface_spot`, `discoloration`, `contamination`, `lead_breakage`, `body_chip`, `body_crack`입니다. `specimen_status`와 한 view에서 보이는 class를 혼동하지 않습니다.

## 11. 검증 명령

validator의 built-in positive/negative test를 먼저 실행합니다.

```powershell
python scripts/validate_real_lighting_capture_manifest.py --self-test
```

실제 행이 없는 public template의 schema만 검증합니다.

```powershell
python scripts/validate_real_lighting_capture_manifest.py --schema-only
```

실제 행을 채운 manifest의 값, 그룹 누수, profile 일관성을 검증합니다.

```powershell
python scripts/validate_real_lighting_capture_manifest.py --manifest annotations/real_lighting_capture.csv
```

원본 파일 존재 여부와 SHA-256까지 확인합니다.

```powershell
python scripts/validate_real_lighting_capture_manifest.py --manifest annotations/real_lighting_capture.csv --check-files --images-root data/real_lighting
```

빈 template은 일반 mode에서 FAIL하는 것이 정상입니다. 데이터 행이 없을 때 PASS가 필요한 경우에만 `--schema-only`를 사용합니다. populated manifest에 `--schema-only`를 사용하면 의미 검증 누락을 막기 위해 FAIL합니다.
