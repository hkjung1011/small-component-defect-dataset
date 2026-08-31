# 부품 식별: KEC KIA7809AF

## 선정 식별

| 항목 | 값 | 상태 |
|---|---|---|
| Manufacturer | KEC Corporation | CONFIRMED |
| Base model | KIA7809AF | CONFIRMED |
| 정식 기능 | Three-terminal positive standard voltage regulator | CONFIRMED |
| Nominal output | 9 V | CONFIRMED |
| Output current | 1 A | CONFIRMED |
| Maximum input | 35 V | CONFIRMED |
| Typical dropout | 2 V at 1 A | CONFIRMED |
| Package family | DPAK / DPAK(3), TO-252 계열 | CONFIRMED |
| Exact DPAK variant | DPAK 또는 DPAK(3) | TO_VERIFY |
| Exact ordering suffix | 사진 마킹에 없음 | NOT VERIFIED |
| Lot marking | 208 = 2022년 08주차 | CONFIRMED |

## 마킹 대조

사진을 180도 회전해 읽으면 다음 3줄입니다.

```text
KIA
7809AF
208
```

KEC 공식 `M KIA7809AF` marking specification은 `KIA`를 trade name, `7809AF`를 device name, 3자리 값을 lot number로 정의합니다. 첫 자리는 2020-2029년의 마지막 숫자이고 뒤 두 자리는 생산 주차이므로 `208`은 2022년 08주차입니다.

사진의 DPAK 계열 body, 후면 방열 tab, 3개 lead 위치도 제조사 package drawing과 일치합니다. 다만 사진에는 치수 기준자가 없고 DPAK와 DPAK(3)의 세부 형상이 유사하므로 정확한 variant와 ordering suffix는 확정하지 않았습니다.

## 주요 기능 정보

- Pin 1: INPUT
- Pin 2: GND
- Pin 3: OUTPUT
- Internal thermal overload protection
- Internal short-circuit current limiting
- KIA7809AF nominal output: 9.0 V
- Datasheet test 조건상 output current up to 1 A

## 정상품 판정과의 관계

모델 식별은 외관 마킹의 식별일 뿐입니다. 진품 여부와 전기적 정상 여부는 확인되지 않았습니다. 최소 확인에는 무부하 출력, 100 mA 및 목표 부하에서 regulation, 소비전류, short/thermal protection의 안전한 별도 시험이 필요합니다.

공식 URL과 문서 revision 및 SHA-256은 [SOURCES.md](SOURCES.md)에 기록했습니다.
