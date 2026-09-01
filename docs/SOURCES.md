# Official sources

조회일: 2026-08-30

| 자료 | 공식 URL | 문서 정보 | 공식 URL에서 받은 원문의 SHA-256 |
|---|---|---|---|
| KIA7809AF product page | https://www.kec.co.kr/kr/product/product_view.asp?idx=1859 | KEC product page | - |
| Product datasheet | https://www.kec.co.kr/kr/product/image_product.asp?gu=1&idx=1859 | `KIA7805-24AF_API`, 2022-03-29, Rev.17 | `BEDB497F1693CDCB66E4250A82ECE810EB76BA5626295AB2708DBDD9744BDA42` |
| Marking specification | https://www.kec.co.kr/kr/product/image_product.asp?gu=2&idx=1859 | `M KIA7809AF`, 2023-06-21, Rev.1 | `D68EA44FB82C07E26F41EEADA6097B96D52669C53D881AA80BFBA41B0D8F0DFD` |
| DPAK package drawing | https://www.kec.co.kr/kr/support/image_pack.asp?idx=380&gu=2 | `DPAK PKG`, 2020-10-05, Rev.4 | `FBFBE6CD462A35AA21CCA799561C5D590887EE7D4A6A2E497BB45D77E32D7D8B` |
| DPAK(3) package drawing | https://www.kec.co.kr/kr/support/image_pack.asp?idx=382&gu=2 | `DPAK(3)`, 2018-07-13, Rev.0 | `B55274CCD189062AE20C0FED6003BF87B61EE68797896F1F9D7EE7D6AAEA44E8` |

제3자 문서의 재배포 권한을 별도로 확인하지 않았으므로 PDF 원문은 repository에 포함하지 않습니다. 위 SHA-256은 조회일에 공식 KEC URL에서 받은 원문을 검증하기 위한 값이며, 문서는 각 공식 URL에서 직접 내려받아야 합니다.

주의: product page에는 여러 ordering suffix와 EOL 관련 정보가 함께 표시됩니다. 사진의 top marking만으로 suffix와 현재 조달 상태를 특정할 수 없으므로 구매 전 KEC 또는 공인 유통사 확인이 필요합니다.

## Synthetic v4 conveyor asset provenance

생성일: 2026-08-31

`synthetic-v4-conveyor`에는 인터넷에서 수집한 컨베이어 사진이나 제3자 제품 사진을 추가하지 않았습니다. 빈 검정 belt texture만 OpenAI built-in image generation으로 만들고, 부품·결함은 기존 repository source와 deterministic mask composition으로 생성합니다.

| Asset | 생성/파생 방법 | SHA-256 |
|---|---|---|
| `synthetic/sources/conveyor_black_imagegen_v1.png` | OpenAI built-in image generation, 빈 matte-black conveyor background | `4f126b1373f98256da366b271a585939ff4a551daeb2efc0b8cd3b8ec6392543` |
| `synthetic/sources/conveyor_black_imagegen_v1.prompt.txt` | 위 background를 생성한 보존 prompt | `061ac9444a8025ed54ba1d29ce14e7315c97a5ebfc37f6a3a1e7e87cdfadbfed` |
| `synthetic/sources/nominal_component_alpha_v1.png` | v2 component ROI union에서 mount slot을 제외한 deterministic alpha | `3fbdf78090c1b3ac12b91a08bd2be20ee43e1c69f76665d5a34f12785773f043` |
| `synthetic/sources/nominal_component_alpha_v1_overlay.jpg` | alpha 경계 확인용 overlay | `477e688f473175513dd63540ac02e23def13d717e18116b2da2d379c3cc8773d` |

배경 prompt 원문과 asset hash는 repository에 함께 보존합니다. ImageGen은 빈 배경 asset에만 사용했고 부품의 결함 형상이나 label geometry를 생성·수정하는 데 사용하지 않았습니다. 네 가지 조명 profile은 generator가 component alpha 내부에만 deterministic transform으로 적용합니다.

이 provenance 기록은 공개 재사용 허가가 아닙니다. v4 asset, 합성 장면, mask와 annotation의 이용 조건은 [LICENSE_STATUS.md](../LICENSE_STATUS.md)를 따릅니다.

## Synthetic v5 illumination provenance

생성일: 2026-09-01

`synthetic-v5-illumination`에는 인터넷에서 수집한 사진이나 신규 제3자 asset을 추가하지 않았습니다. v4의 고정 composition plan, source image와 canonical label/mask를 hash로 pin하고, 기존 matte-black conveyor asset 위에서 geometry를 중립 replay한 뒤 deterministic 2D multi-source light, contact/directional shadow와 sensor transform을 적용합니다.

한 장면의 2–3개 광원 방향·상대 세기·CCT는 rendering proxy입니다. `P0`–`P5`와 `capture_plan_target_lux`는 향후 실물 촬영 조건을 연결하기 위한 계획값일 뿐 실측 photometry가 아닙니다. 정확한 source SHA-256 map, generator/config hash, runtime과 release payload hash는 `synthetic/v5_illumination/annotations/release.json`에 기록합니다.

실제 조명 촬영에는 [실물 조명 촬영 프로토콜](REAL_LIGHTING_CAPTURE_PROTOCOL.md)과 [`real_lighting_capture_template.csv`](../annotations/real_lighting_capture_template.csv)를 사용합니다. 현재 이 프로토콜로 repository에 추가된 실물 이미지는 0장입니다.

이 provenance 기록은 공개 재사용 허가가 아닙니다. v5 합성 장면, mask, annotation과 metadata의 이용 조건은 [LICENSE_STATUS.md](../LICENSE_STATUS.md)를 따릅니다.
