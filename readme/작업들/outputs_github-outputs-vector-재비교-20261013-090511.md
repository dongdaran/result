# outputs_github / outputs vector 재비교

- 비교 시각: 2026-10-13T09:05:11.847443+00:00
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)
- [이전 비교 기록](outputs_github-outputs-direction-비교.md)

현재 디스크의 벡터를 읽어 비교했다. 원본 산출물은 수정하지 않았으며 새 벡터 추출은 수행하지 않았다.

- outputs_github: directions.pt 1개, 상대 경로가 대응되지 않는 파일 []

- outputs: directions.pt 1개, 상대 경로가 대응되지 않는 파일 []

## smoke-model1-001/model_1/fold_1/vectors/directions.pt

[A: outputs_github](../../outputs_github/smoke-model1-001/model_1/fold_1/vectors/directions.pt) · [B: outputs](../../outputs/smoke-model1-001/model_1/fold_1/vectors/directions.pt)

원소 일치는 저장 float32 값으로 판정한다. norm·cosine·오차는 CPU float64로 계산한다. 상대 차이는 `||B-A||₂ / ||A||₂ × 100`이다.

| 레벨 | 레이어 | 정확 일치 | 다른 원소 수 | A norm | B norm | cosine | 차이 L2 norm | 상대 차이 | 최대 절대 차이 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 27 | True | 0/3584 | 341.860294517 | 341.860294517 | 1.000000000000 | 0.000000000 | 0.000000% | 0.000000000 | 0.000000000 |
| L2 | 27 | True | 0/3584 | 392.903178622 | 392.903178622 | 1.000000000000 | 0.000000000 | 0.000000% | 0.000000000 | 0.000000000 |
| L3 | 27 | True | 0/3584 | 351.281216123 | 351.281216123 | 1.000000000000 | 0.000000000 | 0.000000% | 0.000000000 | 0.000000000 |

### 무결성 및 provenance

- `outputs_github/smoke-model1-001/model_1/fold_1/vectors/directions.pt` SHA-256: `69fe95cc0b706e80c34d8794f3bf650838aeed8c163bdcb364747808615ede90`
- direction run fingerprint: `860abdf14143990199eb9941b98dc65ba10c029178ba83fbd20b5cc529f223f3`
- `outputs/smoke-model1-001/model_1/fold_1/vectors/directions.pt` SHA-256: `17e9bcddff0be9ce2eee56fb897a6d889997ff46dde2cfef7d374447f5567dd1`
- direction run fingerprint: `4658bbf24090b9877345ed7cf127bde8d24c0d3e69bec9777a504c23a93849b7`

각 checksum·fingerprint는 해당 sidecar와 일치한다. 양쪽 schema·shape·float32 CPU dtype·유한값 검증 통과.

| 생성 환경 로그 항목 | outputs_github | outputs |
|---|---|---|
| torch | 2.14.1+cu130 | 2.14.1+cu130 |
| cuda | 13.0 | 13.0 |
| transformers | 4.57.6 | 4.57.6 |
| python | 3.11.13 | 3.11.13 |
| gpu | NVIDIA A40 | NVIDIA A40 |
| dtype | bfloat16 | bfloat16 |
| platform | Linux-6.8.0-134-generic-x86_64-with-glibc2.39 | Linux-6.8.0-49-generic-x86_64-with-glibc2.39 |

manifest의 차이 필드: `['dependency_lock_sha256']`.
- outputs_github dependency_lock_sha256: `c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a`
- outputs dependency_lock_sha256: `c68d0a6a6a182181390e53fdd59869e164530f197220cbc9ea99c8897b68605b`

벡터 외 payload 메타데이터의 차이 필드: `['run_fingerprint']`.

공통 설정은 Qwen/Qwen2.5-7B-Instruct, revision a09a35458c702b33eeacc393d103063234e8bc28, harmful_minus_benign, decoder_block_output:last_nonpadding_prompt_token, normalize=false다. 저장 벡터는 각 (3584,)이며 요청 layer -1은 저장 layer 27이다.

추출 표본 수: {"benign": 6, "levels": {"1": 2, "2": 2, "3": 2}}

## 결론과 이전 기록과의 관계

이번에 비교한 모든 벡터의 정확 일치 여부: **True**. L1·L2·L3 모두 다른 원소 수 0, 차이 L2 norm 0, 최대 절대 차이 0이다. cosine은 반올림해 1이다. 저장 텐서 값의 정확 일치와 파일 전체의 바이트 일치는 다른 개념이며 메타데이터·직렬화 때문에 파일 hash는 다를 수 있다.

이전 비교에서 L3 상대 차이는 0.585764%였지만, 이번 현재 파일에서는 0%다. 이전 보고서에 기록된 outputs 파일 SHA-256은 `a9157bff93196a598d9f2b95bd39601c613cb16f96553219edb8243831d5c23e`였다. 현재 파일 hash와 생성 환경은 위에 기록했다. 이 비교 작업에서는 파일을 재추출하거나 수정하지 않았으므로 외부에서 이루어진 파일 갱신 방법·시각·원인을 확정하지 않는다.

fold 1은 severity L1과 다르다. 해당 smoke는 L1·L2·L3 fit 문항을 레벨별 2개, benign fit 6개 사용한 작은 산출물이다. 전체 5-fold OOF 결과가 아니다.

현재 비교 프로세스의 PyTorch는 2.14.1+cu130, CUDA runtime은 13.0다. 수치 비교는 CPU에서 수행했다. 파일의 생성 환경은 위 target_environment.json을 따른다. 벡터가 같다는 결과만으로 응답·rule 판정·ASR·전체 OOF 성능이 같다고 주장하지 않는다.

## 검증 방법

`uv run --extra gpu python`에서 `hb_steering.vectors.load_vectors`로 CPU weights_only 로딩 후 schema·shape·dtype·유한값을 확인했다. torch.equal 및 float64 cosine·norm·최대 절대 차이·RMSE를 계산했다. SHA-256과 sidecar fingerprint를 검증하고 문서 링크 및 git diff 공백 검사도 확인했다. 문서만 추가했으며 전체 pytest는 재실행하지 않았다.
