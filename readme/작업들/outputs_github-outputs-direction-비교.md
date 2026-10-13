# outputs_github와 outputs direction 비교

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)

## 결론

L1·L2 direction은 모든 저장 원소가 정확히 같다. L3는 완전히 같지는 않지만 cosine similarity는 0.9999828544이고 차이의 L2 norm은 outputs_github 벡터 norm의 약 0.585764%다. 이 수치만으로 생성 응답이나 ASR이 같다고 판단할 수는 없다.

모델·설정·데이터·split·코드 hash는 같고 PyTorch/CUDA 버전과 dependency lock hash는 다르다. 라이브러리·커널의 수치 연산 차이는 가능한 설명이지만, 통제된 재추출 실험을 하지 않았으므로 L3 차이의 원인을 확정하지 않는다.

## 비교 대상과 smoke 범위

두 디렉터리에서 찾은 direction 파일은 각각 한 개다.

- [outputs_github directions.pt](../../outputs_github/smoke-model1-001/model_1/fold_1/vectors/directions.pt)
- [outputs directions.pt](../../outputs/smoke-model1-001/model_1/fold_1/vectors/directions.pt)

대상은 `smoke-model1-001/model_1/fold_1`이다. **fold 1은 severity L1과 다르다.** 현재 smoke는 기본적으로 fold 1만 선택하고 `limit_per_level=2`, `max_new_tokens=16`을 사용한다. 추출 코드는 L1·L2·L3 각각의 fit 문항으로 세 direction을 만든다.

두 payload 모두 benign fit 6개, harmful fit L1 2개·L2 2개·L3 2개를 기록한다. 이번 비교는 fold 1의 세 severity direction 비교이며, 전체 5-fold OOF 결과가 아닌 작은 smoke 산출물이다.

## 공통 설정

| 항목 | 두 파일의 값 |
|---|---|
| 모델 | Qwen/Qwen2.5-7B-Instruct |
| revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| 방향 | harmful_minus_benign |
| activation 위치 | decoder_block_output:last_nonpadding_prompt_token |
| 요청 레이어 / 저장 레이어 | -1 / 27 (0부터 시작, 마지막 decoder block) |
| shape / 저장 dtype / 저장 device | 각 벡터 (3584,) / float32 / CPU |
| 모델 연산 dtype | bfloat16 (환경 로그 기준) |
| 정규화 / 개입 계수 | false / 1.0 |

## 수치 비교

A = outputs_github, B = outputs. 차이는 B - A, 상대 차이는 `||B-A||₂ / ||A||₂`다. 저장 float32 원소를 정확 비교하고, norm·cosine·오차는 CPU float64로 계산했다.

| 레벨 | 레이어 | 정확 일치 | 다른 원소 수 | A L2 norm | B L2 norm | Cosine similarity | 차이 L2 norm | 상대 차이 | 최대 절대 차이 | RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 27 | True | 0/3584 | 341.860294517 | 341.860294517 | 1.000000000000 | 0.000000000 | 0.000000% | 0.000000000 | 0.000000000 |
| L2 | 27 | True | 0/3584 | 392.903178622 | 392.903178622 | 1.000000000000 | 0.000000000 | 0.000000% | 0.000000000 | 0.000000000 |
| L3 | 27 | False | 2968/3584 | 351.281216123 | 351.224283112 | 0.999982854405 | 2.057677514 | 0.585764% | 0.250000000 | 0.034371090 |

L1·L2 cosine의 1 근처 반올림 오차는 계산 정밀도 때문이다. 원소는 정확히 일치하고 차이 norm은 0이다. L3 최대 절대 차이 0.25는 정규화하지 않은 activation 차이 벡터의 단위다.

## provenance와 무결성

| 항목 | outputs_github | outputs |
|---|---|---|
| PyTorch | 2.14.1+cu130 | 2.7.1+cu128 |
| CUDA runtime | 13.0 | 12.8 |
| GPU | NVIDIA A40 | NVIDIA A40 |
| Python / Transformers | 3.11.13 / 4.57.6 | 3.11.13 / 4.57.6 |
| Linux kernel | 6.8.0-134-generic | 6.8.0-49-generic |
| dependency_lock_sha256 | c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a | 1f720f77541b9f50f262acaef2ebe8177060bf0ca733350e0c3a5ca4465980fd |
| run_fingerprint (direction) | 860abdf14143990199eb9941b98dc65ba10c029178ba83fbd20b5cc529f223f3 | 41236f710eaebbe14e9be3e32b826f4a0b7589c2a1af4a3a1182c719b4a82c77 |
| directions.pt SHA-256 | 69fe95cc0b706e80c34d8794f3bf650838aeed8c163bdcb364747808615ede90 | a9157bff93196a598d9f2b95bd39601c613cb16f96553219edb8243831d5c23e |

- `effective_config.json`과 `code_hashes.json`은 바이트 단위로 동일하다.
- `manifest.json`의 JSON 값 비교에서 다른 최상위 필드는 `dependency_lock_sha256` 하나다. 데이터 fingerprint·split hash·코드 hash·모델·limit·평가 설정은 같다.
- `config.toml`의 텍스트 차이는 모델 설정 앞의 설명 주석 두 줄뿐이다.
- 실제 `directions.pt` SHA-256은 각 `directions.json`의 기록과 일치한다. payload와 sidecar의 run fingerprint도 각각 일치한다.
- payload의 vectors와 run_fingerprint를 제외한 메타데이터는 같다.

같은 run 이름이어도 fingerprint와 의존성 provenance가 다르므로 산출물을 같은 run으로 섞거나 덮어쓰지 않는다. 파일 hash는 메타데이터와 직렬화에도 영향을 받으므로 실제 텐서 값을 별도로 비교했다.

근거는 각 디렉터리의 `manifest.json`, `effective_config.json`, `code_hashes.json`, `config.toml`, `model_1/fold_1/vectors/directions.json`, `directions.pt`, `model_1/fold_1/logs/target_environment.json`이다. smoke 범위는 `src/hb_steering/cli.py`의 smoke 설정 및 `src/hb_steering/experiment.py`의 fit 선택·추출 코드에서도 확인했다.

## 재현 명령

저장소 루트에서 실행한다. `load_vectors`는 CPU `weights_only=True` 로딩 후 schema·shape·dtype·유한값을 검증한다.

```sh
uv run --extra gpu python - <<'PY'
import torch
from hb_steering.vectors import load_vectors

suffix = 'smoke-model1-001/model_1/fold_1/vectors/directions.pt'
a = load_vectors('outputs_github/' + suffix)
b = load_vectors('outputs/' + suffix)
for level in sorted(a['vectors']):
    for layer, x in a['vectors'][level].items():
        y = b['vectors'][level][layer]
        exact = torch.equal(x, y)
        x, y = x.double(), y.double()
        delta = y - x
        print(level, layer, {
            'exact_equal': exact,
            'cosine': torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
            'difference_l2': delta.norm().item(),
            'relative_difference': (delta.norm() / x.norm()).item(),
            'max_abs_difference': delta.abs().max().item(),
        })
PY
```

## 검증 및 범위

양쪽 벡터 schema·shape·dtype·유한값, sidecar checksum·fingerprint, 공통 메타데이터와 설정·코드 hash를 검증했다. 기존 산출물은 수정하지 않았고 GPU 재추출은 실행하지 않았다. 응답 내용·rule 판정·ASR·전체 OOF 성능은 이번 비교 범위에 포함하지 않았다. 문서만 추가하므로 전체 테스트는 재실행하지 않았다.
