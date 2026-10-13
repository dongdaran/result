# 모델 설정 pyproject 이전, uv.lock 삭제 및 SHA 검증

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)

## 요청과 구현

사용자 요청에 따라 모델 ID와 전체 revision SHA를 pyproject.toml에 옮기고 uv.lock을 삭제했다. 이후 전체 테스트와 현재 outputs / outputs_github 벡터 SHA·원소 일치를 확인했다.

```toml
[tool.hb-steering.models.model_1]
hf_id = "Qwen/Qwen2.5-7B-Instruct"
revision = "a09a35458c702b33eeacc393d103063234e8bc28"
dtype = "auto"
```

- configs/experiment.toml의 모델 테이블 제거. 데이터·runtime·steering·judge 설정은 유지.
- load_config가 저장소 root의 pyproject.toml 모델 테이블을 읽어 유효 설정에 합침. 구 config에 모델 테이블이 남아 있으면 중복 소스를 조용히 선택하지 않고 이전 안내 오류를 냄.
- pin-models가 target revision을 pyproject.toml에 저장. 선택적 classifier judge 설정은 실험 config에 유지. 전체 revision SHA 검증 유지.
- uv.lock 삭제 및 Git 제외. uv sync는 로컬 lock을 자동 생성할 수 있으나 추적하지 않음. 일반 GPU 스크립트는 uv run --no-sync로 실행해 자체적으로 lock을 재생성하거나 의존성을 변경하지 않음.
- torch==2.14.1, transformers==4.57.6을 명시. 다른 transitive dependency는 lock 없이 설치하면 달라질 수 있음.
- 새 run은 pyproject hash·snapshot을 보존. lock이 없으면 dependency_lock_sha256은 null이고 실제 설치 패키지 버전을 manifest에 보존. 같은 run ID에서 pyproject·패키지 버전이 달라지면 재개 거부.
- README, 팀 가이드, AGENTS.md를 최신 사용자 요청에 맞게 갱신. 기존 산출물 snapshot·manifest·벡터는 수정하지 않음.

## 테스트

| 확인 | 결과 |
|---|---|
| ./scripts/check | 통과 |
| helper source 후 uv run --no-sync --extra gpu --extra dev pytest | 223 passed, 1 skipped (6.10초) |
| uv run --no-sync --extra gpu --extra dev ruff check . | 통과 |
| ./scripts/preflight --model model_1 | 종료 코드 0, CUDA true, A40, PyTorch 2.14.1+cu130, Transformers 4.57.6 |
| ./scripts/pin-models | model_1 already pinned, model_2·3 pending |
| uv.lock 존재 여부 | false |

추가 테스트는 pyproject에서 모델을 읽는지, 중복/누락 모델 설정이 실패하는지, lock 없는 run의 pyproject snapshot과 설치 버전 기록 및 변경 후 재개 거부를 검증한다. 이 작업에서 Qwen 생성 smoke·벡터 재추출은 수행하지 않았다.

## 현재 벡터 SHA 확인

| 항목 | outputs_github | outputs | 일치 |
|---|---|---|---|
| directions.pt 파일 SHA-256 | 69fe95cc0b706e80c34d8794f3bf650838aeed8c163bdcb364747808615ede90 | 17e9bcddff0be9ce2eee56fb897a6d889997ff46dde2cfef7d374447f5567dd1 | 아니오 |
| 정렬한 벡터 값 전용 SHA-256 | 09145f90c44fa18da0bd9d9f0671a0e984340893529689273914c013f1cc5b43 | 09145f90c44fa18da0bd9d9f0671a0e984340893529689273914c013f1cc5b43 | 예 |
| 기존 manifest dependency_lock_sha256 | c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a | c68d0a6a6a182181390e53fdd59869e164530f197220cbc9ea99c8897b68605b | 아니오 |

L1·L2·L3 모두 torch.equal=True. 각 3584개 원소 전부 동일하며 차이 norm은 0이다. 벡터 외 payload 메타데이터의 다른 필드는 `['run_fingerprint']`다. 전체 .pt 파일 SHA에는 run fingerprint와 직렬화 정보가 포함되므로, 텐서 값이 같아도 파일 SHA는 다를 수 있다.

벡터 값 전용 hash는 level·layer를 정렬한 뒤 각 level/layer/shape/dtype의 UTF-8 헤더와 little-endian float32 contiguous tensor bytes를 순서대로 SHA-256에 넣어 계산했다. 이는 이번 비교용 정의이며 기존 directions.json checksum을 대체하지 않는다. 양쪽 원본 파일 checksum은 각각 sidecar와 일치함을 앞선 검증에서 확인했다.

현재 uv.lock은 요청대로 삭제되어 hash 자체가 없다. 과거 manifest의 dependency_lock_sha256은 당시 provenance이므로 값을 조작해 맞추지 않았다. 새 run ID로 실행해야 변경된 설정과 기존 산출물을 혼합하지 않는다.

## 실행 방법

```sh
uv sync --extra gpu --extra dev
./scripts/preflight --model model_1
```

uv sync가 생성하는 로컬 uv.lock은 Git에서 제외된다. 위 설정 이전 검증에서는 --no-sync를 사용해 현재 uv.lock이 없는 상태를 유지했다. 기존 환경에 설치된 패키지로 테스트했으며, lock 삭제 뒤 완전히 새 환경에서 transitive dependencies를 재설치하는 검증은 수행하지 않았다.

## 모델 checkpoint commit SHA 검증 (사용자 요청 정정)

요청한 SHA는 벡터 또는 의존성 잠금 파일의 SHA가 아니라 **모델 checkpoint revision**이다. 모델 ID는 `Qwen/Qwen2.5-7B-Instruct`이다.

| 출처 | 모델 revision SHA |
|---|---|
| pyproject.toml | a09a35458c702b33eeacc393d103063234e8bc28 |
| outputs_github manifest | a09a35458c702b33eeacc393d103063234e8bc28 |
| outputs_github target_environment | a09a35458c702b33eeacc393d103063234e8bc28 |
| outputs manifest | a09a35458c702b33eeacc393d103063234e8bc28 |
| outputs target_environment | a09a35458c702b33eeacc393d103063234e8bc28 |
| Hugging Face resolved commit | a09a35458c702b33eeacc393d103063234e8bc28 |

모든 모델 revision SHA가 정확히 일치한다. Hugging Face의 해당 revision API가 반환한 sha도 일치한다. main/latest로 변경하지 않았다. 이것은 고정된 repository commit의 일치 검증이며, 모델 가중치 shard 파일을 모두 다시 다운로드해 hash를 계산한 검증은 아니다.
