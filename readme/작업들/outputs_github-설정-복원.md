# outputs_github 설정 복원

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록](../overview.md)
- [이전 direction 비교](outputs_github-outputs-direction-비교.md)

## 요청과 결과

현재 설정을 `outputs_github`의 실행 설정에 맞추라는 요청에 따라 원래 의존성 설정과 잠금 파일을 복원하고 설치했다. 설치 결과 PyTorch 2.14.1+cu130, Transformers 4.57.6으로 참조 환경과 일치한다.

단, 현재 NVIDIA 드라이버 570.195.03은 CUDA 12.8 환경이다. CUDA 13 빌드로 복원한 뒤 preflight는 CUDA 사용 불가로 실패했다. 모델 설정·tokenizer 조회는 성공했다. **설정 복원 완료와 GPU 실행 가능 여부는 별개이며, 현재 환경에서 GPU 실험을 실행할 수 없다.**

## 변경 사항

| 대상 | 변경 전 → 변경 후 | 근거 |
|---|---|---|
| `pyproject.toml` | torch==2.7.1, Linux cu128 전용 인덱스, hf-transfer 추가 → Git HEAD의 원래 의존성 설정(torch>=2.6,<3, 기본 인덱스, hf-transfer 제외) | 참조 산출물의 잠금 파일과 동일한 의존성 구성 복원 |
| `uv.lock` | CUDA 12.8용 수정 잠금 파일 → Git HEAD의 원래 잠금 파일 | 원래 파일 SHA-256이 outputs_github manifest 기록과 정확히 일치 |
| 로컬 `.venv` | PyTorch 2.7.1+cu128 → 2.14.1+cu130 및 잠금 파일의 CUDA 13 의존성 | uv sync --frozen으로 실제 설치 |
| `README.md` | CUDA 12.8 설치 안내 → 복원한 CUDA 13 환경·현재 드라이버 제한·HF 다운로드 옵션 안내 | 현재 설정과 실행 상태 반영 |

`configs/experiment.toml`은 TOML 파싱 결과가 참조 `config.toml`과 이미 동일하므로 설정 값을 변경하지 않았다. 텍스트 주석 두 줄만 다르다. Qwen 모델 ID와 전체 revision SHA는 그대로 유지했다.

복원한 잠금 파일 SHA-256:

```text
c721bf9fa2acd09333fdb86fa5ba2501cf73ecc892e4b86773f05926a56aea9a
```

기존 `outputs/`, `outputs_github/`의 snapshot·direction·응답은 수정하지 않았다. 이전 작업 로그도 당시 수행 내역으로 보존한다.

## 참조 설정과 일치 범위

- 모델: Qwen/Qwen2.5-7B-Instruct
- revision: a09a35458c702b33eeacc393d103063234e8bc28
- dtype: auto (참조 GPU 실행에서는 bfloat16)
- device: cuda, 입력 한도 4096, do_sample=false, seed=42
- steering: layers=[-1], coefficient=1.0, normalize=false
- judge: CPU rule, gcg_refusal_v1
- 기본 config의 max_new_tokens: 512
- smoke 실행 시 덮어쓰는 설정: fold 1, limit_per_level=2, max_new_tokens=16

Python 3.11.13, PyTorch 2.14.1+cu130, Transformers 4.57.6은 참조 로그와 일치한다. GPU는 NVIDIA A40이지만 현재 CUDA 초기화가 실패한다. Linux kernel은 현재 6.8.0-49-generic, 참조는 6.8.0-134-generic이다. 참조 로그에는 NVIDIA 드라이버 버전이 없으므로 참조 드라이버와의 정확한 일치는 확인할 수 없다. 커널·드라이버는 이번 저장소 설정 복원에서 변경하지 않았다.

## 검증 결과

| 명령 또는 확인 | 결과 |
|---|---|
| TOML 값 비교 | 현재 configs/experiment.toml과 참조 config.toml 일치 |
| uv.lock SHA-256 비교 | 참조 manifest의 dependency_lock_sha256과 일치 |
| `uv lock --check` | 통과 |
| `uv sync --frozen --extra gpu --extra dev` | 설치 성공 |
| `./scripts/check` | 통과: HarmBench 300, severity 300, benign safe 250, 5-fold, 외부 harmful 46·benign 99 |
| `uv run --extra gpu --extra dev pytest` | 220 passed, 1 skipped (8.12초) |
| `uv run --extra gpu --extra dev ruff check .` | 통과 |
| `HF_HUB_ENABLE_HF_TRANSFER=0 ./scripts/preflight --model model_1` | 종료 코드 2: CUDA 드라이버 호환성 실패. Qwen config·tokenizer 조회 성공 |

preflight는 `torch=2.14.1+cu130`, `transformers=4.57.6`, `cuda_available=false`를 보고했고, 드라이버가 너무 오래됐다는 CUDA 초기화 경고를 출력했다. 실패를 숨기거나 CPU·더 작은 모델로 대체해 실행하지 않았다. 모델 가중치 로딩·생성 smoke·벡터 재추출은 실행하지 않았다.

## 재실행 방법

CUDA 13과 호환되는 드라이버가 있는 환경에서 실행한다. 이번 작업에서 드라이버 설치나 인스턴스 생성은 수행하지 않았다.

```sh
uv sync --frozen --extra gpu --extra dev
./scripts/check
uv run --extra gpu --extra dev pytest
uv run --extra gpu --extra dev ruff check .
HF_HUB_ENABLE_HF_TRANSFER=0 ./scripts/preflight --model model_1
```

복원한 잠금 파일에는 hf-transfer가 없어 기존 환경의 HF_HUB_ENABLE_HF_TRANSFER=1을 그대로 쓰면 다운로드 오류가 날 수 있다. 위 명령은 해당 프로세스에서만 표준 다운로드를 사용한다. 이후 smoke·추출·생성 명령에도 같은 환경변수를 적용한다.

preflight가 통과한 뒤 smoke를 재실행한다면 기존 snapshot과 섞이지 않도록 **새 run ID**를 사용한다.

```sh
HF_HUB_ENABLE_HF_TRANSFER=0 ./scripts/smoke --model model_1 --run-id smoke-model1-cu130-001
```

위 smoke 명령은 안내용이며 이번 작업에서 실행하지 않았다. 설정을 복원했다고 L3 direction의 원소가 정확히 일치한다고 보장할 수는 없다. 재추출 결과를 별도로 확인해야 한다.
