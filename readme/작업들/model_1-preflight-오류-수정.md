# model_1 preflight 오류 수정

- 날짜: 2026-10-13 (UTC)
- session id: `01a13446-d303-7f72-ab96-9719cc0287d3`
- [작업 목록으로 돌아가기](../overview.md)

## 요청

`./scripts/preflight --model model_1` 오류를 해결하고 다음 모델과 revision을 지정한다.

```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
```

모델 설정은 프로그램이 실제로 읽는 `configs/experiment.toml`에 반영했다. `pyproject.toml`은 설치 의존성과 PyTorch 다운로드 인덱스 설정을 수정했다.

## 발견한 오류와 원인

1. `model_1`의 `hf_id`와 `revision`이 비어 있어 `Set models.model_1.hf_id in configs/experiment.toml first` 오류가 발생했다.
2. 기존 잠금 파일로 설치된 PyTorch 2.14.1 CUDA 13 빌드는 현재 NVIDIA 드라이버 570.195.03(CUDA 12.8)과 맞지 않았다. CUDA 초기화 경고와 함께 `torch.cuda.is_available()`이 false를 반환했다.
3. 모델 설정 후 Hugging Face 설정 파일 로딩이 실패했다. 상세 다운로드 예외를 확인한 결과, 환경의 `HF_HUB_ENABLE_HF_TRANSFER=1`에 필요한 `hf_transfer` 모듈이 설치되지 않았다.

지정 revision은 40자리 전체 SHA이며, Hugging Face 모델 revision API와 해당 revision의 `config.json` 요청이 모두 HTTP 200을 반환하는 것을 확인했다.

## 변경 사항

| 파일 | 변경 전 → 변경 후 | 이유 |
|---|---|---|
| `configs/experiment.toml` | `model_1`의 빈 ID·revision → 요청한 Qwen ID·전체 SHA | 실제 모델 설정을 고정하고 미지정 모델 오류 해결 |
| `pyproject.toml` | `torch>=2.6,<3` → `torch==2.7.1`; Linux용 `pytorch-cu128` explicit 인덱스 지정 | 현재 CUDA 12.8 드라이버와 호환되는 GPU 빌드를 설치하도록 고정 |
| `pyproject.toml` | GPU extra에 `hf-transfer>=0.1.9,<0.2` 추가 | Runpod 환경에서 활성화된 HF 전송 옵션에 필요한 패키지 설치 |
| `uv.lock` | CUDA 13 의존성 → PyTorch 2.7.1+cu128 및 CUDA 12.8 의존성; hf-transfer 0.1.9 추가 | 수정한 의존성을 재현 가능한 잠금 파일에 반영 |
| `README.md` | 미지정 모델 예시·CUDA 13 안내 → 고정 Qwen 설정·CUDA 12.8 설치·HF 전송 패키지 안내 | 실행 안내를 실제 설정과 일치시킴 |
| `tests/test_cli.py` | 미지정 모델 테스트 대상을 model_1 → model_2로 변경; pin 결과·주석 기대값 수정 | model_1이 이미 지정된 기본 설정에 맞추면서 미지정 모델 및 pin 검증 유지 |
| `tests/test_prepare.py` | model_1이 빈 ID라는 기대값 → Qwen ID·revision 검증 및 model_2의 빈 ID 검증 | 변경한 기본 설정과 남은 placeholder를 확인 |

PyTorch CUDA 12.8 빌드는 [공식 PyTorch 설치 안내](https://pytorch.org/get-started/previous-versions/)에서 확인했다. preflight 스크립트와 모델 실행 코드 자체는 수정하지 않았다.

## 설치 및 검증 기록

의존성 변경 시 `uv lock`으로 잠금 파일을 갱신하고 다음 명령으로 설치했다.

```sh
uv sync --frozen --extra gpu --extra dev
```

| 명령 | 결과 |
|---|---|
| `./scripts/check` | 통과: HarmBench 300개, severity 라벨 300개, benign safe 250개, 5-fold OOF, 외부 harmful 46개·benign 99개 |
| `uv run --extra gpu --extra dev pytest` | 220 passed, 1 skipped (8.50초) |
| `uv run --extra dev ruff check .` | 통과 |
| `./scripts/preflight --model model_1` | 최종 실행 종료 코드 0, CUDA 사용 가능, Qwen 설정·tokenizer 로딩 통과 |
| `uv run --extra gpu --extra dev ruff check .` | hf-transfer 추가 후 최종 lint 통과 |
| `git diff --check` | 당시 변경 사항의 공백 검사 통과 |

전체 pytest는 CUDA 의존성 및 모델 설정 변경 후 실행했다. 이후 hf-transfer를 추가한 최종 상태에서는 설치·preflight·lint·diff 검사를 확인했으며, 전체 pytest를 다시 실행하지 않았다.

### 최종 preflight 환경

```json
{
  "torch": "2.7.1+cu128",
  "transformers": "4.57.6",
  "requested_device": "cuda",
  "cuda_available": true,
  "pending_models": ["model_2", "model_3"],
  "gpu": "NVIDIA A40",
  "free_vram_gib": 44.17,
  "total_vram_gib": 44.43,
  "bf16_supported": true,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
  "architecture": "qwen2",
  "context_window": 32768
}
```

VRAM 수치는 검증 당시 값이다. 데이터 검사 fingerprint는 `dff8a7883dfd3028c1eb48df8d8d58d555b5473a8ae6497e8d952f6ec779f67f`였다.

## 미검증 범위

- 실제 Qwen 모델 가중치 로딩과 GPU 추론
- 벡터 추출 및 생성 smoke
- 전체 OOF·외부·safe GPU 실험

preflight 성공과 CPU fixture 테스트 결과를 실제 GPU 실험 성공으로 해석하지 않는다.

## 재실행 명령

저장소 루트에서 실행한다.

```sh
uv sync --frozen --extra gpu --extra dev
./scripts/check
uv run --extra gpu --extra dev pytest
uv run -

-extra gpu --extra dev ruff check .
./scripts/preflight --model model_1
```

## 작업 로그 문서 추가

이 세션의 후속 요청에 따라 `readme/overview.md`에 작업내용·날짜·session id 표를 만들고 이 상세 로그로 연결했다. `readme/log.md` 대신 작업별 문서 구조를 사용한다. 상세 파일명의 공백은 `-`로 표기한다.
