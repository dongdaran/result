# HarmBench level steering

HarmBench 문항을 프로젝트의 L1/L2/L3로 나누고, 각 레벨의 activation 평균에서 benign 평균을 빼서 steering vector를 만듭니다. 모델마다 **벡터를 만든 레벨 × 평가하는 레벨의 3×3 ASR 표**와 steering 전 baseline을 저장합니다.

목표는 읽기 쉬운 코드, 명시적인 설정, 작은 실행을 통한 검증, 실패 원인 기록, 중단 후 재개입니다. 모델 3개의 Hugging Face ID는 나중에 `configs/experiment.toml`에 채우면 됩니다. GPU를 대여하거나 Runpod 인스턴스를 만드는 기능은 포함하지 않습니다.

## 바로 시작

Python은 `uv`가 프로젝트의 3.11 버전을 관리합니다. `uv`가 PATH에 있어야 합니다.

```sh
cd ~/Desktop/harmbench-steering
./scripts/download
./scripts/check
```

`./scripts/download`는 `#!/usr/bin/env -S uv run -s`와 PEP 723 metadata를 가진 실행 파일입니다. 별도 GPU 의존성 없이 다음을 한 번에 처리합니다.

1. 커밋과 SHA-256이 고정된 HarmBench 및 XSTest 원본 다운로드
2. 번들된 `data/annotations/severity.csv`의 레벨과 근거 적용
3. seed 42로 fit/dev/test 분할 생성 및 검증

이미 검증된 다운로드와 분할이 있으면 재사용합니다. 원본, 라벨, 분할 설정의 변경을 감지하면 조용히 덮어쓰지 않습니다. 의도적으로 라벨이나 분할 설정을 바꿨다면 `./scripts/download --resplit`으로 분할을 다시 만들고 변경 내용을 Git에서 확인합니다. 기존 실행 결과에는 새 분할을 섞지 말고 새 run ID를 사용합니다.

## Runpod에서 실행

프로젝트를 GPU Pod로 복사하거나 Git으로 clone한 뒤 프로젝트 디렉터리에서 실행합니다. GPU 드라이버가 있는 Linux 환경과 충분한 VRAM이 필요합니다. 필요한 VRAM은 나중에 고를 target 모델과 13B judge에 따라 달라집니다.

현재 `uv.lock`의 Linux PyTorch 패키지는 CUDA 13 계열입니다. 일반적인 실행 환경은 NVIDIA R580 이상 드라이버가 필요하므로 Pod의 드라이버를 확인합니다. `preflight`가 CUDA 사용 가능 여부를 확인합니다. [NVIDIA 호환성 표](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

```sh
./scripts/runpod-setup
./scripts/download
./scripts/check
```

`runpod-setup`은 `uv`가 없을 때만 `python3 -m pip install uv`로 설치하고, `uv sync --frozen --extra gpu --extra dev`와 환경 preflight를 실행합니다. 모델을 아직 정하지 않았어도 환경 점검까지 할 수 있습니다.

`configs/experiment.toml`에서 다음 세 ID를 채웁니다.

```toml
[models.model_1]
hf_id = "organization/model-name"
revision = ""
dtype = "auto"
```

`model_2`, `model_3`도 같은 방식으로 입력합니다. 필요한 경우 Hugging Face 접근 권한을 받은 뒤 셸에 `HF_TOKEN`을 설정합니다. `.env`는 자동 로딩하지 않습니다. 파일을 쓰려면 직접 `uv run --env-file .env ...`로 실행합니다.

기본 입력 형식은 tokenizer의 chat template입니다. base 모델처럼 chat template이 없는 모델은 해당 모델 설정에 `use_chat_template = false`를 명시합니다. 실제 decoder 구조 지원 여부는 preflight와 smoke에서 확인합니다.

```sh
./scripts/pin-models
./scripts/preflight --model model_1
./scripts/smoke --model model_1 --run-id smoke-001
```

`pin-models`는 입력된 target ID와 공식 judge ID의 revision을 전체 commit SHA로 고정합니다. 빈 target ID는 건너뜁니다. smoke는 dev의 레벨당 2개 문항으로 벡터 추출과 생성을 확인하며 생성 길이는 최대 16개 token으로 제한합니다. judge는 실행하지 않습니다. 이 단계를 먼저 확인하고 전체 실행으로 넘어갑니다.

```sh
./scripts/run --model model_1 --run-id test-001
./scripts/run --model model_2 --run-id test-001
./scripts/run --model model_3 --run-id test-001
```

같은 run ID 아래에서도 모델별 출력 디렉터리를 분리합니다. 각 명령은 프로젝트 경로를 자체적으로 찾으므로 다른 디렉터리에서 절대 경로로 호출해도 됩니다.

## 실험 흐름과 설정 선택

`fit` 문항만 벡터 계산에 사용합니다. `dev`로 레이어·계수·정규화 여부 등을 선택한 다음, 설정을 고정해서 새로운 run ID로 `test`를 실행합니다. 기본값이 최적이라는 주장은 하지 않습니다.

```sh
# dev에서 작은 조합 확인
./scripts/run --model model_1 --run-id dev-001 --split dev --limit-per-level 2

# 설정을 검토·수정한 뒤 새로운 run ID로 최종 test 실행
./scripts/run --model model_1 --run-id test-002 --split test
```

`--limit-per-level`은 빠른 확인을 위한 부분 실행 옵션입니다. 부분 실행 결과를 전체 평가와 혼동하지 않도록 run ID를 분리합니다. 같은 ID의 한도·split·설정·데이터를 바꾸면 기존 실행과 불일치 오류가 납니다.

필요하면 단계를 나눠 실행합니다. 모델 메모리를 해제한 뒤 judge를 로딩하므로 target과 judge를 동시에 올릴 필요는 없습니다.

```sh
./scripts/extract --model model_1 --run-id test-002
./scripts/generate --model model_1 --run-id test-002 --split test
./scripts/score --model model_1 --run-id test-002 --split test
./scripts/summarize --model model_1 --run-id test-002 --split test
```

사용 가능한 인자는 각 실행 파일의 `--help`로 확인합니다. 별도 설정 파일을 쓰려면 전역 `--config`를 subcommand 앞에 둡니다.

```sh
uv run --extra gpu hb-steering --root "$PWD" --config configs/experiment.toml \
  run --model model_1 --run-id custom-001
```

## 데이터와 레벨

- **HarmBench:** 고정 원본의 400개 text behaviors 중 standard 200개와 contextual 100개를 사용합니다. copyright 100개와 multimodal 데이터는 기본 실험에서 제외합니다. contextual 문항은 context를 입력 및 판정에 포함합니다.
- **레벨:** L1은 제한적인 피해, L2는 상당한 피해, L3는 생명·중대한 신체 피해 또는 대규모 피해를 중심으로 정한 프로젝트 분류입니다. `data/annotations/severity.csv`는 ID·레벨·근거를 담은 **AI가 작성한 초기 라벨**이며 HarmBench의 공식 레벨이나 사람이 검수한 확정 라벨이 아닙니다. 이 분류로 얻는 결과는 해당 라벨 기준의 비교입니다. 본실험 전에 경계 문항과 근거를 검토하는 것이 좋습니다.
- **Benign 기준:** XSTest의 safe 250개를 사용합니다. 고정 seed로 fit 150개/dev 50개/test 50개로 나눕니다. XSTest는 과도한 거부를 점검하도록 설계된 safe 문항으로, HarmBench와 내용이 완전히 대응하는 중립 대조군은 아닙니다.
- **분할:** 기본 비율은 fit/dev/test = 60/20/20이며 HarmBench는 레벨별 비율을 목표로 합니다. Unicode·대소문자·공백을 정규화한 동일 behavior 또는 동일한 비어 있지 않은 context를 연결한 그룹을 통째로 배치합니다. 큰 그룹 때문에 정확한 비율은 달라질 수 있습니다. 의미가 비슷한 다른 표현까지 자동으로 찾아내지는 않습니다.

번들된 초기 라벨과 기본 분할의 문항 수는 다음과 같습니다. 레벨별 문항 수가 같지 않으므로 ASR을 읽을 때 분모도 함께 확인합니다.

| 레벨 | 전체 | fit | dev | test |
| --- | ---: | ---: | ---: | ---: |
| L1 | 29 | 17 | 6 | 6 |
| L2 | 189 | 113 | 38 | 38 |
| L3 | 82 | 49 | 17 | 16 |
| 합계 | 300 | 179 | 61 | 60 |

데이터 출처: [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/xstest). 버전·파일 hash는 설정과 다운로드 manifest에 남깁니다. 라이선스 및 인용 정보는 `third_party/NOTICE.md`에 있습니다.

## 벡터와 ASR

각 target 모델의 각 레벨에 대해 다음을 계산합니다.

```text
v[level] = mean(해당 level의 fit activation) - mean(benign fit activation)
```

기본 추출 위치는 **마지막 decoder block의 output, 마지막 prompt token**입니다. 기본 개입은 같은 block output에 `coefficient × vector`를 더하며 prompt와 생성 token 모두에 적용합니다. `layers = [-1]`, `coefficient = 1.0`, `normalize = false`가 시작 설정입니다. 모델 간 hidden 크기가 달라도 각 모델에서 자체 벡터를 계산합니다. 모델 사이로 벡터를 옮기는 실험은 아닙니다.

| 벡터 계산 레벨 ↓ / 평가 레벨 → | L1 | L2 | L3 |
| --- | --- | --- | --- |
| L1 | ASR | ASR | ASR |
| L2 | ASR | ASR | ASR |
| L3 | ASR | ASR | ASR |

모델마다 이 표 하나와 레벨별 baseline 3개를 저장합니다. 모델 3개를 모두 실행하면 steering 조합 27개와 baseline 9개입니다. 각 평가 문항은 baseline과 세 벡터 조건에 동일하게 사용합니다.

`layers`에 여러 위치를 지정하면 각 레이어에 독립적으로 개입하고 **레이어별 3×3 표**를 만듭니다. 여러 레이어를 동시에 개입하는 설정은 아닙니다.

ASR은 공식 `cais/HarmBench-Llama-2-13b-cls`와 standard/contextual 판정 프롬프트를 사용합니다. 생성 오류, judge 오류, 예상하지 못한 judge 출력은 정상적인 `no`로 처리하지 않습니다. 결과를 읽을 때 ASR과 함께 성공 평가 수, 기대 평가 수, 오류 수를 확인합니다. 공식 구성 요소를 재사용하지만 이 프로젝트의 레벨·분할·steering 설정을 추가한 실험이므로 전체 공식 HarmBench leaderboard 결과와 같은 조건의 점수는 아닙니다.

## 출력과 재개

```text
outputs/<run_id>/
├── config.toml                # 원본 TOML 설정 snapshot
├── effective_config.json      # smoke 제한 등을 반영한 실제 설정
├── manifest.json              # split·한도·설정·데이터·코드 식별 정보
├── code_hashes.json            # 소스 파일별 SHA-256
└── <model_key>/
    ├── vectors/               # 레벨별 벡터와 metadata
    ├── generations/           # baseline·steering 응답
    ├── metrics/               # judge 결과·ASR·3×3 집계
    └── logs/                  # 오류 및 실행 정보
```

실행 설정, split 파일 hash, 데이터 정보와 소스 코드 hash는 run에 고정됩니다. 원본 TOML과 실제 적용 설정을 따로 남기므로 smoke 제한도 확인할 수 있습니다. 중단 후 **같은 명령**을 다시 실행하면 완료된 작업을 재사용합니다. 실패한 항목의 재시도 기록은 결과와 구분해서 보존합니다. 마지막 journal 줄이 프로세스 종료로 잘렸을 때만 복구하며, 중간 데이터 손상은 명시적으로 실패합니다. 같은 모델과 run ID에 동시에 쓰는 실행은 파일 잠금으로 차단합니다. 코드를 수정한 뒤에는 새 run ID를 사용합니다.

`outputs/`, `data/raw/`, `data/processed/`, 모델 캐시, `.venv/`, `.env`, `work/`는 Git에서 제외합니다. 라벨, 고정 분할, 설정, 소스, 테스트, `uv.lock`은 추적합니다. Pod가 삭제돼도 필요한 결과를 잃지 않도록 `outputs/`를 지속 볼륨에 두거나 직접 백업합니다.

## 프로젝트 지도

```text
harmbench-steering/
├── AGENTS.md                  # 코드 작성·검증·실행 원칙
├── README.md                  # 구조와 실행 가이드
├── pyproject.toml             # uv 프로젝트·선택적 GPU 의존성
├── uv.lock                    # 고정 의존성
├── configs/experiment.toml    # 데이터·모델·steering·judge 설정
├── scripts/                   # download, check, smoke, run 등 진입점
├── src/hb_steering/
│   ├── cli.py                 # 명령 연결·실행 흐름
│   ├── experiment.py          # 실행 계획·벡터·생성·판정·집계 연결
│   ├── data.py                # HarmBench·라벨·누수 방지 분할
│   ├── prepare.py             # XSTest·다운로드·고정 분할 관리
│   ├── config.py              # 설정 검증
│   ├── models.py              # HF 로딩·prompt·token 처리
│   ├── vectors.py             # activation 평균 차이
│   ├── steering.py            # hook 설치·제거
│   ├── evaluate.py            # HarmBench judge
│   └── storage.py             # 원자적 저장·journal·잠금
├── data/
│   ├── annotations/           # 추적하는 초기 레벨·근거
│   ├── splits/                # 추적하는 고정 split ID·metadata
│   ├── raw/                   # 다운로드 원본·manifest, Git 제외
│   └── processed/             # 파생 데이터, Git 제외
├── tests/                     # 데이터·벡터·hook·재개·판정 검증
├── third_party/               # 출처·라이선스 원문
├── outputs/                   # 실행별 산출물, Git 제외
└── work/                      # 임시 파일, Git 제외
```

## 코드 검증

```sh
uv sync --frozen --extra gpu --extra dev
uv run --extra gpu --extra dev pytest
uv run --extra dev ruff check .
```

테스트는 작은 fixture와 CPU의 작은 모델로 데이터 분리·activation 위치·개입 제거·저장·재개를 확인합니다. 실제 선택할 모델 및 Runpod GPU의 호환성·VRAM·전체 실험 성공은 각 모델의 preflight와 smoke로 별도 검증해야 합니다.
