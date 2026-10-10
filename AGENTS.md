# 작업 목표

목표는 정확하고 유지보수하기 쉬운 실험 코드를 작성하고, Runpod GPU에서 데이터 준비부터 결과 집계까지 안정적으로 실행하는 것이다. 설정을 명시하고 작은 실행으로 검증하며, 실패 원인을 기록하고 재개할 수 있게 한다.

이 파일은 프로젝트 전체에 적용한다. 사용자 지시와 실제 코드·테스트를 우선하며, README와 실행 명령을 구현에 맞게 함께 유지한다.

## 코드와 실행 원칙

- `uv`, `pyproject.toml`, `uv.lock`을 사용한다. Python 지원 범위를 지키고 잠금 파일을 임의로 무시하지 않는다. 새 의존성은 필요한 이유가 있을 때만 추가한다.
- 명령은 작은 단위로 나눈다. 데이터 준비, 벡터 추출, 생성, 판정, 집계를 각각 재실행할 수 있어야 한다. 스크립트는 호출한 작업 디렉터리에 의존하지 않는다.
- 모델 ID를 추정해서 채우지 않는다. 실제 GPU 실험은 target의 전체 revision SHA가 명시돼야 한다. 기본 rule scorer는 CPU에서 실행하며 별도 모델을 로딩하지 않는다. 선택적으로 classifier를 쓰는 이전 설정은 judge의 전체 revision SHA도 명시한다. 원격 모델 코드를 자동으로 신뢰하거나 실행하지 않는다.
- 입력 형식, 모델 구조, token 길이, 레벨 coverage, 벡터 shape, dtype, 유한한 값, CUDA 환경을 필요한 위치에서 검증한다. OOM 또는 잘림을 숨기거나 임의의 축소 설정으로 바꿔 계속하지 않는다.
- 추출 activation과 개입 위치를 일치시킨다. decoder block output, token 위치, 벡터 방향, 정규화, 계수의 의미를 명시한다. hook은 예외 발생 시에도 반드시 제거한다.
- 설정과 데이터 provenance를 run에 고정한다. 같은 run ID의 설정·split·한도를 조용히 변경하거나 다른 모델 결과와 섞지 않는다.
- 산출물은 원자적으로 저장하고, 완료된 항목을 확인해서 재개한다. 같은 산출물에 동시에 쓰는 실행을 막는다. 마지막 미완성 journal 줄 외의 손상을 임의로 고치지 않는다.
- `except Exception: pass` 또는 포괄적인 실패 은폐를 쓰지 않는다. 생성 오류나 judge 오류를 ASR 0 또는 정상 거부로 변환하지 않는다. 기대 수·처리 완료 수·성공 수를 함께 집계하고 오류 원인을 로그에 보존한다.
- 로그에는 필요한 ID와 오류 정보를 우선 기록한다. 원본 prompt·모델 응답·토큰 등 큰 내용은 지정된 산출물에만 저장하고 credential을 출력하지 않는다.

## 데이터와 실험 규칙

- HarmBench 원본을 수정하지 않는다. L1/L2/L3는 프로젝트가 추가한 severity 분류이며 공식 HarmBench 라벨이 아니다. 초기 AI 라벨이라는 provenance를 유지한다.
- 레벨 파일은 behavior ID, level, rationale을 정확히 대응시킨다. 누락·중복·추가 ID·빈 라벨을 자동 추정해서 메우지 않는다.
- 기본 평가는 그룹 기반 5-fold out-of-fold(OOF)이다. 각 fold의 벡터는 다른 네 fold의 fit 문항으로만 계산하고 해당 fold의 held-out 문항에서 평가한다. 모든 문항은 정확히 한 번 held-out 평가에 포함한다. 같은 fold 안에서 fit과 test가 겹치지 않아야 한다.
- 레이어·계수·정규화 등 설정은 OOF 실행 전에 고정하고 held-out 결과로 조정하지 않는다. 설정 탐색이 필요하면 별도 dev 데이터나 nested CV를 사용한다. partial/smoke 결과는 전체 OOF 결과와 구분한다.
- 동일한 정규화 behavior 또는 비어 있지 않은 동일 context로 연결되는 문항을 하나의 그룹으로 유지한다. 모든 fold의 전체 coverage, 중복, 그룹 분리, 레벨 coverage, 데이터 hash를 검증한다. 의미가 비슷한 다른 표현을 자동 검출한다고 주장하지 않는다.
- 모델과 fold마다 자체 벡터 세 개를 계산하고, 각 벡터를 해당 fold의 모든 평가 레벨에 적용한다. baseline은 같은 held-out ID와 생성 설정으로 계산한다.
- 모델별 최종 3×3 ASR_rule은 fold별 비거부 응답 수 합계 / 기대 평가 수 합계로 계산한다. 필요한 생성과 판정이 모두 완료됐을 때만 점수를 제공하고, 미완료 점수와 구간은 null로 둔다. fold별 비율의 단순 평균을 쓰지 않는다. 필수 fold와 평가 항목이 모두 있는지 검증하고 비거부 수·생성 수·판정 수·기대 수·그룹 수·완료 여부를 보고하며 오류 원인을 로그에 보존한다.
- ASR_rule은 고정된 거부 표현 규칙으로 계산한 비거부율이며 유해한 요청을 실제로 수행했는지 확인하는 공식 HarmBench ASR이 아니다. scorer의 버전·규칙 hash를 기록한다. safe 입력의 over_refusal은 같은 rule scorer로 계산한 1 - ASR_rule이다. baseline과 세 벡터 조건을 비교하며 safe 입력에 위해도 L1/L2/L3 라벨을 붙이지 않는다.
- 외부 평가에서는 설정을 먼저 고정하고 HarmBench 전체 문항과 XSTest fit만으로 최종 벡터를 만든다. 외부 유해 문항과 safe 평가 문항을 벡터 추출·설정 선택에 사용하지 않는다. HarmBench와 정규화된 입력/context가 중복된 외부 유해 문항은 제외하고 source/provenance를 남긴다. OOF와 고정 최종 벡터의 외부 결과는 별도로 저장한다.
- XSTest의 test safe 문항만 over-refusal 평가에 사용하고 fit/dev를 독립 test로 합치지 않는다. 외부 benign 데이터는 별도 결과로 보고한다. 외부 severity도 프로젝트의 AI 초기 라벨이며 공식 데이터셋 라벨이 아님을 명시한다.
- OOF 결과는 fold마다 벡터가 달라지는 벡터 생성 절차의 성능이다. 고정 벡터 하나의 평가로 표현하지 않는다. Wilson 95% 구간은 문항 독립을 가정한 참고 구간이며 공유 그룹과 학습 fold 중첩에 따른 불확실성을 모두 반영하지 않는다고 명시한다.
- classifier judge의 예상 출력은 명시적으로 파싱한다. 예상하지 못한 출력은 별도 오류로 남긴다. 공식 프롬프트·규칙·데이터를 재사용할 때 출처와 라이선스를 유지한다.

## 검증과 변경 완료 조건

큰 GPU 실행 전에 `./scripts/check`, 관련 테스트, 모델 preflight, 작은 smoke를 순서대로 확인한다. 데이터 분리, 평균 차이, hook 적용·제거, 응답 token 분리, judge 파싱, 저장·재개 변경에는 이를 직접 검증하는 테스트를 유지한다.

```sh
uv sync --frozen --extra gpu --extra dev
uv run --extra gpu --extra dev pytest
uv run --extra dev ruff check .
```

CPU fixture 테스트 통과를 실제 GPU 실행 성공으로 표현하지 않는다. 선택한 target 모델과 GPU에서 확인한 범위, 아직 확인하지 않은 범위를 구분한다. 설정 변경 후 새 우려가 없으면 같은 검증을 불필요하게 반복하지 않는다.

## 파일 관리

- Git 추적: 소스, 테스트, 실행 스크립트, 문서, 설정, severity 라벨·근거, 고정 fold ID·metadata, `uv.lock`, 출처·라이선스.
- Git 제외: `outputs/`, 원본 다운로드와 파생 데이터, 모델 캐시, `.venv/`, credential, `.env`, 임시 `work/`.
- 실험 산출물은 `outputs/<run_id>/<model_key>/fold_<N>/` 안에 저장하고 모델별 OOF 집계는 `<model_key>/metrics/`에 저장한다. 최종 고정 벡터 및 외부·safe 결과는 `<model_key>/transfer/` 안에 따로 저장한다. run snapshot을 삭제하거나 덮어써서 설정 불일치를 우회하지 않는다.
- GPU 대여·인스턴스 생성은 이 저장소의 준비 작업에 포함되지 않는다. 기존 환경에서 승인된 실험을 실행하는 데 필요한 가역적인 코드 수정·검증은 계속 진행한다.
