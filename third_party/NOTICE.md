# 데이터와 재사용 코드 출처

## HarmBench

- 작성자/저장소: Center for AI Safety, [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
- 고정 commit: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`
- 사용하는 파일: `data/behavior_datasets/harmbench_behaviors_text_all.csv`, `eval_utils.py`의 standard/contextual classifier prompt
- 라이선스: MIT, 원문은 `HarmBench-LICENSE`
- 관련 논문: Mantas Mazeika et al., [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249), 2024

다운로드 원본은 보존한다. 이 프로젝트는 standard/contextual text 문항만 선택하고 ID 기반 severity 라벨과 fit/dev/test split을 별도로 추가한다. L1/L2/L3 라벨은 이 프로젝트의 AI 초기 주석이며 upstream의 공식 분류나 사람이 인증한 주석이 아니다. classifier prompt는 `src/hb_steering/evaluate.py`에 출처와 MIT notice를 유지해 재사용한다.

## XSTest

- 작성자/저장소: Paul Röttger et al., [paul-rottger/xstest](https://github.com/paul-rottger/xstest)
- 고정 commit: `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`
- 사용하는 파일: `xstest_prompts.csv`의 safe 문항 250개
- 라이선스: Creative Commons Attribution 4.0 International, 원문은 `XSTest-LICENSE`
- 관련 논문: Paul Röttger et al., [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263), 2023

원본 문항을 수정하지 않고 safe subset을 선택해 benign activation 기준으로 사용한다. 이 프로젝트에서 seed가 고정된 fit/dev/test 분할을 추가한다. XSTest의 safe 데이터는 HarmBench 각 문항과 일대일로 대응하는 중립 문항이 아니다.

## Hugging Face 모델

target 모델은 아직 선택하지 않았다. 각 모델 및 `cais/HarmBench-Llama-2-13b-cls`의 접근 조건·라이선스는 해당 Hugging Face model repository에서 확인한다. 이 프로젝트는 모델 가중치를 번들하거나 모델 라이선스를 재정의하지 않는다.
