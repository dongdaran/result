# 데이터와 재사용 코드 출처

## HarmBench

- 작성자/저장소: Center for AI Safety, [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
- 고정 commit: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`
- 사용하는 파일: `data/behavior_datasets/harmbench_behaviors_text_all.csv`
- 라이선스: MIT, 원문은 `HarmBench-LICENSE`
- 관련 논문: Mantas Mazeika et al., [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249), 2024

다운로드 원본은 보존한다. 이 프로젝트는 standard/contextual text 문항만 선택하고
ID 기반 severity 라벨과 그룹 기반 5-fold OOF를 별도로 추가한다. L1/L2/L3 라벨은
이 프로젝트의 AI 초기 주석이며 upstream의 공식 분류나 사람이 인증한 주석이 아니다.
현재 기본 판정은 아래 GCG 거부 문자열 규칙이다. 이 값은 공식 HarmBench classifier의
ASR과 동일한 점수가 아니다.

## XSTest

- 작성자/저장소: Paul Röttger et al., [paul-rottger/xstest](https://github.com/paul-rottger/xstest)
- 고정 commit: `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`
- 사용하는 파일: `xstest_prompts.csv`의 safe 문항 250개
- 라이선스: Creative Commons Attribution 4.0 International, 원문은 `XSTest-LICENSE`
- 관련 논문: Paul Röttger et al., [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263), 2023

원본 문항을 수정하지 않고 safe subset을 선택한다. 프로젝트의 seed가 고정된
fit 150개는 benign activation 기준, dev 50개는 보존, test 50개는 과도한 거부 평가에
사용한다. XSTest의 safe 데이터는 HarmBench 각 문항과 일대일로 대응하는 중립 문항이
아니다. 외부 평가의 중복 참조에는 fit 150개만 사용한다.

## Hugging Face 모델

target 모델 계열은 Qwen, Llama, Gemma이며 정확한 버전·크기·checkpoint는 아직 선택하지 않았다. 각 모델 및 선택적 `cais/HarmBench-Llama-2-13b-cls`의 접근 조건·라이선스는 해당 Hugging Face model repository에서 확인한다. 이 프로젝트는 모델 가중치를 번들하거나 모델 라이선스를 재정의하지 않는다.

## GCG 거부 문자열 규칙

- 공식 저장소: [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)
- 고정 commit: `098262edf85f807224e70ecd87b9d83716bf6b73`
- 파일: `experiments/evaluate.py`의 29개 `test_prefixes` 거부 문자열
- 파일 SHA-256: `c849b4caa1667314d10310161a218edd1ffea98d38f91088b9798848e82d3afd`
- 라이선스: MIT, Copyright (c) 2023 Andy Zou. 재사용 모듈 `evaluate.py`에도 notice 보존
- 논문: [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), 2023

프로젝트 규칙 버전은 `gcg_refusal_v1`이다. 원본 문자열 목록에 NFKC·casefold 및
apostrophe 정규화를 더하고 전체 생성 응답에서 거부 문자열을 찾는다. 원본 GCG의
생성 길이·평가 범위를 완전히 재현하는 점수로 표현하지 않는다. `ASR_rule`은 거부
표시가 없는 응답 비율이며 유해 내용의 정확성이나 유용성을 판정하지 않는다.
안전 문항에서는 거부 표시가 있는 응답 비율을 과도한 거부로 보고한다. 별도 GPU
classifier 없이 CPU에서 판정하며, 실패한 생성이나 빈 응답을 정상 성공으로 메우지 않는다.

## JailbreakBench 외부 평가

- 작성자: Patrick Chao, Edoardo Debenedetti, Alexander Robey et al.
- 공식 데이터: [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- 고정 HF dataset commit: `886acc352a31533ffbcf4ef22c744658688086fc`
- 파일: `data/harmful-behaviors.csv`, `data/benign-behaviors.csv`
- 원본 SHA-256: harmful `4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1`, benign `3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468`
- 라이선스: MIT, dataset 저장소의 원문은 `JailbreakBench-LICENSE`로 보존
- 논문: [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://arxiv.org/abs/2404.01318), NeurIPS 2024

harmful 100문항의 55개 Original만 후보로 선택하고, AdvBench 18개와 TDC/HarmBench
27개는 출처에 따라 제외한다. CSV의 원본 Goal을 변경하지 않고 namespaced Index로
ID를 부여한다. benign 100문항은 원래 harmful 문항과 주제가 유사한 별도 안전 문항이다.
프로젝트의 severity·의미 중복 제외 판단은 `AI_initial_v1`이며 공식 위해도나 사람이
검수한 주석이 아니다. 외부 참조와의 정확·가까운 문구 검사 및 명시적인 초기 의미
중복 제외를 시행하나, 완전한 의미적 독립성을 보장하지 않는다. 자세한 기준과
제외는 `data/annotations/EXTERNAL.md` 및 준비 manifest에 남긴다.
