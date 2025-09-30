# 1주차 워크숍: LLM 개요 및 개발 환경 구축

## **워크숍 소개: 대형 언어 모델의 여정 탐험**

이번 LLM 워크숍에 참여하신 것을 환영한다. 1주차 목표는 다음과 같다.

1. 아이디어 구상부터 배포 및 유지보수에 이르는 대형 언어 모델(LLM) 프로젝트의 전체 수명주기에 대한 포괄적인 이해를 확립한다.
2. 업계 표준 도구를 사용하여 강력한 GPU 가속 개발 환경을 구축한다.
3. 사전 훈련된 LLM을 활용하여 첫 번째 텍스트 생성 과제를 성공적으로 수행한다.

이번 워크숍에서는 두 가지 핵심 도구를 중심으로 실습을 진행한다. 첫 번째는 **Hugging Face Transformers**다. 이 라이브러리는 방대한 사전 훈련 모델과 데이터셋에 접근할 수 있는 사실상의 표준으로, NLP 커뮤니티의 "공용어"와 같은 역할을 한다. 두 번째는

**NVIDIA NeMo 프레임워크**다. NeMo는 대규모 생성형 AI 모델을 구축, 맞춤화, 배포하기 위해 설계된 엔터프라이즈급 클라우드 네이티브 프레임워크다. 이 프레임워크의 핵심 가치는 NVIDIA 하드웨어 생태계에 깊이 최적화된 성능과 확장성에 있다.

본 워크숍은 Hugging Face의 광범위한 모델 및 데이터 생태계와 NeMo의 강력하고 확장 가능한 훈련 및 맞춤화 기능을 결합하여, 현대 LLM 개발에 있어 두 세계의 장점을 모두 활용하는 접근 방식을 제시한다.

---

## **파트 1: LLM 전체 수명주기 심층 분석**

LLM 프로젝트는 단순한 모델 훈련을 넘어, 명확한 목표 설정부터 지속적인 유지보수까지 이어지는 복잡한 과정을 포함한다. 이 과정은 여러 단계를 거치며, 각 단계는 후속 단계에 지대한 영향을 미친다.

### **1단계: 범위 설정 및 문제 정의**

- **목표:** 모호한 아이디어를 명확한 비즈니스 목표와 기술적 제약 조건을 갖춘 잘 정의된 프로젝트로 전환한다.
- **주요 활동:**
  - **문제 이해:** 해결하고자 하는 문제를 명확하게 기술한다. 예를 들어, "고객 지원 업무량 감소" 또는 "마케팅 문구 자동 생성"과 같이 구체적이어야 한다.
  - **실현 가능성 분석:** 사용 가능한 기술, 데이터, 리소스를 검토하고 LLM이 해당 문제를 해결하기에 적합한 도구인지 평가한다.
  - **성공 지표 정의:** 정량화 가능한 핵심 성과 지표(KPI)를 설정한다. 예를 들어, "지원 업무량 30% 감소", "고객 만족도 4.5/5 이상 유지", "응답 지연 시간 2초 미만" 등이 있다.
  - **제약 조건 식별:** 데이터 개인정보 보호(GDPR 등), 지연 시간 요구사항, 인간 상담원에게 이관해야 하는 조건 등 모든 운영, 법률, 윤리적 경계를 목록화한다.

### **2단계: 데이터 수집 및 정제**

- **목표:** 모델의 지식과 행동의 기반이 될 고품질의 다양하고 대표성 있는 데이터셋을 구축한다.
- **주요 활동:**
  - **데이터 소싱:** 서적, 웹사이트, 기사, 코드 등 다양한 출처에서 방대한 양의 텍스트를 수집한다.
  - **전처리 및 정제:** 노이즈 제거, 정규화, 중복 제거, 저품질 또는 유해 콘텐츠 필터링 등 매우 중요한 단계를 수행한다.
  - **토큰화:** 원시 텍스트를 모델이 처리할 수 있는 숫자 형식(토큰)으로 변환한다.
  - **편향 및 개인정보 문제 완화:** 소스 데이터에 존재하는 편향을 완화하고 개인 식별 정보(PII) 제거 등 개인정보 보호 문제를 해결하기 위해 적극적으로 노력한다.

### **3단계: 사전 훈련 (Pre-training)**

- **목표:** 레이블이 없는 대규모 코퍼스를 기반으로 범용 "기반 모델(Foundation Model)"을 훈련한다. 이 모델은 문법, 사실, 추론 능력 및 언어에 대한 광범위한 이해를 학습한다.
- **주요 활동:**
  - **비지도 학습:** 일반적으로 다음 토큰 예측(next-token prediction) 목표를 사용한다. 모델은 문장에서 다음 단어를 예측하며 학습한다.
  - **대규모 컴퓨팅:** 수천만 GPU 시간을 요구하는, 재정 및 계산적으로 가장 비용이 많이 드는 단계다.
- **도전 과제:** 막대한 비용, 과적합(overfitting) 위험, 신중한 정규화(regularization) 기법의 필요성 등이 있다.

### **4단계: 지도 미세 조정 (Supervised Fine-Tuning, SFT)**

- **목표:** 더 작고 레이블이 지정된 데이터셋을 사용하여 범용 기반 모델을 특정 작업이나 도메인에 맞게 조정한다.
- **주요 활동:**
  - **고품질 데이터셋 생성:** 원하는 입출력 쌍(예: 지시-응답 쌍)의 예시를 큐레이션하거나 직접 생성한다.
  - **작업별 훈련:** 요약, 질의응답, 지시 따르기 등 특정 작업에 대한 성능을 향상시키기 위해 이 레이블된 데이터로 모델을 추가 훈련한다.
- **도전 과제:** 특정 작업에 전문화되면서 일반적인 능력을 잃는 파국적 망각(catastrophic forgetting) 현상과, 상당한 양의 고품질 레이블 데이터 확보의 어려움이 있다.

### **5단계: 정렬 및 안전성 조정 (RLHF/DPO)**

- **목표:** 모델의 행동을 인간의 가치, 선호도, 안전 가이드라인에 맞춰 정렬하여 더 유용하고, 무해하며, 정직하게 만든다.
- **주요 활동:**
  - **인간 피드백 기반 강화 학습 (RLHF):** (1) 인간 선호도 데이터 수집(다양한 모델 출력에 순위 매기기), (2) 인간 선호도를 예측하는 "보상 모델" 훈련, (3) 강화 학습(PPO 등)을 사용하여 보상 점수를 최대화하도록 LLM 미세 조정의 3단계 과정을 거친다.
  - **직접 선호도 최적화 (DPO):** 별도의 보상 모델 훈련 없이 동일한 선호도 데이터를 사용하여 LLM을 직접 미세 조정하는 더 새롭고 직접적인 방법이다.
- **도전 과제:** 모델이 진정으로 유용하지 않으면서 높은 보상을 받는 허점을 찾는 보상 해킹(reward hacking)과, 인간 피드백을 확장하는 데 드는 높은 비용 및 복잡성이 있다.

### **6단계: 평가 및 벤치마킹**

- **목표:** 광범위한 학술 벤치마크와 맞춤형 작업별 테스트셋을 통해 모델의 성능을 엄격하게 테스트한다.
- **주요 활동:**
  - **자동화된 메트릭:** 요약 작업의 ROUGE 점수나 분류 작업의 정확도와 같은 지표를 사용한다.
  - **인간 평가:** 자동화된 메트릭으로 포착하기 어려운 유용성, 일관성, 안전성과 같은 품질을 평가한다.
  - **레드 티밍 (Red Teaming):** 모델의 취약점, 편향 또는 안전성 실패를 적극적으로 찾아내고 악용하려는 시도를 한다.

### **7단계: 배포 및 추론 최적화**

- **목표:** 훈련되고 정렬된 모델을 신뢰할 수 있고 확장 가능하며 비용 효율적인 방식으로 실제 애플리케이션에서 사용할 수 있도록 한다.
- **주요 활동:**
  - **모델 최적화:** 모델 크기를 줄이고 지연 시간을 개선하기 위해 양자화(가중치 정밀도 감소), 프루닝(불필요한 가중치 제거), 증류(더 작은 모델이 더 큰 모델을 모방하도록 훈련)와 같은 기술을 적용한다.
  - **서빙 인프라:** Triton이나 vLLM과 같은 추론 서버를 사용하여 클라우드 인프라, 온프레미스 서버 또는 엣지 디바이스에 모델을 배포한다.

### **8단계: 모니터링 및 유지보수**

- **목표:** 시간이 지나도 모델의 성능이 높게 유지되도록 보장하고 실제 사용 데이터를 기반으로 지속적으로 개선한다.
- **주요 활동:**
  - **성능 모니터링:** 지연 시간, 오류율, 사용자 만족도와 같은 지표를 추적한다.
  - **드리프트 감지:** 입력 데이터 패턴의 변화("데이터 드리프트")로 인해 모델 성능이 저하되는 시점을 식별한다.
  - **지속적 개선 루프 (데이터 플라이휠):** 실제 추론 데이터와 사용자 피드백을 사용하여 추가 미세 조정 및 모델 업데이트를 위한 새로운 훈련 데이터를 생성한다.

### **체크포인트 질문 1: LLM 수명주기에서 가장 중요한 단계는 무엇인가?**

모든 단계가 상호 의존적이지만, **범위 설정(1단계)과 데이터 정제(2단계)** 단계가 가장 중요하다고 할 수 있다. 잘못 정의된 문제는 잘못된 솔루션 개발로 이어져 모든 후속 노력을 낭비하게 만든다. 마찬가지로, 저품질의 편향되거나 관련 없는 데이터는 사전 훈련에 얼마나 많은 컴퓨팅 자원을 투입하든, 정렬 기술이 얼마나 정교하든 상관없이 모델의 잠재력을 근본적으로 제한한다. "쓰레기가 들어가면 쓰레기가 나온다"는 원칙이 철저히 적용된다. 이러한 초기 단계에서 발생한 오류는 나중에 수정하기 매우 어렵다. 결함 있는 데이터로 사전 훈련된 모델은 정렬하는 데 훨씬 더 많은 노력이 필요하며, 원하는 성능이나 안전 수준에 결코 도달하지 못할 수도 있다.

이러한 단계들을 선형적인 과정으로 이해하기 쉽지만, 실제 LLM 개발은 훨씬 더 동적이고 반복적인 활동이다. 예를 들어, 모니터링 단계(8단계)에서 성능 저하가 감지되면, 이는 단순히 모델을 재배포하는 것으로 끝나지 않는다. 이 피드백은 새로운 미세 조정 데이터셋을 구축(4단계)하거나, 심지어 근본적인 데이터 품질 문제(2단계)를 해결하기 위해 이전 단계로 돌아가는 계기가 된다. 따라서 LLM 수명주기는 일직선이 아니라, 각 단계가 서로에게 피드백을 제공하며 지속적으로 개선을 이끌어내는 "데이터 플라이휠(Data Flywheel)"과 같은 순환적인 시스템으로 이해해야 한다.

---

## **파트 2: 개발 환경 구축: NVIDIA NGC 실습 가이드**

이 섹션에서는 워크숍을 위한 개발 환경을 단계별로 구축하는 방법을 안내한다.

### **사전 요구사항 체크리스트**

- **하드웨어:** NVIDIA GPU (WSL2의 경우 Pascal 아키텍처 이상 권장).
- **소프트웨어:**
  - 운영체제에 맞는 최신 NVIDIA 드라이버
  - Docker Engine 설치 및 실행
  - Windows 사용자: Windows 10 (21H2 이상) 또는 Windows 11, WSL2 활성화 및 Linux 배포판(예: Ubuntu) 설치.

### **단계별 설치 가이드**

1. **NGC 접속:**
   - NGC 웹사이트(ngc.nvidia.com)로 이동하여 로그인하거나 무료 계정을 생성한다.
2. **API 키 생성:**
   - Setup \> API Keys로 이동하여 Generate Personal Key를 클릭한다. 이 키는 NGC 서비스에 프로그래밍 방식으로 접근하기 위한 비밀번호 역할을 한다.
   - **중요:** 생성된 키는 즉시 복사하여 안전한 곳에 보관해야 한다. NGC는 키를 저장하지 않는다.
3. **Docker와 NGC 레지스트리 인증:**
   - 터미널(Windows의 경우 PowerShell)을 연다.
   - docker login nvcr.io 명령을 실행한다.
   - Username 프롬프트에 $oauthtoken이라는 문자열을 그대로 입력한다.
   - Password 프롬프트에 방금 생성한 NGC API 키를 붙여넣는다.
4. **워크숍 컨테이너 다운로드:**
   - 재현성을 보장하기 위해 특정 버전의 NVIDIA PyTorch 컨테이너를 사용한다.
   - docker pull nvcr.io/nvidia/pytorch:23.10-py3 명령을 실행한다. 이 컨테이너에는 PyTorch, CUDA, cuDNN 등 NVIDIA GPU에 최적화된 필수 라이브러리가 모두 포함되어 있다.
5. **대화형 컨테이너 실행:**

   - 다음 명령을 실행한다. 각 플래그의 의미는 다음과 같다.  
     Bash  
     docker run \--gpus all \-it \--rm \-v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.10-py3

   - \--gpus all: 사용 가능한 모든 호스트 GPU를 컨테이너에 노출시켜 GPU 가속 환경의 핵심을 이룬다.
   - \-it: 컨테이너를 대화형 모드로 실행하여 컨테이너 내부 셸에 접근할 수 있게 한다.
   - \--rm: 컨테이너를 종료할 때 자동으로 삭제하여 시스템을 깨끗하게 유지한다.
   - \-v $(pwd):/workspace: 호스트 머신의 현재 디렉토리($(pwd))를 컨테이너 내부의 /workspace 디렉토리에 마운트한다. 작업 내용을 저장하고 로컬 데이터에 접근하는 데 필수적이다.

6. **GPU 접근 확인:**
   - 컨테이너 셸 내부에서 nvidia-smi를 실행한다.
   - GPU, 드라이버 버전, CUDA 버전이 포함된 테이블이 출력되면 컨테이너가 하드웨어에 정상적으로 접근하고 있는 것이다.

### **필수 Python 라이브러리 설치**

실행 중인 컨테이너 내부에서 다음 pip 명령을 실행하여 NeMo와 Hugging Face 생태계를 설치한다.

Bash

\# Hugging Face 라이브러리 설치  
pip install transformers datasets accelerate

\# 전체 NVIDIA NeMo 툴킷 설치  
pip install nemo-toolkit\[all\]

### **문제 해결 가이드**

원활한 워크숍 진행을 위해 가장 일반적인 설치 문제를 미리 해결하는 것이 중요하다.

| 오류 메시지                                                                                                    | 일반적인 원인                                                                                                                                            | 해결책                                                                                                                                                                                                                                                                                                                                                                                       |
| :------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| docker: Error response from daemon: OCI runtime create failed... nvidia-container-cli: initialization error... | 호스트의 NVIDIA 드라이버 문제; NVIDIA Container Toolkit 미설정; Docker 데몬 재시작 필요.                                                                 | 1\. 호스트 드라이버 확인: 호스트 머신에 최신 NVIDIA 드라이버가 올바르게 설치되었는지 확인합니다. 호스트에서 nvidia-smi가 작동해야 합니다. 2\. 툴킷 설정: sudo nvidia-ctk runtime configure \--runtime=docker 실행 후 sudo systemctl restart docker를 실행합니다. 3\. WSL2 특이사항: WSL2 내부에 Linux GPU 드라이버를 설치하지 않았는지 확인합니다. Windows 드라이버가 자동으로 연동됩니다.18 |
| Failed to initialize NVML: Driver/library version mismatch                                                     | 커널에 로드된 NVIDIA 드라이버 버전과 사용자 공간 라이브러리(libnvidia-ml.so) 버전이 다릅니다. 드라이버 업데이트 후 재부팅하지 않았을 때 자주 발생합니다. | 1\. 가장 쉬운 해결책: 재부팅. 시스템 재부팅이 커널 모듈과 사용자 공간 라이브러리를 동기화하는 가장 확실한 방법입니다. 2\. 수동 리로드 (고급): 모든 NVIDIA 커널 모듈을 언로드(sudo rmmod...)한 후 다시 로드합니다. 복잡하고 위험하므로 재부팅을 권장합니다. 3\. 클린 재설치: 모든 NVIDIA 드라이버를 완전히 제거하고 최신 버전을 새로 설치합니다.27                                            |
| WSL2 내부에서 nvidia-smi 실행 시 couldn't communicate with the NVIDIA driver 오류                              | (구 버전 문제) 초기 WSL2 버전에서 NVML이 완전히 지원되지 않았습니다.                                                                                     | 1\. WSL 커널 업데이트: PowerShell에서 wsl \--update를 실행하여 최신 커널을 받습니다. 2\. Windows 업데이트: 최신 빌드의 Windows 10/11을 사용 중인지 확인합니다. 3\. NVIDIA 드라이버 업데이트: WSL2 지원을 위해 설계된 최신 드라이버를 설치합니다.18                                                                                                                                           |

사전 구성된 NGC 컨테이너를 사용하는 것은 단순한 편의성을 넘어, 과학적 재현성을 보장하고 의존성 문제를 완화하는 근본적인 실천이다. 개발자마다 다른 환경에서 수동으로 드라이버, 툴킷, 라이브러리를 설치하려는 시도는 실패와 시간 낭비로 이어지기 쉽다. NGC 컨테이너는 NVIDIA가 최적화하고 테스트한 드라이버, CUDA 라이브러리, cuDNN, PyTorch의 완벽한 조합을 캡슐화한다. 모든 참가자가 동일한 컨테이너(

nvcr.io/nvidia/pytorch:23.10-py3)로 시작함으로써, "내 컴퓨터에서는 되는데"라는 종류의 문제를 원천적으로 차단한다. 이 컨테이너화 접근 방식은 복잡하고 오류가 발생하기 쉬운 환경 설정 작업을 단일 docker run 명령으로 전환하여, 모든 참가자에게 일관된 기준선을 제공한다. 이는 현대 MLOps의 초석이자 그 자체로 중요한 교훈이다.

---

## **파트 3: 첫 만남: Hugging Face Transformers로 LLM 실행하기**

이 섹션에서는 구축된 환경의 강력함을 보여주는 첫 번째 핸즈온 코딩 실습을 진행한다.

### **단순함의 힘: pipeline API**

Hugging Face pipeline은 추론을 위한 가장 높은 수준의 사용하기 쉬운 API다. 이 API는 (1) 토큰화(전처리), (2) 모델 추론(순전파), (3) 출력 디코딩(후처리)의 세 가지 핵심 단계를 추상화하여 제공한다.

### **실습 1: 첫 텍스트 생성**

다음은 제공된 Python 스크립트다.

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# 간단한 텍스트 생성 파이프라인
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1
)

# 텍스트 생성 테스트
prompt = "The future of Artificial Intelligence is"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

- **코드 분석:**
  - pipeline("text-generation", model="gpt2",...): 파이프라인을 인스턴스화한다.
    - task="text-generation": 원하는 작업을 지정한다. 사용 가능한 전체 작업 목록은 문서에서 확인할 수 있다.
    - model="gpt2": Hugging Face Hub에서 사용할 모델을 지정한다. gpt2는 작은 크기와 친숙함 덕분에 좋은 시작점이다.
    - device=0: GPU 가속의 핵심이다. 파이프라인에 모델과 데이터를 첫 번째 GPU(CUDA 장치 0)에 올리도록 지시한다.

### **실습 2: 한국어 텍스트 생성**

배운 내용을 한국어 모델에 적용해 본다. EleutherAI/polyglot-ko-1.3b는 잘 문서화된 오픈소스(Apache 2.0 라이선스) 한국어 LLM으로 좋은 선택이다.

Python

\# 한국어 텍스트 생성 파이프라인  
korean_generator \= pipeline("text-generation",  
 model="EleutherAI/polyglot-ko-1.3b",  
 device=0 if torch.cuda.is_available() else \-1)

prompt \= "대한민국 인공지능의 미래는"  
result \= korean_generator(prompt, max_length=50, num_return_sequences=1)  
print(result\['generated_text'\])

### **출력 제어: 생성 파라미터 가이드**

기본 생성 결과는 반복적이거나 의미가 없을 수 있다. 디코딩 전략을 제어하는 파라미터를 사용하여 "작동하게" 만드는 것을 넘어 "잘 작동하게" 만들 수 있다.

| 파라미터                    | 설명                                                                                                                               | 출력에 미치는 영향                                                                                             | 권장 사용 사례                                                                                   |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| max_length / max_new_tokens | 출력 시퀀스의 최대 총 길이 또는 생성할 새 토큰의 최대 개수다.                                                                      | 응답 길이를 제어하고 무한 루프를 방지한다. 애플리케이션 요구에 맞게 적절한 값으로 설정한다.                    |                                                                                                  |
| temperature                 | 0보다 큰 실수. 다음 토큰 확률을 조절한다. 낮은 값(\<1.0)은 모델을 더 결정론적으로 만들고, 높은 값(\>1.0)은 더 무작위적으로 만든다. | 낮음: 예측 가능, 보수적, 반복적. 높음: 창의적, 다양함, 무의미한 결과 생성 위험 증가.                           | **창의적 작업:** 0.7 \- 1.0. **사실 기반 작업:** 0.2 \- 0.5.                                     |
| top_k                       | 정수. 어휘를 가장 가능성 있는 k개의 다음 토큰으로 필터링한다. 모델은 이 축소된 집합에서 샘플링한다.                                | 선택 풀을 제한하여 매우 낮은 확률의 토큰이 선택되는 것을 방지한다. 너무 낮게 설정하면 창의성을 저해할 수 있다. | 시작점으로 50이 적합하다. 이상한 단어 선택을 피하고 싶을 때 사용한다.                            |
| top_p (Nucleus Sampling)    | 0과 1 사이의 실수. 누적 확률이 p를 초과하는 가장 작은 토큰 집합으로 어휘를 필터링한다.                                             | top_k보다 동적이다. 다음 토큰의 예측 가능성에 따라 어휘 크기를 조정한다. 더 나은 범용 샘플링 방법이다.         | 시작점으로 0.9에서 0.95가 적합하다. OpenAI는 temperature나 top_p 중 하나만 변경할 것을 권장한다. |
| num_return_sequences        | 정수. 독립적으로 생성할 시퀀스의 수다.                                                                                             | 동일한 프롬프트에서 여러 다른 완료 문장을 생성할 수 있다.                                                      | 브레인스토밍이나 사용자에게 여러 옵션을 제공할 때 유용하다.                                      |
| no_repeat_ngram_size        | 정수. 이 크기의 n-gram이 두 번 이상 나타나는 것을 방지한다.                                                                        | 탐욕적/빔 검색의 일반적인 실패 모드인 반복적인 구문 문제를 직접적으로 해결한다.                                | 유창성을 높이고 명백한 반복을 줄이려면 2 또는 3으로 설정한다.                                    |

다양한 디코딩 전략은 단순히 임의의 값을 조정하는 것이 아니라, 창의적이고 다양한 텍스트를 생성하는 것과 일관되고 예측 가능한 텍스트를 생성하는 것 사이의 근본적인 긴장 관계를 관리하는 접근법을 나타낸다. LLM의 핵심은 다음 토큰에 대한 어휘의 확률 분포다. 순수하게 가장 가능성 있는 토큰만 선택하는 탐욕적 접근 방식은 결정론적이지만 종종 단조롭고 부자연스러운 텍스트를 생성한다. 창의성을 도입하려면 분포에서 샘플링해야 하지만, 제약 없는 샘플링은 무의미한 결과를 낳는다. 따라서

top_k, top_p, temperature와 같은 파라미터는 순수한 활용(탐욕적 검색)과 순수한 탐색(제약 없는 샘플링) 사이의 스펙트럼을 탐색하는 도구다. 주어진 작업에 맞는 올바른 균형을 찾는 것이 추론 구성의 핵심 기술이다.

---

## **파트 4: 두 프레임워크 이야기: NeMo와 Hugging Face**

이 섹션에서는 워크숍의 두 핵심 프레임워크를 비교하며 두 번째 체크포인트 질문에 대한 심층적인 답변을 제공한다.

### **철학적 심층 분석**

- **Hugging Face:**
  - **핵심 철학:** 민주화와 협업. 최첨단 ML을 모든 사람이 접근할 수 있도록 만드는 것이 목표다.
  - **구현:** 수십만 개의 모델과 데이터셋이 있는 거대한 커뮤니티 기반 Model Hub, 사용하기 쉬운 고수준 API(  
    pipeline), 그리고 업계 표준이 된 오픈소스 라이브러리(transformers, datasets, accelerate)로 나타난다.
- **NVIDIA NeMo:**
  - **핵심 철학:** 성능, 확장성, 엔터프라이즈 준비성. 단일 GPU 실험부터 대규모 다중 노드 훈련 클러스터에 이르기까지 NVIDIA 하드웨어에서 생성형 AI 모델을 효율적으로 구축, 맞춤화, 배포하는 엔드투엔드 플랫폼을 제공하는 것이 목표다.
  - **구현:** NVIDIA 하드웨어 및 소프트웨어 스택(CUDA, Transformer Engine, Megatron-Core)과의 깊은 통합, 고급 병렬 처리 기술(텐서, 파이프라인, 데이터) 지원, 그리고 데이터 큐레이션(  
    NeMo Curator) 및 최적화된 배포를 포함한 전체 MLOps 수명주기에 대한 집중으로 나타난다.

### **비교 분석: NeMo 대 Hugging Face Transformers**

| 기능            | Hugging Face Transformers                                                                                           | NVIDIA NeMo 프레임워크                                                                                            |
| :-------------- | :------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------- |
| **주요 목표**   | 접근성, 커뮤니티, 사용 편의성                                                                                       | 성능, 확장성, 엔터프라이즈급 훈련                                                                                 |
| **아키텍처**    | PyTorch/TensorFlow/JAX 기반의 고수준 API(pipeline) 및 모듈식 Auto\* 클래스. 핵심적으로 프레임워크에 구애받지 않음.3 | PyTorch Lightning 및 Megatron-Core, Transformer Engine과 같은 NVIDIA 자체 고성능 백엔드와 긴밀하게 통합됨.17      |
| **핵심 강점**   | **생태계:** Hub의 독보적인 모델 및 데이터셋 컬렉션. 추론 및 기본 미세 조정을 시작하기 매우 쉬움.2                   | **대규모 성능:** 수천 개의 GPU에서 대규모 모델 훈련에 최적화됨. NVIDIA 하드웨어에서 우수한 처리량과 효율성 제공.5 |
| **사용 편의성** | **매우 높음.** pipeline API는 초보자에게 친숙함. 광범위한 문서와 커뮤니티 지원.1                                    | **중상급.** 분산 훈련 구성에 중점을 두어 학습 곡선이 더 가파름 (NeMo 2.0의 Python 기반 설정으로 개선됨).34        |
| **구성**        | 주로 Python 코드를 통해 프로그래밍 방식. Trainer API를 위한 TrainingArguments 클래스 사용.12                        | 역사적으로 YAML 기반. NeMo 2.0은 더 유연하고 강력한 Python 기반 구성 시스템(Fiddle)으로 전환.34                   |
| **사용 사례**   | 신속한 프로토타이핑, 추론 애플리케이션, 학술 연구, 중소 규모 모델 미세 조정.                                        | 기반 모델 처음부터 사전 훈련, 독점 데이터에 대한 대규모 SFT/RLHF, 기업 R\&D.                                      |

### **격차 해소: 상호 운용성과 공존**

개발자들은 "이것 아니면 저것"의 선택을 강요받지 않는다. 두 프레임워크는 점점 더 상호 보완적으로 작동하고 있다.

- **NeMo의 AutoModel 기능:** 이는 판도를 바꾸는 기능이다. 수동 변환 단계 없이 Hugging Face 모델을 직접 로드하고 미세 조정하도록 설계된 NeMo 내의 고수준 인터페이스다. 이를 통해 Hugging Face Hub의 방대한 생태계와 NeMo의 고성능 훈련 환경을 결합하여 새로운 모델에 대한 "Day-0 지원"이 가능해진다.
- **체크포인트 변환 스크립트:** AutoModel을 사용하지 않을 경우, NeMo는 convert_llama_hf_to_nemo.py와 같은 명시적인 스크립트를 제공하여 Hugging Face와 NeMo의 .nemo 형식 간에 모델 가중치를 변환할 수 있다.
- **HFDatasetDataModule을 통한 데이터 통합:** NeMo는 Hugging Face datasets 라이브러리의 데이터셋을 NeMo 훈련 파이프라인 내에서 직접 래핑하고 사용할 수 있는 전용 데이터 모듈(HFDatasetDataModule)을 제공하여 데이터 로딩 프로세스를 간소화한다.

NeMo와 Hugging Face 간의 상호 운용성 증가는 AI 개발 생태계가 성숙하고 있음을 보여준다. 이는 폐쇄적인 단일 플랫폼으로 경쟁하기보다, 전문화된 도구들이 함께 작동하도록 설계되고 있음을 의미한다. Hugging Face는 모델과 데이터를 공유하고 발견하는 보편적인 "거래소" 역할을 하고, NeMo는 전문화된 "고성능 엔진" 역할을 한다. 이러한 모듈식, 상호 운용 가능한 접근 방식은 개발자가 LLM 수명주기의 각 단계에 가장 적합한 도구를 사용하여 모두를 위한 혁신을 가속화할 수 있게 한다.

---

## **결론 및 1주차 팀 챌린지**

### **1주차 요약**

이번 주에는 LLM 수명주기에 대한 견고한 개념적 모델을 구축하고, 완전히 작동하는 GPU 기반 개발 환경을 설정했으며, 구성 가능한 출력을 가진 사전 훈련된 모델을 직접 실행하는 실습 경험을 쌓았다.

### **2주차 예고**

다음 주에는 datasets와 NeMo Curator를 사용한 데이터 준비 심층 분석과, 맞춤형 데이터셋에 대한 지도 미세 조정(SFT) 수행에 대해 다룰 예정이다.

### **1주차 팀 챌린지 (권장)**

- **목표:** 이번 주에 배운 기술을 실제 팀 기반 탐색 과제에 적용한다.
- **과제:**
  1. **탐색:** 팀원들과 함께 Hugging Face Hub를 탐색한다.
  2. **작업 선택:** 관심 있는 작업(예: 요약, 감성 분석)을 선택한다.
  3. **한국어 데이터셋 찾기:** 선택한 작업에 적합한 한국어 데이터셋을 찾는다.
     - _감성 분석 추천:_ e9t/nsmc (네이버 영화 리뷰 코퍼스).
     - _요약 추천:_ nglaura/koreascience-summarization 또는  
       gogamza/kobart-summarization과 같은 모델 탐색.
  4. **한국어 모델 찾기:** 작업에 적합한 사전 훈련된 모델을 찾는다. EleutherAI/polyglot-ko-1.3b와 같은 일반 모델이나 작업별로 미세 조정된 모델이 될 수 있다.
  5. **실험:** NGC 컨테이너 환경에서 pipeline API를 사용하여 모델을 로드하고 선택한 데이터셋의 3-5개 예시에 대해 추론을 실행한다.
  6. **토론:** 팀원들과 결과를 분석한다. 결과가 좋은가? 어떻게 개선할 수 있을까? 이 토론은 다음 주 미세 조정 주제에 대한 완벽한 진입점이 될 것이다.

## **참고자료**

1. What are the features of Hugging Face's Transformers? \- Milvus, accessed September 28, 2025, [https://milvus.io/ai-quick-reference/what-are-the-features-of-hugging-faces-transformers](https://milvus.io/ai-quick-reference/what-are-the-features-of-hugging-faces-transformers)
2. Hugging Face Transformers: Leverage Open-Source AI in Python, accessed September 28, 2025, [https://realpython.com/huggingface-transformers/](https://realpython.com/huggingface-transformers/)
3. Introduction to Hugging Face Transformers \- GeeksforGeeks, accessed September 28, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/Introduction-to-hugging-face-transformers/](https://www.geeksforgeeks.org/artificial-intelligence/Introduction-to-hugging-face-transformers/)
4. NVIDIA NeMo Framework, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/index.html](https://docs.nvidia.com/nemo-framework/index.html)
5. What Is The Difference Between NVIDIA NeMo Framework & NeMo Microservices?, accessed September 28, 2025, [https://cobusgreyling.medium.com/what-is-the-difference-between-nvidia-nemo-framework-nemo-microservices-9339dbb62226](https://cobusgreyling.medium.com/what-is-the-difference-between-nvidia-nemo-framework-nemo-microservices-9339dbb62226)
6. LLM Development: Step-by-Step Phases of LLM Training \- lakeFS, accessed September 28, 2025, [https://lakefs.io/blog/llm-development/](https://lakefs.io/blog/llm-development/)
7. LLM Project Lifecycle: Revolutionized by Generative AI \- Data Science Dojo, accessed September 28, 2025, [https://datasciencedojo.com/blog/llm-project-lifecycle/](https://datasciencedojo.com/blog/llm-project-lifecycle/)
8. Large Language Model Lifecycle: An Examination Challenges, accessed September 28, 2025, [https://www.computer.org/publications/tech-news/trends/large-language-model-lifecycle/](https://www.computer.org/publications/tech-news/trends/large-language-model-lifecycle/)
9. What are the Stages of the LLMOps Lifecycle? \- Klu.ai, accessed September 28, 2025, [https://klu.ai/glossary/llm-ops-lifecycle](https://klu.ai/glossary/llm-ops-lifecycle)
10. The LLM Project Lifecycle: A Practical Guide | by Tony Siciliani \- Medium, accessed September 28, 2025, [https://medium.com/@tsiciliani/the-llm-project-lifecycle-a-practical-guide-9117228664d4](https://medium.com/@tsiciliani/the-llm-project-lifecycle-a-practical-guide-9117228664d4)
11. LLM Development Life Cycle, accessed September 28, 2025, [https://muoro.io/llm-development-life-cycle](https://muoro.io/llm-development-life-cycle)
12. Analysis of the Hugging Face Transformers Library: Purpose and Component Classes : 1, accessed September 28, 2025, [https://medium.com/@danushidk507/analysis-of-the-hugging-face-transformers-library-purpose-and-component-classes-1-8f5bdc7a3b17](https://medium.com/@danushidk507/analysis-of-the-hugging-face-transformers-library-purpose-and-component-classes-1-8f5bdc7a3b17)
13. LLM Post-Training: A Deep Dive into Reasoning Large Language Models \- arXiv, accessed September 28, 2025, [https://arxiv.org/pdf/2502.21321](https://arxiv.org/pdf/2502.21321)
14. Understanding the Effects of RLHF on LLM Generalisation and Diversity \- arXiv, accessed September 28, 2025, [https://arxiv.org/html/2310.06452v2](https://arxiv.org/html/2310.06452v2)
15. Reinforcement Learning From Human Feedback (RLHF) For LLMs \- Neptune.ai, accessed September 28, 2025, [https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms](https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms)
16. Summarization \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/tasks/summarization](https://huggingface.co/docs/transformers/tasks/summarization)
17. Run Hugging Face Models Instantly with Day-0 Support from NVIDIA ..., accessed September 28, 2025, [https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework/](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework/)
18. CUDA on WSL User Guide — CUDA on WSL 13.0 documentation, accessed September 28, 2025, [https://docs.nvidia.com/cuda/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
19. Enable NVIDIA CUDA on WSL 2 \- Microsoft Learn, accessed September 28, 2025, [https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
20. GPU support \- Docker Docs, accessed September 28, 2025, [https://docs.docker.com/desktop/features/gpu/](https://docs.docker.com/desktop/features/gpu/)
21. 1\. Overview — NVIDIA GPU Cloud Documentation, accessed September 28, 2025, [https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
22. Containers For Deep Learning Frameworks User Guide \- NVIDIA Docs, accessed September 28, 2025, [https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
23. Troubleshooting — NVIDIA Container Toolkit \- NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)
24. Docker Nvidia Runtime error : r/archlinux \- Reddit, accessed September 28, 2025, [https://www.reddit.com/r/archlinux/comments/1mqlv5k/docker_nvidia_runtime_error/](https://www.reddit.com/r/archlinux/comments/1mqlv5k/docker_nvidia_runtime_error/)
25. NVIDIA Docker \- initialization error: nvml error: driver not loaded · Issue \#1393 \- GitHub, accessed September 28, 2025, [https://github.com/NVIDIA/nvidia-docker/issues/1393](https://github.com/NVIDIA/nvidia-docker/issues/1393)
26. Docker Fails to Launch GPU Containers with NVIDIA Runtime, but Podman Works, accessed September 28, 2025, [https://forums.docker.com/t/docker-fails-to-launch-gpu-containers-with-nvidia-runtime-but-podman-works/147966](https://forums.docker.com/t/docker-fails-to-launch-gpu-containers-with-nvidia-runtime-but-podman-works/147966)
27. GPU Troubleshooting Guide: Resolving Driver/Library Version Mismatch Errors, accessed September 28, 2025, [https://support.exxactcorp.com/hc/en-us/articles/32810166604183-GPU-Troubleshooting-Guide-Resolving-Driver-Library-Version-Mismatch-Errors](https://support.exxactcorp.com/hc/en-us/articles/32810166604183-GPU-Troubleshooting-Guide-Resolving-Driver-Library-Version-Mismatch-Errors)
28. How to resolve "Failed to initialize NVML: Driver/library version mismatch" error \- D2iQ, accessed September 28, 2025, [https://support.d2iq.com/hc/en-us/articles/4409480561300-How-to-resolve-Failed-to-initialize-NVML-Driver-library-version-mismatch-error](https://support.d2iq.com/hc/en-us/articles/4409480561300-How-to-resolve-Failed-to-initialize-NVML-Driver-library-version-mismatch-error)
29. Failed to initialize NVML: Driver/library version mismatch \- Reddit, accessed September 28, 2025, [https://www.reddit.com/r/freebsd/comments/18zhf55/failed_to_initialize_nvml_driverlibrary_version/](https://www.reddit.com/r/freebsd/comments/18zhf55/failed_to_initialize_nvml_driverlibrary_version/)
30. Nvidia NVML Driver/library version mismatch \[closed\] \- Stack Overflow, accessed September 28, 2025, [https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)
31. Hiccups setting up WSL2 \+ CUDA \- NVIDIA Developer Forums, accessed September 28, 2025, [https://forums.developer.nvidia.com/t/hiccups-setting-up-wsl2-cuda/128641](https://forums.developer.nvidia.com/t/hiccups-setting-up-wsl2-cuda/128641)
32. Installing the NVIDIA Container Toolkit, accessed September 28, 2025, [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
33. Install NeMo Framework \- NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
34. NVIDIA-NeMo/NeMo: A scalable generative AI framework built for researchers and developers working on Large Language Models, Multimodal, and Speech AI (Automatic Speech Recognition and Text-to-Speech) \- GitHub, accessed September 28, 2025, [https://github.com/NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)
35. Quickstart \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour)
36. Pipelines \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/main_classes/pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
37. The pipeline API \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers.js/v3.0.0/pipelines](https://huggingface.co/docs/transformers.js/v3.0.0/pipelines)
38. pipelines \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers.js/api/pipelines](https://huggingface.co/docs/transformers.js/api/pipelines)
39. Pipeline \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/pipeline_tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)
40. EnverLee/polyglot-ko-1.3b-Q4_0-GGUF \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/EnverLee/polyglot-ko-1.3b-Q4_0-GGUF](https://huggingface.co/EnverLee/polyglot-ko-1.3b-Q4_0-GGUF)
41. EleutherAI/polyglot-ko-1.3b at main \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main)
42. EleutherAI/polyglot-ko-1.3b · Hugging Face, accessed September 28, 2025, [https://huggingface.co/EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
43. How to run a Hugging Face text generation AI model Locally? step-by-step tutorial, accessed September 28, 2025, [https://www.youtube.com/watch?v=Ez_bHdET0iw](https://www.youtube.com/watch?v=Ez_bHdET0iw)
44. Generation \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
45. How to generate text: using different decoding methods for language generation with Transformers \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)
46. Temperature, top_p and top_k for chatbot responses \- OpenAI Developer Community, accessed September 28, 2025, [https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542)
47. Utilities for Generation \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/en/internal/generation_utils](https://huggingface.co/docs/transformers/en/internal/generation_utils)
48. Transformers \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
49. Using transformers at Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/hub/transformers](https://huggingface.co/docs/hub/transformers)
50. Hugging Face pitches HUGS as an alternative to Nvidia's NIM for open models \- Reddit, accessed September 28, 2025, [https://www.reddit.com/r/AMD_Stock/comments/1gct7mt/hugging_face_pitches_hugs_as_an_alternative_to/](https://www.reddit.com/r/AMD_Stock/comments/1gct7mt/hugging_face_pitches_hugs_as_an_alternative_to/)
51. Master Generative AI with NVIDIA NeMo, accessed September 28, 2025, [https://resources.nvidia.com/en-us-ai-large-language-models/watch-78](https://resources.nvidia.com/en-us-ai-large-language-models/watch-78)
52. NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support, accessed September 28, 2025, [https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/](https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/)
53. Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities, accessed September 28, 2025, [https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/](https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/)
54. NVIDIA NeMo Framework Developer Docs, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html)
55. Advantage of NEMO models & Toolkit \#7073 \- GitHub, accessed September 28, 2025, [https://github.com/NVIDIA/NeMo/discussions/7073](https://github.com/NVIDIA/NeMo/discussions/7073)
56. Configuring Nemo-Guardrails Your Way: An Alternative Method for Large Language Models, accessed September 28, 2025, [https://towardsdatascience.com/configuring-nemo-guardrails-your-way-an-alternative-method-for-large-language-models-c82aaff78f6e/](https://towardsdatascience.com/configuring-nemo-guardrails-your-way-an-alternative-method-for-large-language-models-c82aaff78f6e/)
57. Configure NeMo-Run — NVIDIA NeMo Framework User Guide \- NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html)
58. RunBot's math-to-text on NVIDIA NeMo Framework AutoModel \- LoRA \- vLLM Forums, accessed September 28, 2025, [https://discuss.vllm.ai/t/runbots-math-to-text-on-nvidia-nemo-framework-automodel/637](https://discuss.vllm.ai/t/runbots-math-to-text-on-nvidia-nemo-framework-automodel/637)
59. Community Model Converter User Guide — NVIDIA NeMo ..., accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/checkpoints/user_guide.html](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/checkpoints/user_guide.html)
60. Checkpoint Conversion — NVIDIA NeMo Framework User Guide 24.07 documentation, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/checkpointconversion.html](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/checkpointconversion.html)
61. Checkpoint conversion \- NeMo-Skills, accessed September 28, 2025, [https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion/](https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion/)
62. HFDatasetDataModule — NVIDIA NeMo Framework User Guide, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/automodel/codedocs/hf_dataset_data_module.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/automodel/codedocs/hf_dataset_data_module.html)
63. e9t/nsmc · Datasets at Hugging Face, accessed September 28, 2025, [https://huggingface.co/datasets/e9t/nsmc](https://huggingface.co/datasets/e9t/nsmc)
64. nglaura/koreascience-summarization · Datasets at Hugging Face, accessed September 28, 2025, [https://huggingface.co/datasets/nglaura/koreascience-summarization](https://huggingface.co/datasets/nglaura/koreascience-summarization)
65. gogamza/kobart-summarization \- Hugging Face, accessed September 28, 2025, [https://huggingface.co/gogamza/kobart-summarization](https://huggingface.co/gogamza/kobart-summarization)
