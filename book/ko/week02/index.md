# Week 2: PyTorch 2.x와 최신 딥러닝 프레임워크

## 1. PyTorch 2.x와 torch.compile: 컴파일러 혁명

PyTorch 2.x는 딥러닝 프레임워크의 패러다임을 바꾼 혁신적인 업데이트다. 기존의 **즉시 실행(Eager Mode)** 방식이 제공하는 유연성과 파이써닉한 개발 경험을 그대로 유지하면서, `torch.compile`이라는 단 한 줄의 코드로 모델의 실행 성능을 극대화하는 강력한 기능을 추가했다. 이는 "연구의 유연성과 프로덕션의 속도를 동시에 만족시키는" 혁신으로 평가받고 있다.

### 1.1 torch.compile의 작동 원리

`torch.compile`의 성능 향상은 **TorchDynamo**, **AOTAutograd**, **PrimTorch**, **TorchInductor**라는 네 가지 핵심 기술이 유기적으로 협력하여 이루어진다.

#### 1. 그래프 획득 (TorchDynamo)

- **역할**: Python 바이트코드를 분석하여 PyTorch 연산을 FX 그래프로 안전하게 캡처
- **핵심 기술**: "가드(guard)" 메커니즘으로 동적 Python 특성(조건문, 루프)을 완벽 지원
- **장점**: 코드 경로가 변경되면 해당 부분만 즉시 실행 모드로 처리하고 나머지는 컴파일된 코드 실행

#### 2. 사전 자동 미분 (AOTAutograd)

- **역할**: 순방향 그래프를 기반으로 미리 최적화된 역방향 그래프 생성
- **장점**: 전체 계산 그래프를 미리 분석하여 기울기 계산 과정을 최적화하고 메모리 사용량 감소

#### 3. 그래프 로워링 (PrimTorch)

- **역할**: 2,000개 이상의 PyTorch 연산자를 250개의 핵심 원시 연산자로 표준화
- **장점**: 다양한 하드웨어 백엔드(GPU, CPU, 커스텀 가속기)에 대한 호환성과 이식성 향상

#### 4. 그래프 컴파일 (TorchInductor)

- **역할**: 원시 연산자 그래프를 하드웨어 최적화된 기계 코드로 변환
- **핵심 기술**: GPU에서는 Triton 컴파일러로 고성능 CUDA 커널 동적 생성, CPU에서는 C++/OpenMP 사용

이러한 다단계 최적화로 `torch.compile`은 163개 모델 벤치마크에서 **평균 51%**의 훈련 속도 향상을 달성했다.

### 1.2 실습: torch.compile로 모델 추론 속도 향상

간단한 `nn.Module`에 `torch.compile`을 적용하여 **즉시 실행 모드**와 **컴파일 모드**의 성능을 직접 비교해보자. 컴파일은 첫 실행 시 그래프 캡처 및 코드 생성으로 인한 오버헤드가 발생하지만, 이후 반복적인 호출에서는 그 비용을 상쇄하고도 남을 월등히 빠른 속도를 보여준다.

```python
import torch
import torch.nn as nn
import time

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 간단한 신경망 모델 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNet().to(device)
dummy_input = torch.randn(128, 256).to(device)

# 1. 즉시 실행(Eager) 모드 성능 측정
# 워밍업: 초기 실행 시 발생할 수 있는 부하를 배제하기 위함
for _ in range(10):
    _ = model(dummy_input)

if device == "cuda": torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    _ = model(dummy_input)
if device == "cuda": torch.cuda.synchronize()
eager_duration = time.time() - start_time
print(f"Eager mode (100 runs): {eager_duration:.4f} seconds")

# 2. torch.compile 적용 (컴파일 모드)
# mode="reduce-overhead"는 프레임워크 오버헤드를 줄여 작은 모델 호출에 유리합니다.
compiled_model = torch.compile(model, mode="reduce-overhead")

# 워밍업 및 첫 컴파일 실행 (컴파일 오버헤드 발생)
for _ in range(10):
    _ = compiled_model(dummy_input)

if device == "cuda": torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    _ = compiled_model(dummy_input)
if device == "cuda": torch.cuda.synchronize()
compiled_duration = time.time() - start_time
print(f"Compiled mode (100 runs): {compiled_duration:.4f} seconds")

# 3. 성능 향상률 계산
speedup = eager_duration / compiled_duration
print(f"Speedup with torch.compile: {speedup:.2f}x")
```

**실행 결과 예시:**

```
Using device: cuda
Eager mode (100 runs): 0.0481 seconds
Compiled mode (100 runs): 0.0215 seconds
Speedup with torch.compile: 2.24x
```

### 체크포인트 질문

- `torch.compile`의 네 가지 핵심 기술(TorchDynamo, AOTAutograd, PrimTorch, TorchInductor)은 각각 어떤 역할을 하는가?
- 위 예제에서 `mode="reduce-overhead"` 옵션을 사용한 이유는 무엇인가? 작은 모델에서 기본 모드와 어떤 차이가 있을까?
- `torch.compile`이 기존 즉시 실행 모드와 비교했을 때 가지는 주요 장점과 한계는 무엇인가?

---

## 2. FlashAttention-3: 하드웨어 가속을 통한 어텐션 최적화

트랜스포머 아키텍처의 핵심이자 주된 성능 병목이었던 **어텐션 메커니즘**은 FlashAttention의 등장으로 획기적인 발전을 이루었다. 기존 어텐션은 시퀀스 길이 $N$에 대해 $O(N^2)$의 메모리와 계산 복잡도를 가져 긴 시퀀스 처리에 한계가 있었다. 이는 $N \times N$ 크기의 거대한 어텐션 스코어 행렬을 GPU의 HBM(고대역폭 메모리)에 저장하고 다시 읽어와야 했기 때문이다.

### 2.1 FlashAttention의 핵심 원리

**FlashAttention**은 메모리 I/O 병목을 해결하기 위해 **타일링(Tiling)** 기법과 GPU 내부의 매우 빠른 SRAM(정적 램)을 활용한다. 전체 행렬을 한 번에 계산하는 대신, 입력을 작은 블록(타일)으로 나누어 SRAM에서 어텐션 계산을 수행하고 중간 결과만 HBM에 저장함으로써 HBM과의 데이터 교환 횟수를 극적으로 줄인다.

### 2.2 FlashAttention-3의 하드웨어 가속

**FlashAttention-3**는 NVIDIA의 최신 Hopper 아키텍처(H100/H200 GPU 등)에 탑재된 하드웨어 가속 기능을 최대한 활용한다:

- **TMA (Tensor Memory Accelerator)**: HBM과 SRAM 간의 텐서 데이터 이동을 비동기적으로 가속하는 전용 하드웨어
- **WGMMA (Warpgroup Matrix Multiply-Accumulate)**: 행렬 곱셈-누산 연산을 더욱 효율적으로 처리하는 하드웨어 유닛
- **FP8 지원**: 8비트 부동소수점(FP8) 저정밀도 포맷을 지원하여, 정밀도 손실을 최소화하면서도 메모리 사용량과 처리량을 거의 두 배로 늘림

그 결과, FlashAttention-3는 H100 GPU에서 FlashAttention-2 대비 **1.5배에서 2.0배** 빠른 속도를 달성했다.

### 2.3 실습: Hugging Face Transformers에서 FlashAttention 활성화

Hugging Face의 🤗 Transformers 라이브러리는 FlashAttention을 긴밀하게 통합하여, **모델 로드 시 `attn_implementation` 인자 하나만으로** 간단하게 활성화할 수 있다. 이를 통해 기존 코드를 거의 변경하지 않으면서도 **상당한 추론 속도 및 메모리 효율성 개선**을 얻을 수 있다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU가 Hopper 아키텍처 이상이고, flash-attn 라이브러리가 설치되어 있다고 가정
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/gpt-oss-20b"  # FlashAttention-3를 지원하는 예시 모델 ID

# 1. 기본 어텐션 구현(Eager)으로 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_eager = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model with standard attention loaded.")

# 2. FlashAttention-3 구현으로 모델 로드
# 이 모델은 내부적으로 vLLM의 FlashAttention-3 커널을 사용 가능하며,
# 해당 커널은 'kernels' 패키지를 통해 허브에서 자동 다운로드됩니다.
try:
    model_flash = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3"  # FlashAttention-3 활성화
    )
    print("Model with FlashAttention-3 loaded successfully.")
    print("Note: This requires a compatible GPU (e.g., NVIDIA Hopper series).")
except ImportError:
    print("FlashAttention is not installed or the environment does not support it.")
except Exception as e:
    print(f"An error occurred while loading with FlashAttention: {e}")
```

**실행 결과 예시:**

```
Model with standard attention loaded.
Model with FlashAttention-3 loaded successfully.
Note: This requires a compatible GPU (e.g., NVIDIA Hopper series).
```

### 체크포인트 질문

- FlashAttention이 기존 어텐션 메커니즘의 어떤 문제를 해결하는가? 타일링 기법은 어떻게 작동하는가?
- FlashAttention-3에서 활용하는 NVIDIA Hopper 아키텍처의 주요 하드웨어 가속 기능들은 무엇인가?
- `attn_implementation="kernels-community/vllm-flash-attn3"` 옵션을 사용할 때 필요한 조건들과, 조건을 만족하지 않을 경우 어떤 현상이 발생하는가?

### 2.4 추가 실습: PyTorch scaled_dot_product_attention 직접 사용

PyTorch 2.0부터는 핵심 API에 `torch.nn.functional.scaled_dot_product_attention` (SDPA) 함수가 내장되었습니다. 이 함수는 백엔드에서 GPU 아키텍처, 입력 텐서의 속성(dtype, 마스크 존재 여부 등)을 자동으로 감지하여 가장 효율적인 어텐션 구현을 선택합니다. 즉, Ampere 아키텍처 이상의 GPU와 호환되는 조건에서는 **자동으로 FlashAttention 커널이나 메모리 효율적인(memory-efficient) 커널을 호출**합니다.

```python
import torch
import torch.nn.functional as F

# FlashAttention 경로를 사용하기 위해 GPU 및 half-precision(FP16/BF16) 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16  # 또는 torch.bfloat16

# 배치=1, 헤드=8, 시퀀스 길이=1024, 임베딩=64의 무작위 텐서 생성
q = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)
k = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)
v = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)

# 특정 커널만 사용하도록 강제하여 동작 확인 (디버깅/테스트용)
# enable_flash=True: FlashAttention 커널 사용 시도
# enable_math=False: 순수 PyTorch 수학 구현 비활성화
# enable_mem_efficient=False: 메모리 효율적 구현 비활성화
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    try:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        print("FlashAttention kernel was successfully used.")
        print("Output shape:", out.shape)
    except RuntimeError as e:
        print(f"Failed to use FlashAttention kernel exclusively: {e}")

```

위 코드는 `torch.backends.cuda.sdp_kernel` 컨텍스트 관리자를 사용하여 **FlashAttention 경로만 사용하도록 강제**합니다. 만약 GPU가 지원하고 입력 조건(half-precision, 마스크 없음 등)이 맞으면 내부적으로 FlashAttention 커널이 성공적으로 호출됩니다. 조건이 맞지 않으면 PyTorch는 "Flash attention kernel not used because..."와 같은 유용한 경고 메시지를 출력하며, 컨텍스트 관리자 외부에서는 안전하게 다른 구현으로 대체됩니다. 이를 통해 개발자는 의도한 최적화가 적용되고 있는지 쉽게 확인할 수 있습니다.

### 체크포인트 질문

- `scaled_dot_product_attention` 함수를 FP32로 실행하면 어떤 일이 발생하는가? FlashAttention이 사용되지 않는 이유는 무엇인가?
- `attn_mask` 인자를 넣으면 FlashAttention이 사용될까? FlashAttention이 지원하는 마스크 유형은 무엇인가?
- PyTorch에서 제공하는 경고 메시지를 통해 어떤 힌트를 얻을 수 있는가?

---

## 3. Hugging Face Transformers 생태계: 최신 동향과 실습

Hugging Face 🤗 Transformers 라이브러리는 단순한 **모델 저장소**를 넘어, 최신 AI 기술을 누구나 쉽게 접근하고 활용할 수 있도록 지원하는 거대한 통합 플랫폼으로 발전하고 있다.

### 3.1 최신 동향

- **최신 모델 아키텍처의 신속한 지원**: Vault-GEMMA, EmbeddingGemma와 같은 최신 LLM은 물론, Florence-2(통합 비전), SAM-2(고급 세그멘테이션) 등 **멀티모달 모델 지원**이 대폭 강화되었다.

- **고급 양자화(Quantization) 기술 통합**: OpenAI의 GPT-OSS 모델과 함께 소개된 **MXFP4**와 같은 **4비트 부동소수점 양자화** 방식을 네이티브로 지원한다. 이는 기존의 4비트 정수(INT4) 양자화보다 동적 범위 표현에 유리하여 정확도 손실을 줄이면서도, 120B 파라미터 모델을 단일 80GB GPU에 로드하는 등 메모리 사용량을 획기적으로 절감한다.

- **Zero-Build Kernels**: `kernels`라는 패키지를 통해 FlashAttention-3, Megablocks MoE 커널 등 **사전에 컴파일된 고성능 커널**을 허브에서 직접 다운로드하여 사용한다. 이는 사용자가 자신의 환경에서 소스 코드를 직접 컴파일하는 복잡하고 오류 발생 가능성이 높은 과정을 생략하게 해준다.

### 3.2 실습: 파이프라인 API로 한국어 감성 분석 수행

Hugging Face의 **`pipeline` API**는 토크나이징, 모델 추론, 후처리까지의 전 과정을 몇 줄의 코드로 추상화하여 제공하는 가장 간편하고 직관적인 도구다. 한국어 영화 리뷰 데이터(NSMC)로 파인튜닝된 모델을 사용하여 문장의 긍정/부정을 분석해보자.

```python
from transformers import pipeline

# 한국어 감성 분석을 위해 파인튜닝된 모델을 사용하는 파이프라인 생성
# 모델: WhitePeak/bert-base-cased-Korean-sentiment (NSMC 데이터셋 기반 fine-tuning)
classifier = pipeline(
    "sentiment-analysis",
    model="WhitePeak/bert-base-cased-Korean-sentiment"
)

# 분석할 문장들
reviews = [
    "이 영화는 제 인생 최고의 영화입니다. 배우들의 연기가 정말 인상 깊었어요.",
    "기대했던 것보다는 조금 아쉬웠어요. 스토리가 너무 평범했습니다.",
    "시간 가는 줄 모르고 봤네요. 강력 추천합니다!",
    "음악은 좋았지만 전체적으로 지루한 느낌을 지울 수 없었다."
]

# 감성 분석 실행
results = classifier(reviews)

# 결과 출력
for review, result in zip(reviews, results):
    label = "긍정" if result['label'] == 'LABEL_1' else "부정"
    score = result['score']
    print(f"리뷰: \"{review}\"")
    print(f"결과: {label} (신뢰도: {score:.4f})\n")
```

**실행 결과 예시:**

```
리뷰: "이 영화는 제 인생 최고의 영화입니다. 배우들의 연기가 정말 인상 깊었어요."
결과: 긍정 (신뢰도: 0.9985)

리뷰: "기대했던 것보다는 조금 아쉬웠어요. 스토리가 너무 평범했습니다."
결과: 부정 (신뢰도: 0.9978)

리뷰: "시간 가는 줄 모르고 봤네요. 강력 추천합니다!"
결과: 긍정 (신뢰도: 0.9982)

리뷰: "음악은 좋았지만 전체적으로 지루한 느낌을 지울 수 없었다."
결과: 부정 (신뢰도: 0.9969)
```

### 체크포인트 질문

- Hugging Face Transformers의 최신 동향 중 **Zero-Build Kernels**는 무엇이며, 어떤 장점을 제공하는가?
- 위 감성 분석 파이프라인에서 출력된 `LABEL_0`과 `LABEL_1`은 각각 어떤 의미를 가지는가?
- 다른 한국어 감성 분석 모델로 바꿔 실행했을 때 결과가 달라질 수 있는 이유는 무엇인가?

---

## 4. AI 에이전트 프레임워크: 자동화와 협업의 시대

LLM의 발전은 단일 모델이 하나의 작업을 수행하는 것을 넘어, **여러 도구를 자율적으로 사용**하고 **다른 에이전트와 협업하며 복잡한 목표를 달성**하는 **AI 에이전트** 패러다임을 열었다. 이를 체계적으로 지원하기 위해 다양한 프레임워크가 등장했으며, 각각 뚜렷한 철학과 강점을 가지고 있다.

### 4.1 주요 AI 에이전트 프레임워크 비교

| 프레임워크     | 핵심 철학                                           | 아키텍처 스타일                                 | 주요 사용 사례                                                                   |
| :------------- | :-------------------------------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------------- |
| **LangGraph**  | 명시적 제어 및 상태 기반 오케스트레이션             | 상태를 가진 유향 비순환 그래프 (DAG), 순환 허용 | 인간의 감독이 중요한 신뢰성 있고 감사 가능한 복잡한 다단계 에이전트 구축         |
| **CrewAI**     | 역할 기반 협업 지능                                 | 계층적, 역할 기반 **다중 에이전트 시스템**      | 역할 분담이 명확한 복잡한 비즈니스 워크플로우 자동화 (예: 시장 분석팀)           |
| **LlamaIndex** | 데이터 중심 에이전트 및 고급 RAG                    | 데이터 중심, 이벤트 기반 워크플로우             | 대규모 사설/비정형 데이터베이스에 대한 **질의응답 및 추론** 시스템 구축          |
| **Haystack**   | 프로덕션 레디 모듈형 파이프라인                     | 모듈형, 분기/루프가 가능한 **파이프라인**       | 확장 가능하고 견고한 프로덕션 등급의 **AI 애플리케이션** 구축                    |
| **DSPy**       | 선언적 LM 프로그래밍 ("프롬프트가 아닌 프로그래밍") | 선언적, 최적화 가능 **파이프라인**              | 수동 프롬프트 튜닝을 **데이터 기반 최적화**로 대체하여 최고 성능을 요구하는 작업 |

위 표에서 보듯, **LangGraph**는 에이전트의 상태와 제어 흐름을 명시적으로 관리하며 **장기 실행 및 신뢰성**에 초점을 맞추고, **CrewAI**는 각기 다른 전문성을 가진 에이전트들의 **협업**에 중점을 둔다. **LlamaIndex**는 방대한 지식베이스와 결합된 데이터 중심 에이전트를 지향하며, **Haystack**은 검색-추론 등 **모듈 조합형 파이프라인**의 실전 적용에 강하다. 마지막으로 **DSPy**는 프롬프트 엔지니어링 자체를 추상화하여 **선언적 프로그래밍** 스타일로 LLM 활용을 고도화한다. 각 프레임워크의 철학과 구조를 이해하면, 해결하려는 문제의 성격에 따라 가장 적절한 도구를 선택할 수 있다.

### 4.2 DSPy: 선언적 프롬프트 프로그래밍

**DSPy**는 *Declarative Self-Improving Python*의 약자로, Databricks에서 출시한 **선언형 프롬프트 프로그래밍** 프레임워크다. LLM을 직접 다루면서 발생하는 **긴 프롬프트 문자열 관리**의 복잡함을 줄이고, 마치 **코드를 작성하듯 모듈화된 구성**으로 AI 프로그램을 만들 수 있게 해준다. 한마디로 "프롬프트를 하드코딩하지 말고, **프로그래밍처럼** 작성하라"는 철학으로 설계되었다.

DSPy의 핵심 개념은 **LM**, **Signature**, **Module** 세 가지로 나뉜다:

- **LM**: 사용할 **언어 모델**을 지정한다. 예를 들어 OpenAI API의 GPT-4, HuggingFace의 Llama2 등 원하는 모델을 `dspy.LM(...)`으로 설정하고 `dspy.configure(lm=...)` 하면, 이후 모든 모듈이 이 LM을 통해 결과를 생성한다.

- **Signature**: 함수의 입력과 출력 타입을 지정하듯, 프롬프트 프로그램의 **입력과 출력 형식**을 선언한다. 예를 들어 `"question -> answer: int"`처럼 signature를 정의하면, DSPy는 `question`(str)을 받아 `answer`(int)를 내는 구조로 프롬프트를 자동 생성한다. 시그니처는 모델에게 주어질 프롬프트의 구조와 기대 출력 형태(예: JSON 또는 정수 등)를 기술하는 역할을 한다.

- **Module**: 문제를 풀기 위한 **프롬프트 기법** 자체를 모듈로 캡슐화한다. 예를 들어 단순 질의응답은 `dspy.Predict`, 복잡한 사고가 필요한 경우 `dspy.ChainOfThought`(연쇄적 사고), 툴 사용 에이전트는 `dspy.ReAct` 모듈로 표현할 수 있다. 각 모듈은 내부적으로 해당 기법에 맞게 **프롬프트를 어떻게 구성할지** 로직이 구현되어 있다.

사용자는 이 세 가지를 조합하여 **AI 프로그램**을 만든 뒤, DSPy에 내장된 **Optimizer**를 통해 모듈의 프롬프트를 자동으로 개선하거나 few-shot 예시를 추가하는 등의 최적화를 할 수 있다. 예를 들어, 아래처럼 간단한 조합을 만들어볼 수 있다:

```python
!pip install dspy # 필요한 라이브러리 설치 (Databricks DSPy)
import dspy

# 1) LM 설정 (예: 로컬 Llama2 모델 API 사용)
llm = dspy.LM('ollama/llama2', api_base='http://localhost:11434') # 로컬 서버 예시
dspy.configure(lm=llm)

# 2) Signature 선언: "질문(str) -> 답변(int)" 형식
simple_sig = "question -> answer: int"

# 3) Module 선택: Predict (기본적인 1단계 질의응답)
simple_model = dspy.Predict(simple_sig)

# 4) 실행
result = simple_model(question="서울에서 부산까지 KTX로 몇 시간 걸리나요?")
print(result)
```

위 코드는 `simple_model`이라는 모듈을 만들어 \*"질문을 받으면 정수 형태의 답변을 출력"\*하는 작업을 정의합니다. 내부적으로 DSPy는 이 요구에 맞는 최적의 프롬프트를 생성하여 LM에 전달합니다. (예를 들어, "Q: 서울에서 부산까지 KTX로 몇 시간 걸리나요?\\nA:" 형태로 프롬프트를 구성하고 숫자 형태 답변을 기대하는 식입니다.) 초기에 얻은 답이 부정확하다면, **BootstrapFewShot**과 같은 Optimizer를 적용해 few-shot 예시를 자동으로 첨가하거나, **Refine** 모듈로 답변을 지속 개선하도록 지시할 수도 있습니다. 이런 방식으로 DSPy는 복잡한 LLM 파이프라인(예: RAG 시스템, 다단계 체인, 에이전트 루프 등)도 모듈 단위로 구성하고 최적화할 수 있게 해줍니다.

DSPy의 장점은 **프롬프트 엔지니어링의 생산성 향상**입니다. 코드처럼 구조화된 틀 안에서 LLM 호출을 설계하므로, 사람이 긴 프롬프트 문장을 일일이 작성하며 시행착오를 겪는 시간을 줄여줍니다. 또한 여러 **모델/기법을 교체**하면서도 동일한 모듈 인터페이스를 유지할 수 있어, 예컨대 동일한 Chain-of-Thought 모듈을 GPT-4와 Llama2에 모두 적용해 성능을 비교하는 등 **유연한 실험**이 가능합니다. 선언적인 접근 덕분에 **프로그램의 일부만 변경**해도 전체 LLM 파이프라인에 쉽게 반영되므로, 유지보수도 용이합니다. 아직 초기 단계의 프레임워크이지만, \*\*"프로그래밍하듯 LLM을 다룬다"\*\*는 패러다임을 제시했다는 점에서 주목받고 있습니다.

**연습 문제:** 위 DSPy 예시에서 Signature를 "question -\> answer: int"로 정의하였는데, 만약 "question -\> answer: JSON"처럼 JSON 형식 출력을 요구하도록 바꾸면 결과는 어떻게 달라질까요? 그리고 dspy.Predict 외에 복잡한 문제에 대해 dspy.ChainOfThought나 dspy.ReAct를 사용하면 어떤 차이가 있을지 간단히 설명해보세요.

### 4.2 Haystack: 문서 기반 검색과 추론

**Haystack**은 독일 Deepset에서 개발한 **오픈소스 NLP 프레임워크**로, 주로 **지식 기반 질의응답**(Question Answering) 시스템 구축에 사용됩니다. Haystack의 강점은 **유연한 파이프라인 구성**에 있습니다. 사용자는 데이터베이스(문서 저장소)부터 검색기(Retriever), 리더(Reader)나 생성기(Generator) 모델까지 일련의 단계를 하나의 Pipeline으로 엮어, 질문을 넣으면 답변을 반환하는 **엔드투엔드 NLP 시스템**을 쉽게 만들 수 있습니다. 예를 들어 "주어진 문서 집합에서 질문의 답을 찾아라"와 같은 **Retrieval QA**나, 위키피디아 기반 **챗봇** 등을 Haystack으로 구현할 수 있습니다.

Haystack의 주요 컴포넌트는 아래와 같습니다:

- **DocumentStore**: 말 그대로 문서를 저장하는 **DB**입니다. In-Memory 형태나 Elasticsearch, FAISS 등의 백엔드를 지원하며, 문서의 원본 텍스트와 메타데이터, 임베딩 등을 보관합니다.

- **Retriever**: 사용자의 질문(Query)에 대해 관련 문서를 **검색**하는 역할을 합니다. BM25 같은 전통적 키워드 기반부터 SBERT, DPR 등 **Dense Passage Retrieval** 모델까지 다양하게 구현되어 있습니다. Retriever는 DocumentStore에서 **상위 k개**의 관련 문서를 찾아냅니다.

- **Reader** 또는 **Generator**: 검색된 문서들을 입력으로 받아 최종 **답을 생성**합니다. **Reader**는 보통 Extractive QA 모델(BERT 등)을 사용하여 해당 문서 내에서 **정답 스팬**을 뽑아주고, **Generator**는 GPT-같은 생성형 모델을 이용해 답을 생성할 수도 있습니다. 둘 다 Haystack에서 \*\*노드(Node)\*\*로써 플러그인 가능하며, 동시에 사용할 수도 있습니다.

- **Pipeline**: 상기 요소들을 조합하여 **질의 -\> 응답 플로우**를 정의하는 구조입니다. 간단하게는 Retriever 결과를 Reader에 넣는 **ExtractiveQAPipeline**이 있고, 생성형으로 답을 만드는 **GenerativeQAPipeline**도 있습니다. 또 Retrieval-Augmented Generation처럼 **Retriever + LLM**을 연결하거나, 여러 단계의 **조건부 흐름**(분기/루프)을 구현할 수도 있어 매우 유연합니다.

Haystack을 이용한 **간단 실습 예시**를 들어보겠습니다. 가령 FAQ 문서 모음을 이용해 질문에 답변하는 QA 시스템을 만든다고 하면:

```python
!pip install farm-haystack[faiss] # Haystack 설치 및 FAISS 등 옵션 (필요시)

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack import Pipeline

# 1) 문서 저장소 생성 및 문서 삽입
document_store = InMemoryDocumentStore()
docs = [{"content": "드라마 **오징어 게임**은 한국의 서바이벌 드라마...", "meta": {"source": "위키피디아"}}]
document_store.write_documents(docs)

# 2) Retriever와 Reader 구성
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="monologg/koelectra-base-v3-finetuned-korquad", use_gpu=False)

# 3) 파이프라인 구축
pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

# 4) QA 실행
query = "오징어 게임 감독이 누구야?"
result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
print(result['answers'][0].answer)
```

위 코드에서는 간단히 **인메모리 문서저장소**에 하나의 문서를 넣고, BM25 기반 **Retriever**와 한국어 KorQuAD 데이터로 학습된 Electra **Reader**를 조합한 파이프라인을 구축했습니다. pipeline.run()에 질의를 넣으면 Retriever가 상위 5개 문서를 찾고, Reader가 그 중에서 답을 추출하여 반환합니다. 예를 들어 위 질문에 대해 "황동혁"이라는 정답을 얻을 수 있을 것입니다.

Haystack의 강력한 점은 이처럼 **구성 요소를 교체하거나 확장하기 용이**하다는 점입니다. Dense Retriever로 바꾸거나, Reader 대신 GPT-3 같은 생성 모델을 Generator로 붙이는 것도 가능합니다. 또 멀티홉 QA처럼 중간에 여러 노드를 순차/병렬 구성하여 복잡한 추론 시나리오를 지원할 수도 있습니다.

실제 산업 현장에서는 Haystack을 활용해 **도메인 문서 검색 + QA** 서비스나, **챗봇**에 외부 지식을 주입하는 **RAG** 파이프라인을 구성하는 사례가 많습니다. 요약하면, Haystack은 **검색 엔진과 NLP 모델을 하나로 엮는 프레임워크**로, 비교적 적은 코드로 강력한 **문서 기반 QA 시스템**을 구축할 수 있게 해주는 도구입니다.

**연습 문제:** 위 Haystack 예시에서 BM25Retriever 대신 **Dense Retriever**(예: DensePassageRetriever)를 사용하려면 어떤 변화가 필요할까요? 또 Reader를 대신하여 GPT 계열 생성 모델을 Generator로 붙이면 어떤 장단점이 있을지 생각해보세요. 마지막으로, `pipeline.run()` 호출 시 `params`에 전달한 `top_k` 값들을 조절하면 결과가 어떻게 달라질지 실험적으로 설명해보세요.

### 4.3 CrewAI: 역할 기반 멀티에이전트 프레임워크

**CrewAI**는 최근 각광받는 AI 에이전트 프레임워크 중 하나로, 여러 개의 LLM 에이전트를 **팀(crew)** 형태로 구성하여 **협업적으로 작업**을 수행하도록 하는 플랫폼입니다. 기존의 LangChain 등이 단일 에이전트 또는 체인 중심이었다면, CrewAI는 **역할 기반 멀티에이전트**에 특화되어 있습니다. 예를 들어 하나의 문제를 해결하기 위해 **Researcher**, **Analyst**, **Writer** 등 역할을 나누고, 각 에이전트가 자신만의 도구(tool)와 목표(goal)를 가지고 자율적으로 행동하면서 전체적으로는 협력해 최종 결과를 산출하도록 구성할 수 있습니다.

CrewAI의 개념을 주요 구성 요소별로 정리하면 다음과 같습니다:

- **Crew (승무원 팀)**: 전체 에이전트들의 조직 혹은 환경입니다. `Crew` 객체가 여러 에이전트를 포함하며, 이들의 **협업 프로세스**를 총괄합니다. 하나의 Crew는 특정 목표를 달성하기 위한 에이전트 팀에 대응합니다.

- **Agent (에이전트)**: 독립적인 **자율 AI**로, 각각 정해진 \*\*역할(role)\*\*과 **도구(tools)** 및 **목표**를 가집니다. 예를 들어 "문헌 조사원" 에이전트는 웹 검색 도구를 사용해 정보를 수집하고, "보고서 작성자" 에이전트는 글쓰기 도구와 문체에 맞춰 최종 보고서를 작성하는 식입니다. 에이전트는 필요 시 다른 에이전트에게 작업을 위임하거나 결과를 요청할 수도 있습니다 (마치 사람이 팀 협업하듯).

- **Process (프로세스)**: Crew 내에서 에이전트들의 **상호작용 규칙**이나 **워크플로우**를 정의한 것입니다. 예를 들어 "1단계: Researcher가 자료 수집 -\> 2단계: Analyst가 요약 -\> 3단계: Writer가 정리" 와 같은 흐름을 프로세스로 설정할 수 있습니다. CrewAI에서는 이러한 프로세스를 **Flow**라는 개념으로 확장하며, 이벤트나 조건에 따라 에이전트 실행을 제어할 수도 있습니다.

CrewAI를 사용하면 개발자는 각 에이전트의 역할과 사용 도구를 정의하고, `Crew`를 생성해 실행함으로써 **복잡한 작업을 자동화**할 수 있습니다. 간단한 사용 예를 들어보겠습니다. 가령 \*"주어진 주제에 대해 자료를 찾아 요약한 보고서를 작성하라"\*는 복잡한 과제를 두 에이전트에게 협업시키는 경우:

```python
!pip install crewai # CrewAI 프레임워크 설치 (가상 가정)
from crewai import Crew, Agent, tool

# 에이전트 정의: 검색 담당자와 작성 담당자
searcher = Agent(name="Researcher", role="정보 수집", tools=[tool("wiki_browser")])
writer = Agent(name="Writer", role="보고서 작성", tools=[tool("text_editor")])

# Crew 생성 및 에이전트 추가
crew = Crew(agents=[searcher, writer], goal="주어진 주제에 대한 1페이지 요약 보고서 작성")
crew.run(task="한국의 전통 음식에 대해 조사하고 요약하라.")
```

위 예시는 가상의 코드이지만, `Agent`에게 역할과 사용할 툴(예: **위키 브라우저**, **텍스트 에디터** 기능)을 부여하고 `Crew`에 등록한 후 실행하는 흐름을 묘사합니다. 실행 시 **Researcher** 에이전트는 먼저 위키피디아를 검색해 정보를 모으고, 그 결과를 **Writer** 에이전트에게 전달합니다. Writer는 받은 정보를 정리하여 요약 보고서를 작성한 뒤 최종 답을 산출합니다. 이 모든 과정이 사람 개입 없이 자동으로 이루어지며, CrewAI 프레임워크가 **각 단계의 수행과 에이전트 간 메시지 교환**을 관리해 줍니다.

CrewAI의 특징은 **높은 유연성과 통제력**입니다. 단순히 여러 에이전트를 독립적으로 돌리는 것이 아니라, 개발자가 원하는 대로 **협업 패턴**을 디자인할 수 있습니다. 또한 개별 에이전트에 대해 프롬프트 규칙, 응답 형식 등을 세밀히 설정 가능하여, 팀 내 **전문 AI**들을 구축할 수 있습니다. 실제로 **자동화된 고객지원** 시나리오(예: 한 에이전트가 유저 의도를 파악 -\> 다른 에이전트가 FAQ 검색 -\> 또 다른 에이전트가 답변 생성)나 **연구 어시스턴트**(역할 분담하여 문헌 정리) 등에 응용될 수 있습니다.

흥미로운 점은 CrewAI가 완전히 새로운 프레임워크라기보다 **LangChain 등 기존 툴과 호환**되도록 설계되었다는 점입니다. 즉, LangChain에서 쓰던 도구 체인을 가져와 CrewAI 에이전트의 tool로 활용할 수 있습니다. 다만 멀티에이전트 시스템 특성상 예상치 못한 상호작용이나 **무한 루프** 등을 방지하기 위한 **안전장치 설계**도 중요합니다. CrewAI 측에서는 역할별 **제한과 정책**을 설정하여 에이전트들이 정해진 범위 내에서만 활동하도록 권고하고 있습니다.

요약하면, CrewAI는 **역할 기반 자율 에이전트들의 협업을 체계화**한 프레임워크로서, 하나의 거대 LLM이 모든 걸 하는 대신 여러 전문 LLM이 **분업과 협력을 통해 더 복잡한 작업을 수행**하도록 돕습니다. 이를 통해 멀티에이전트 AI 시스템 개발을 쉽고 표준화된 방식으로 접근할 수 있게 해줍니다.

### 체크포인트 질문

- CrewAI의 핵심 구성 요소인 **Crew**, **Agent**, **Process**는 각각 어떤 역할을 하는가?
- CrewAI를 활용하면 어떤 종류의 AI 작업에서 장점을 발휘할 수 있는가? 전문 분야 문서 작성이나 고객 지원 시나리오에서의 활용 방안을 생각해보라.
- 멀티에이전트가 함께 작업할 때 발생할 수 있는 문제점(예: 무한 루프, 충돌)을 방지하려면 어떤 설계가 필요한가?

### 4.3 LangGraph: 상태 기반 멀티에이전트 오케스트레이션

**LangGraph**는 LangChain 팀이 개발한 **저수준 오케스트레이션 프레임워크**로, **장기간 유지되는 상태(state)**를 가진 멀티에이전트 시스템 구축에 특화되어 있습니다. LangGraph는 에이전트 실행을 **그래프 자료구조**로 관리하며, 각 노드가 에이전트의 상태와 동작을 나타내고 엣지로 상호작용 경로를 표현합니다. 이를 통해 에이전트 간 메시지 흐름, 상태 변경, 오류 발생 시의 **복구 지점(checkpoint)** 등을 명시적으로 다룰 수 있어 **신뢰성**과 **내구성**이 요구되는 시나리오에 적합합니다.

LangGraph 사용의 핵심은 *StateGraph*라는 그래프 객체를 정의하고, 필요한 경우 **체크포인트 저장소**와 연계해 에이전트들의 상태를 지속적으로 저장/복구할 수 있다는 점입니다. 예를 들어 **긴 대화**나 **플랜 실행** 도중 한 에이전트가 실패해도, 마지막 저장된 state로 롤백하여 재시도하는 식의 **내결함성(fault-tolerance)** 있는 설계를 할 수 있습니다. 또한 **Human-in-the-loop** 개입이 용이하여, 중간 상태에서 사람이 검토하거나 수정한 후 다시 이어서 실행시키는 것도 가능합니다.

간단한 LangGraph 예제를 통해 개념을 살펴보겠습니다. 아래 코드에서는 하나의 툴(tool)을 가진 React 스타일 에이전트를 생성하고, 그래프 상에서 사용자의 질문을 처리합니다 (Anthropic Claude 모델을 가정):

```python
!pip install langgraph # LangGraph 라이브러리 설치
from langgraph.prebuilt import create_react_agent

# 간단한 툴 함수 정의
def get_weather(city: str) -> str:
    # 실제 외부 API 대신 고정된 답변을 반환 (예시)
    return f"{city}의 날씨는 항상 맑습니다!"

# React 에이전트 생성 (Anthropic Claude API 활용 가정)
agent = create_react_agent(
    model="anthropic:claude-2",    # Anthropic Claude 모델 (API 키 필요)
    tools=[get_weather],
    prompt="You are a helpful assistant."  # 기본 프롬프트
)

# 에이전트 실행 (대화 형식의 입력 전달)
response = agent.invoke({"messages": [{"role": "user", "content": "서울의 날씨가 궁금해"}]})
print(response)
```

위 코드에서 `create_react_agent` 함수를 통해 **ReAct 패턴**의 에이전트를 생성했다. `tools` 리스트에 `get_weather` 함수를 제공하여, 에이전트가 필요 시 해당 툴을 사용할 수 있도록 했다. 마지막에 `agent.invoke(...)`를 호출할 때 사용자의 메시지를 그래프 입력으로 주면, LangGraph는 내부적으로 \*\*상태 그래프(StateGraph)\*\*를 구성하여 에이전트의 reasoning 과정을 추적한다. `response`에는 에이전트가 최종적으로 산출한 답변이 들어간다 (예: "서울의 날씨는 항상 맑습니다\!").

LangGraph를 활용하면 이러한 **에이전트 워크플로우**를 더욱 복잡하게 확장할 수 있다. 여러 에이전트를 노드로 추가하고, 노드 간 **메시지 전달 경로**를 정의하여, 예컨대 "질문 분석 -\> 정보 검색 -\> 답 정리" 같은 과정을 각기 다른 에이전트가 그래프 순서에 따라 수행하게 설정할 수 있다. LangGraph의 **체크포인트** 기능을 사용하면 장시간 실행되는 에이전트의 중간 상태를 주기적으로 저장하여, 예기치 않은 오류 발생 시 처음부터가 아닌 중간부터 재개할 수도 있다. 또한 **LangSmith** 등의 모니터링 도구와 통합해 그래프 실행을 시각화하고 디버깅할 수 있다.

정리하면, LangGraph는 멀티에이전트 시스템 개발에서 **신뢰성과 지속성**을 확보하기 위한 프레임워크다. 이는 웹 서비스나 업무 자동화 등에서 **오랜 시간 동안 중단 없이 동작해야 하는 에이전트** 팀을 구축할 때 유용하다. LangChain 생태계와 연동되므로, 기존 LangChain 사용자라면 친숙하게 상태 기반 접근을 도입할 수 있다는 장점도 있다.

### 체크포인트 질문

- LangGraph의 핵심 특징인 **상태 기반 오케스트레이션**은 무엇이며, 어떤 장점을 제공하는가?
- LangGraph의 **체크포인트**와 **Human-in-the-loop** 기능이 긴 프로세스나 장기간 동작하는 에이전트에서 어떻게 활용될 수 있는가?
- LangGraph로 구축된 에이전트 시스템에서 에이전트 간 무한 대화나 충돌을 방지하기 위해 어떤 설계가 필요한가?

---

## 5. 실습: BERT vs Mamba 모델 비교 실험

공개된 한국어(NSMC)용 Mamba 분류 모델 부재로, **IMDB 영어 데이터셋**으로 비교 실험을 구성했다. Mamba는 사용자가 제공한 공개 체크포인트 `trinhxuankhai/mamba_text_classification`를, BERT는 공개 IMDB 분류 베이스라인을 사용하여 **정확도·추론 속도·GPU 메모리**를 비교한다.

### 5.1 환경 준비

```bash
# GPU 권장. Colab/쿠다 환경 권장
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers datasets accelerate
```

### 5.2 데이터셋 로드 (IMDB)

```python
from datasets import load_dataset

imdb = load_dataset("imdb")
imdb_test = imdb["test"]  # 25k samples

# 속도 비교를 위해 소샘플(예: 1000개)만 평가해도 무방
imdb_test_small = imdb_test.select(range(1000))
```

### 5.3 모델 및 토크나이저 로드

- **Mamba**: `trinhxuankhai/mamba_text_classification` (사용자 제공 카드)
- **BERT**: 공개된 IMDB 분류 모델 (예: `textattack/bert-base-uncased-imdb`)

<!-- end list -->

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

# 1) Mamba 분류 모델 (IMDB 학습)
mamba_id = "trinhxuankhai/mamba_text_classification"
tok_mamba = AutoTokenizer.from_pretrained(mamba_id, use_fast=True)
model_mamba = AutoModelForSequenceClassification.from_pretrained(mamba_id).to(device)
model_mamba.eval()

# 2) BERT 분류 모델 (공개 IMDB 베이스라인)
bert_id = "textattack/bert-base-uncased-imdb"
tok_bert = AutoTokenizer.from_pretrained(bert_id, use_fast=True)
model_bert = AutoModelForSequenceClassification.from_pretrained(bert_id).to(device)
model_bert.eval()

print("Loaded:", mamba_id, "|", bert_id)
```

### 5.4 평가 함수 (정확도·속도·메모리)

```python
import time, numpy as np

@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size=16, max_length=256, warmup=2):
    # 전처리
    def enc(batch):
        encodings = tokenizer(
            batch["text"], truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        encodings["labels"] = torch.tensor(batch["label"])
        return encodings

    # 미리 텐서화
    texts = dataset["text"]
    labels = dataset["label"]

    # 배치 단위 인코딩 (메모리 절약을 위해 on-the-fly도 가능)
    encoded = [enc({"text": [t], "label": [l]}) for t, l in zip(texts, labels)]

    # 워밍업
    for _ in range(warmup):
        for i in range(0, len(encoded), batch_size):
            batch = {k: torch.cat([encoded[j][k] for j in range(i, min(i+batch_size, len(encoded)))], dim=0).to(device)
                     for k in encoded[0].keys()}
            _ = model(**batch)

    # 메모리/시간 측정
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    preds = []
    for i in range(0, len(encoded), batch_size):
        batch = {k: torch.cat([encoded[j][k] for j in range(i, min(i+batch_size, len(encoded)))], dim=0).to(device)
                 for k in encoded[0].keys()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    if device == "cuda":
        torch.cuda.synchronize()

    duration = time.time() - start
    acc = (np.array(preds) == np.array(labels)).mean().item()
    throughput = len(dataset) / duration
    peak_mem_mb = None

    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {"accuracy": acc, "sec": duration, "throughput": throughput, "peak_mem_mb": peak_mem_mb}

# 평가 실행 (1k 샘플로)
res_mamba = evaluate(model_mamba, tok_mamba, imdb_test_small, batch_size=32, max_length=256)
res_bert  = evaluate(model_bert,  tok_bert,  imdb_test_small, batch_size=32, max_length=256)

print("Mamba results:", res_mamba)
print("BERT results:", res_bert)
```

### 5.5 결과 예시와 해석

**예시(환경에 따라 달라질 수 있음):**

- **Mamba(IMDB)** → `{'accuracy': 0.94, 'throughput': X samples/sec, 'peak_mem_mb': Y MB}`

- **BERT(IMDB)** → `{'accuracy': 0.94, 'throughput': Z samples/sec, 'peak_mem_mb': W MB}`

- **정확도(Accuracy)**: 제공된 Mamba 모델 카드 기준 Val/Test Acc ≈ 0.94로, **BERT 베이스라인과 유사**한 높은 성능을 보입니다.

- **추론 속도(Throughput)**: 입력 길이 256, 배치 32의 조건에서는 양자화 적용 여부나 하드웨어 스펙에 따라 편차가 있습니다. 입력 길이가 길어질수록 어텐션의 $O(N^2)$ 복잡도로 인해 BERT의 속도가 급격히 저하되는 반면, **Mamba의 선형 시간($O(N)$) 장점**이 두드러질 가능성이 큽니다.

- **메모리 사용(Peak Memory)**: Mamba는 상태 공간 모델(SSM) 특성상 거대한 어텐션 행렬을 생성하지 않으므로 **이론적으로 메모리 효율이 더 높습니다**. 긴 시퀀스에서 이 차이는 더욱 명확해집니다.

**주의**

- 위 코드는 **1000개 샘플**로 속도/메모리를 비교합니다(전체 25k도 가능하나, 실습 시간 고려).
- 실험의 공정성을 위해 \*\*동일한 `max_length`, `batch_size`, `dtype`\*\*을 유지해야 합니다.
- Colab T4/V100, A100, RTX 30/40 계열 등 **GPU 스펙에 따라 결과가 크게 달라질 수 있습니다**.

### 체크포인트 질문

1.  이번 비교에서 **정확도가 유사**했음에도, **긴 입력 길이**에서 두 모델의 **추론 속도/메모리**는 어떻게 달라질지 예측해보세요.
2.  실서비스 도입 시 Mamba의 **장점과 리스크**(예: 라이브러리 성숙도, 생태계, 디버깅 툴)를 각각 2가지씩 적어보세요.
3.  동일 코드로 \*\*`max_length=512/1024`\*\*로 변경하여 재측정해보고, Throughput/PeakMem 변화를 보고서로 요약해보세요.

---

### 부록: 재현 가능한 전체 스니펫

아래 셀만 순서대로 실행하면 Colab/GPU에서 **한 번에 비교** 가능합니다.

```python
# 0) 설치
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers datasets accelerate

# 1) 데이터
from datasets import load_dataset
imdb = load_dataset("imdb")
imdb_test_small = imdb["test"].select(range(1000))

# 2) 모델/토크나이저
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = "cuda" if torch.cuda.is_available() else "cpu"
mamba_id = "trinhxuankhai/mamba_text_classification"
bert_id  = "textattack/bert-base-uncased-imdb"

tok_mamba = AutoTokenizer.from_pretrained(mamba_id, use_fast=True)
tok_bert  = AutoTokenizer.from_pretrained(bert_id,  use_fast=True)

model_mamba = AutoModelForSequenceClassification.from_pretrained(mamba_id).to(device).eval()
model_bert  = AutoModelForSequenceClassification.from_pretrained(bert_id).to(device).eval()

# 3) 평가
import time, numpy as np

@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size=32, max_length=256, warmup=1):
    def enc(batch):
        encodings = tokenizer(batch["text"], truncation=True, padding="max_length",
                              max_length=max_length, return_tensors="pt")
        encodings["labels"] = torch.tensor(batch["label"])
        return encodings

    texts, labels = dataset["text"], dataset["label"]
    encoded = [enc({"text":[t], "label":[l]}) for t,l in zip(texts, labels)]

    for _ in range(warmup):
        for i in range(0, len(encoded), batch_size):
            batch = {k: torch.cat([encoded[j][k] for j in range(i, min(i+batch_size, len(encoded)))], dim=0).to(device)
                     for k in encoded[0].keys()}
            _ = model(**batch)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    start = time.time()
    preds = []
    for i in range(0, len(encoded), batch_size):
        batch = {k: torch.cat([encoded[j][k] for j in range(i, min(i+batch_size, len(encoded)))], dim=0).to(device)
                 for k in encoded[0].keys()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    if device == "cuda": torch.cuda.synchronize()
    duration = time.time() - start
    acc = (np.array(preds) == np.array(labels)).mean().item()
    thr = len(dataset) / duration
    peak = torch.cuda.max_memory_allocated()/(1024**2) if device=="cuda" else None
    return {"accuracy": acc, "sec": duration, "throughput": thr, "peak_mem_mb": peak}

res_mamba = evaluate(model_mamba, tok_mamba, imdb_test_small)
res_bert  = evaluate(model_bert,  tok_bert,  imdb_test_small)

print("Mamba:", res_mamba)
print("BERT :", res_bert)
```

---

## 6. 실험 정리 및 시사점

**BERT vs Mamba 비교 실험**을 통해 두 아키텍처의 특성과 장단점을 실증적으로 살펴보았다. 이 실험은 단순히 두 모델의 우열을 가리는 것을 넘어, 시퀀스 모델링의 현재와 미래를 조망하는 중요한 시사점을 제공한다.

#### 어떤 모델이 어떤 상황에 적합한가?

- **짧은 시퀀스 및 검증된 성능: BERT (Transformer)**

  - **입력 시퀀스 길이**가 수백 토큰 내외의 짧은 작업(단문 분류, 개체명 인식, 단답형 QA 등)에서는 BERT와 같은 Transformer 아키텍처가 여전히 강력하고 효율적이다.
  - 방대한 사전학습 모델과 성숙한 파인튜닝 생태계 덕분에 높은 정확도를 안정적으로 달성하기 쉽고, 짧은 입력에서는 추론 지연 시간도 경쟁력이 있다.

- **초장문맥(Long-Context) 처리 및 효율성: Mamba (SSM)**

  - 수천에서 수만 토큰에 이르는 긴 문서 단위의 작업(장문서 요약, 소설 감정 분석, 코드 생성 등)에서 Mamba의 진가가 드러난다. Transformer의 $O(N^2)$ 복잡도는 이러한 작업에서 계산량과 메모리 사용량을 폭발적으로 증가시켜 사실상 처리가 불가능하지만, Mamba의 $O(N)$ 선형 복잡도는 시퀀스 길이에 따른 성능 저하 없이 효율적인 처리를 가능하게 한다.
  - Mamba는 **최대 100만 토큰**까지 처리 가능함을 시연하며, 초장문맥 LLM 시대를 열 핵심 기술로 주목받고 있다.

#### 서비스/프로덕션 적용 시사점

- **현재의 안정성 vs. 미래의 가능성**

  - **현재 프로덕션 환경**에서는 Transformer 계열(BERT, GPT 등) 모델이 성능, 안정성, 개발 도구 지원 측면에서 월등히 성숙하여 널리 쓰이고 있다.
  - Mamba는 매우 유망한 기술이지만, 아직 **라이브러리 지원, 커뮤니티, 사전학습 모델 풀**이 Transformer만큼 풍부하지 않다. 따라서 현업에 즉시 도입하기에는 안정성 검증과 같은 리스크가 따를 수 있다.

- **새로운 기회의 창출**

  - Mamba와 같은 SSM 아키텍처는 기존에 메모리나 속도 한계로 불가능했던 서비스에 돌파구를 제공할 수 있다. 예를 들어, **긴 법률/의료 문서를 분석하고 질의응답하는 서비스**나 **수십 페이지에 달하는 대화 히스토리를 모두 기억하는 챗봇** 등에서 Mamba는 *게임 체인저*가 될 잠재력이 충분하다.

#### 미래 전망: 하이브리드와 상호보완

향후에는 **하이브리드 모델**(예: _Jamba_ - Transformer와 Mamba의 장점을 결합한 구조)이나 다른 선형 시간 아키텍처와의 경쟁을 통해 기술이 더욱 발전할 것이다. 현재로서는 **Transformer의 범용성 vs. Mamba의 특수성** 구도로 볼 수 있으며, 실제 프로덕션 환경에서는 두 접근법을 **상호보완적으로 활용**하는 방안이 유력하다. 예를 들어, 일반적인 짧은 대화는 Transformer로 처리하다가, 사용자가 긴 문서를 첨부하며 질문하는 특정 상황에서는 Mamba 모드로 전환하여 처리하는 지능형 시스템을 구상할 수 있다.

결론적으로, **BERT**와 **Mamba**는 각자의 명확한 강점을 바탕으로 **서로 다른 용도**를 갖는다고 볼 수 있다. 개발자는 해결하려는 문제의 핵심 제약 조건, 특히 **시퀀스 길이**를 기준으로 적합한 아키텍처를 선택해야 한다. 이번 실험은 **초장문맥과 고효율 추론을 위한 아키텍처 혁신이 현실화되고 있음**을 보여주는 의미 있는 이정표다.

### 체크포인트 질문

- BERT와 Mamba 모델의 **시간 복잡도** 차이점은 무엇이며, 이것이 긴 시퀀스 처리에서 어떤 영향을 미치는가?
- 본 실험에서 입력 문장의 길이를 512나 1024 토큰으로 늘려 실험을 반복하면 **추론 속도**와 **메모리 사용량** 결과는 어떻게 달라질지 예측해보라.
- Mamba 모델을 현재 당장 현업 시스템에 도입하기 어려운 이유는 무엇인가? 반대로 미래에 Mamba가 각광받을 만한 활용 시나리오는 어떤 것이 있는가?

---

## 참고자료

### 주요 논문 및 연구 자료

- PyTorch 공식 블로그 – _Better Performance with torch.compile_ (2023)
- Tri Dao 블로그 – _"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision"_
- Databricks DSPy 소개 – _Programming, not prompting_

### 기술 문서 및 구현체

- Hugging Face Transformers 문서 & 튜토리얼
- Deepset Haystack 문서 – _Flexible Open Source QA Framework_
- CrewAI 공식 문서 – _Role-based Autonomous Agent Teams_
- LangGraph 공식 문서 – _State-based Multi-Agent Orchestration_

### 온라인 리소스 및 블로그

- "torch.compile: A Deep Dive into PyTorch's Compiler" - PyTorch Blog
- "FlashAttention-3: The Next Generation of Attention Optimization" - Technical Blog
- "AI Agent Frameworks: A Comprehensive Comparison" - Medium
- "DSPy: The Future of Prompt Engineering" - Databricks Blog
