# LLM From Scratch 워크숍

## 워크숍 개요

현대의 대규모 언어 모델(LLM)은 종종 그 내부 작동 원리가 감춰진 '블랙박스'처럼 다뤄진다. 그러나 진정한 전문성은 단순히 도구를 사용하는 것을 넘어, 그 근본 원리를 이해하는 데서 비롯된다. 본 워크숍은 이러한 철학에 기반하여, LLM을 '처음부터(from scratch)' 구축하는 과정을 통해 표면적인 응용을 넘어선 심층적인 이해를 목표로 한다.

### LLM의 정의와 특징

본 워크숍에서 다루는 대규모 언어 모델은 트랜스포머(Transformer) 아키텍처의 등장 이후 새롭게 정의된 개념이다. 이는 단순히 크기만을 의미하는 것이 아니다. 현대 LLM은 세 가지 핵심적인 특징으로 이전의 자연어 처리(NLP) 모델과 구분된다:

1. **규모(Scale)**: 수십억에서 수조에 이르는 방대한 매개변수(parameter)
2. **생성적 사전 훈련(Generative Pre-training)**: 특정 작업에 대한 지도 학습 이전에 대규모 텍스트 코퍼스로부터 언어의 통계적 패턴을 학습
3. **창발적 능력(Emergent Abilities)**: 별도의 미세조정(fine-tuning) 없이도 몇 가지 예시만으로 새로운 작업을 수행하는 소수샷 학습(few-shot learning) 능력

비록 교육적인 목적의 소규모 모델일지라도, 직접 구축하는 경험은 LLM의 잠재력, 한계, 그리고 그 행동을 형성하는 설계상의 선택들에 대한 비할 데 없는 통찰력을 제공한다.

## 워크숍 로드맵

| 주차   | 주제                        | 실습 목표                                         | 사용 도구                               | 결과물                                 |
| :----- | :-------------------------- | :------------------------------------------------ | :-------------------------------------- | :------------------------------------- |
| 1주차  | LLM 개요 및 환경 구축       | LLM 수명주기 이해, NeMo/HF 실습 환경 설정         | **NGC 컨테이너**, HF Transformers       | 워크숍 환경 준비, 간단 모델 실행 확인  |
| 2주차  | 데이터 수집 및 정제         | 한국어 말뭉치 수집·전처리, 품질 향상 기법 실습    | **NeMo Curator**, HF Datasets           | 정제된 학습 코퍼스 (한국어 텍스트)     |
| 3주차  | 토크나이저 설계 및 구축     | 한국어 토크나이저 훈련, 토큰화 방식 비교 이해     | **HF 토크나이저**, SentencePiece        | 한국어 BPE 토크나이저 모델             |
| 4주차  | 모델 아키텍처 탐구          | Transformer와 최신 대안(Mamba, RWKV 등) 이해      | PyTorch (HF 또는 NeMo AutoModel)        | 소규모 모델 구현 및 특성 비교          |
| 5주차  | LLM 사전학습 (Pre-training) | 커스텀 GPT 모델 초기화 및 사전학습 진행           | **NeMo Run**, Megatron (AutoModel 통합) | 한국어 기반 LLM 초기 모델              |
| 6주차  | 미세조정 및 PEFT            | 다운스트림 작업용 모델 미세조정, PEFT 기법 적용   | **HF PEFT** (LoRA, WaveFT, DoRA 등)     | 과제 특화 모델 (예: 감성분석기)        |
| 7주차  | 모델 평가와 프롬프트 활용   | KLUE 등 벤치마크로 성능 평가, 프롬프트 튜닝 실습  | **HF 평가**(Metrics), 생성 출력 분석    | 평가 보고서 및 응답 향상 팁            |
| 8주차  | 추론 최적화와 배포          | 추론 속도/메모리 최적화, 실서비스 배포 환경 구성  | **TensorRT-LLM**, Triton, HF Pipelines  | 경량화 모델 및 데모 서비스             |
| 9주차  | 모델 정렬(Alignment)        | RLHF/DPO로 사용자 지침 준수 모델로 재훈련         | **NeMo Aligner**, RLHF(DPO 알고리즘)    | 지침 응답 개선된 LLM (인스트럭트 모델) |
| 10주차 | 통합 및 마무리              | 전체 파이프라인 통합, 모델 공유 및 향후 과제 논의 | **NeMo & HF 연동**, Gradio 데모         | 최종 데모 및 향후 발전 방향 정리       |

## 1. 1주차: LLM 개요 및 환경 구축

1주차에는 대형언어모델(LLM)의 전체 수명주기를 개괄하고 실습 환경을 준비한다. NVIDIA의 **NGC 컨테이너**를 활용해 NeMo 프레임워크와 HuggingFace 툴킷이 포함된 환경을 세팅한다. 간단한 HuggingFace **Transformers** 파이프라인으로 예시 모델을 불러와 작동을 확인하며, NeMo와 HF 도구들이 어떻게 함께 활용될 수 있는지 개념을 잡는다. 이를 통해 향후 실습에 필요한 GPU 가속 환경과 라이브러리 호환성을 확보하고, LLM 워크플로의 큰 그림을 이해한다.

### 1.1 환경 구축 실습

```bash
# NVIDIA NGC 컨테이너 실행
docker run --gpus all -it --rm -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:23.10-py3

# 필요한 라이브러리 설치
pip install transformers datasets accelerate
pip install nemo-toolkit[all]
```

### 1.2 기본 모델 실행 예시

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# 간단한 텍스트 생성 파이프라인
generator = pipeline("text-generation", 
                    model="gpt2", 
                    device=0 if torch.cuda.is_available() else -1)

# 텍스트 생성 테스트
prompt = "인공지능의 미래는"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

### 체크포인트 질문

- LLM의 전체 수명주기에서 가장 중요한 단계는 무엇인가?
- NeMo와 HuggingFace Transformers의 주요 차이점은 무엇인가?
- GPU 환경에서 모델을 실행할 때 고려해야 할 주요 사항들은 무엇인가?

## 2. 2주차: 데이터 수집 및 정제

2주차에서는 **한국어 말뭉치 데이터의 수집과 정제**를 다룬다. **NeMo Curator**를 사용하여 위키피디아, 뉴스 등 방대한 한국어 텍스트를 수집하고 중복 제거 및 필터링을 수행한다. 예를 들어 KLUE 말뭉치나 NSMC 감성 코퍼스 등의 공개 데이터셋을 **HuggingFace Datasets**로 불러와 품질 검토 후 훈련 코퍼스에 추가한다. Curator의 분산 처리로 노이즈가 많은 데이터를 걸러내고 균질한 학습 데이터를 구축한다. 결과적으로 LLM 사전학습에 적합한 **정제된 한국어 텍스트 코퍼스**를 확보하고 데이터 구성에 담긴 고려사항을 익힌다.

### 2.1 한국어 데이터셋 수집

```python
from datasets import load_dataset
import pandas as pd

# 공개 한국어 데이터셋 로드
nsmc = load_dataset("nsmc")
klue_nli = load_dataset("klue", "nli")

# 위키피디아 한국어 데이터 수집 예시
from datasets import load_dataset
wiki_ko = load_dataset("wikipedia", "20220301.ko", split="train[:10000]")

print(f"NSMC 데이터: {len(nsmc['train'])}개")
print(f"KLUE NLI 데이터: {len(klue_nli['train'])}개")
print(f"위키피디아 데이터: {len(wiki_ko)}개")
```

### 2.2 데이터 정제 및 전처리

```python
import re
from collections import Counter

def clean_korean_text(text):
    """한국어 텍스트 정제 함수"""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 특수 문자 정리
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_by_length(texts, min_length=10, max_length=512):
    """길이 기준 필터링"""
    return [text for text in texts 
            if min_length <= len(text) <= max_length]

# 데이터 정제 적용
cleaned_texts = [clean_korean_text(text) for text in raw_texts]
filtered_texts = filter_by_length(cleaned_texts)
```

### 체크포인트 질문

- 한국어 텍스트 데이터를 수집할 때 고려해야 할 주요 품질 지표는 무엇인가?
- NeMo Curator의 분산 처리 방식이 기존 데이터 정제 방법과 어떻게 다른가?
- LLM 사전학습에 적합한 데이터의 특징은 무엇인가?

## 3. 3주차: 토크나이저 설계 및 구축

3주차에는 **한국어 토크나이저**를 직접 만들어 본다. 수집한 말뭉치로부터 **SentencePiece BPE**나 WordPiece 기반의 토크나이저를 학습시키고, 토큰화 결과가 한국어의 단어 단위와 문맥을 잘 보존하는지 분석한다. HuggingFace **🤗Tokenizers** 라이브러리를 활용하여 사용자 정의 토크나이저를 훈련하고, 기존 multilingual 모델의 토크나이저와 **토큰 분할 비교**를 수행한다. 예를 들어 "한국어 형태소" 문장이 토큰화되는 방식을 확인하며, 한국어에 최적화된 어휘집 크기와 토큰화 전략을 결정한다. 이번 실습을 통해 LLM 학습 전에 **맞춤형 토크나이저 모델**을 구축하고, 토크나이즈 단계의 중요성을 체감한다.

### 3.1 한국어 토크나이저 훈련

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

# BPE 토크나이저 초기화
tokenizer = Tokenizer(models.BPE())

# 한국어 전처리 설정
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 훈련 설정
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    min_frequency=2
)

# 한국어 말뭉치로 훈련
tokenizer.train_from_iterator(korean_texts, trainer)

# 후처리 설정
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", 2), ("</s>", 3)]
)
```

### 3.2 토크나이저 성능 비교

```python
def compare_tokenizers(text, tokenizers):
    """여러 토크나이저의 성능을 비교한다"""
    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.encode(text)
        results[name] = {
            'tokens': tokens.tokens,
            'count': len(tokens.tokens),
            'ids': tokens.ids
        }
    return results

# 비교 예시
text = "한국어 자연어 처리는 매우 흥미로운 분야입니다."
tokenizers = {
    'custom_korean': custom_tokenizer,
    'bert_multilingual': bert_tokenizer,
    'sentencepiece': sp_tokenizer
}

comparison = compare_tokenizers(text, tokenizers)
```

### 체크포인트 질문

- 한국어 토크나이저 설계 시 고려해야 할 주요 요소들은 무엇인가?
- BPE와 WordPiece 토크나이저의 한국어 처리에서의 차이점은 무엇인가?
- 토크나이저의 어휘집 크기가 모델 성능에 미치는 영향을 설명하라.

## 4. 4주차: 모델 아키텍처 탐구

4주차에서는 **LLM 모델 아키텍처**의 다양성을 탐구한다. 우선 Transformer 구조의 핵심 (셀프어텐션, 피드포워드 등)을 복습하고, 최신 대안 아키텍처들을 검토한다. 예를 들어 **Mamba**는 SSM(State Space Model) 기반으로 긴 시퀀스에 대한 선형적 추론을 가능케 하여, 트랜스포머와 유사한 성능을 내면서도 추론 지연과 메모리 사용이 대폭 개선된 구조이다. 또한 **RWKV**는 100% RNN 기반의 혁신적 LLM 아키텍처로, KV 캐시 없이도 선형 시간복잡도로 동작하면서 트랜스포머 수준의 성능을 달성한다. 이와 함께 중국발 최신 LLM인 **DeepSeek**의 개념도 다룬다. DeepSeek은 Mixture-of-Experts(MoE) 구조로 입력마다 일부 전문가만 활성화해 효율성을 높이고, Multi-Head Latent Attention 등을 도입하여 낮은 자원으로도 높은 성능을 보이는 것이 특징이다. 실습으로는 PyTorch를 통해 소규모 Transformer와 간단한 RNN 모델을 구현해 같은 데이터에서 **학습 속도와 메모리 사용률을 비교**해본다. 이를 통해 다양한 구조상의 trade-off를 파악하고, 최신 연구 동향을 LLM 설계에 반영하는 법을 배운다.

### 4.1 Transformer 아키텍처 구현

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context)
```

### 4.2 Mamba 아키텍처 구현

```python
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Residual connection with layer norm
        return self.norm(x + self.mamba(x))

# Mamba 모델 구성
class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
```

### 체크포인트 질문

- Transformer와 Mamba 아키텍처의 시간 복잡도 차이점은 무엇인가?
- RWKV가 RNN과 Transformer의 장점을 결합한 방식은 무엇인가?
- MoE(Mixture-of-Experts) 구조가 효율성을 높이는 원리는 무엇인가?

## 5. 5주차: LLM 사전학습 (Pre-training)

5주차에는 본격적으로 **한국어 LLM을 사전학습**한다. 지난 주차에 준비된 토크나이저와 말뭉치를 사용하여, GPT 계열의 **기초 언어모델**을 처음부터 훈련시킨다. NVIDIA의 **NeMo Run** 툴과 Megatron 기반 레시피를 활용해 분산 학습을 수행하고, HuggingFace와의 통합을 위해 **NeMo AutoModel** 기능을 적용한다. AutoModel을 통해 HuggingFace 모델 아키텍처를 NeMo에서 바로 불러와 사용할 수 있으며, 모델 병렬화와 PyTorch JIT 최적화 등이 기본 지원된다. 예를 들어 hidden size나 레이어 수 등을 설정한 커스텀 GPT 모델을 random 초기화 후 다중 GPU 환경에서 학습시킨다. 몇 epoch의 훈련을 거치며 손실 감소 추이를 관찰하고, **한국어 문장 생성 예시**를 통해 초기 모델의 언어 생성 특성을 평가한다. 이번 주차를 통해 자체 말뭉치로 **한국어 기반 LLM 초기 모델**을 얻고, 대규모 사전학습 과정과 분산 훈련 기법을 실습하게 된다.

### 5.1 사전학습 설정 및 구성

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# 한국어 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("custom_korean_tokenizer")

# GPT 모델 초기화
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # 기본 구조 사용
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id
)

# 사전학습 설정
training_args = TrainingArguments(
    output_dir="./korean_llm_pretraining",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-4,
    warmup_steps=1000,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# 훈련 데이터 준비
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
```

### 5.2 분산 학습 설정

```python
# NeMo를 활용한 분산 학습 설정
from nemo.collections.nlp.models.language_modeling import GPTModel

# NeMo GPT 모델 구성
nemo_model = GPTModel.from_pretrained(
    model_name="gpt2",
    trainer=Trainer(
        devices=4,  # 4개 GPU 사용
        accelerator="gpu",
        strategy="ddp",  # 분산 데이터 병렬
    )
)

# 한국어 데이터로 훈련
nemo_model.setup_training_data(
    train_file="korean_corpus.txt",
    validation_file="korean_validation.txt",
    tokenizer=tokenizer
)
```

### 체크포인트 질문

- LLM 사전학습에서 가장 중요한 하이퍼파라미터들은 무엇인가?
- 분산 학습 시 고려해야 할 주요 요소들은 무엇인가?
- 한국어 사전학습에서 영어 모델과 다른 점은 무엇인가?

## 6. 6주차: 미세조정 및 PEFT

6주차에서는 사전학습된 모델을 **다운스트림 과제에 맞게 미세조정(fine-tuning)**한다. 우선 간단한 **지도학습 미세조정**으로 NSMC 영화리뷰 감성분석 데이터에 모델을 특화시켜 본다. HuggingFace의 Trainer를 활용해 전체 파라미터를 업데이트하는 대신, **LoRA**와 같은 PEFT 기법으로 일부 가중치만 조정해 효율을 높인다. LoRA 적용은 HuggingFace **PEFT** 라이브러리를 통해 손쉽게 이루어지며, **NeMo AutoModel**을 사용하면 사전학습한 HF 모델에 바로 LoRA 어댑터를 붙여 훈련할 수도 있다. 이때 **WaveFT**와 **DoRA** 같은 최신 기법도 소개한다. WaveFT는 가중치 잔여행렬의 **웨이블릿 영역**에서 극소 파라미터만 학습하여 LoRA보다 미세한 제어와 고효율 튜닝을 달성하는 방법으로, 매우 적은 변수만으로도 성능을 유지할 수 있음을 실험으로 보여주었다. **DoRA**(Weight-Decomposed LoRA)는 가중치 변화량을 크기와 방향 성분으로 분해하여 학습함으로써, LoRA 대비 **정확도가 원본 풀 파인튜닝에 한층 가까운 결과**를 내는 NVIDIA의 최신 방식이다. 실습에서는 기존 LoRA와 DoRA로 같은 감성분석 태스크를 수행해보고 결과를 비교한다. 이를 통해 적은 자원으로 모델을 효과적으로 **재훈련하는 기술들**을 습득하고, 각 기법의 장단점을 이해하게 된다.

### 6.1 LoRA 미세조정 구현

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.1,
    bias="none"
)

# 모델에 LoRA 적용
model = AutoModelForSequenceClassification.from_pretrained("korean_llm_base")
model = get_peft_model(model, lora_config)

# 훈련 가능한 파라미터 확인
model.print_trainable_parameters()
```

### 6.2 DoRA 미세조정 구현

```python
from peft import DoRAConfig, get_peft_model

# DoRA 설정
dora_config = DoRAConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.1,
    bias="none"
)

# DoRA 적용
model = get_peft_model(model, dora_config)
```

### 체크포인트 질문

- PEFT 기법이 전체 파인튜닝보다 효율적인 이유는 무엇인가?
- LoRA와 DoRA의 주요 차이점은 무엇인가?
- 미세조정 시 어떤 레이어를 대상으로 선택해야 하는가?

## 7. 7주차: 모델 평가와 프롬프트 활용

7주차에는 모델의 **성능 평가와 활용 방법**에 초점을 맞춘다. 우선 미세조정된 모델을 대상으로 **KLUE 벤치마크**의 일부를 사용해 정량 평가를 진행한다. 예를 들어 자연어 추론(NLI)이나 질의응답(MRC) 데이터로 모델의 정확도를 측정하고, **HuggingFace의 evaluate 라이브러리**로 Accuracy, F1 등의 지표를 산출한다. 또한 **생성형 평가**를 위해 준비된 프롬프트에 모델이 응답한 결과를 수동 검토하거나, BLEU/ROUGE 같은 지표로 요약문 정확도를 평가해 본다. 이 과정에서 한국어 평가의 유의점을 다루고, 필요한 경우 GPT-4 등을 활용한 **모델 출력 평점 평가** 기법도 소개한다. 아울러 **프롬프트 최적화(prompt optimization)**에 관한 실습도 병행한다. 동일한 질문에 대해 프롬프트 문구를 조정해보며 모델 응답 내용의 변화를 관찰하고, 원하는 출력 형식을 이끌어내는 **프롬프트 엔지니어링 팁**을 공유한다. 예를 들어 모델에게 단계적 사고를 유도하는 프롬프트를 줘서 추론 과정을 상세히 답변하게 해 보는 식이다. 이번 주를 통해 **모델의 객관적 성능**을 측정하고, **효과적인 프롬프트 설계**로 모델을 활용하는 방법을 익힌다.

### 7.1 모델 성능 평가

```python
from evaluate import load
import torch

# KLUE 벤치마크 평가
def evaluate_model(model, tokenizer, test_dataset):
    """모델의 성능을 평가한다"""
    
    # 정확도 평가
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    
    predictions = []
    references = []
    
    for batch in test_dataset:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            references.extend(batch["label"])
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")
    
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}
```

### 7.2 프롬프트 엔지니어링

```python
def test_prompt_variations(model, tokenizer, question):
    """다양한 프롬프트로 모델 응답을 테스트한다"""
    
    prompts = [
        f"질문: {question}\n답변:",
        f"다음 질문에 대해 단계별로 생각해보세요.\n질문: {question}\n답변:",
        f"당신은 도움이 되는 AI 어시스턴트입니다.\n질문: {question}\n답변:",
    ]
    
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
    
    return responses
```

### 체크포인트 질문

- 한국어 모델 평가에서 고려해야 할 특별한 요소들은 무엇인가?
- 프롬프트 엔지니어링의 핵심 원칙은 무엇인가?
- 생성형 모델의 품질을 객관적으로 평가하는 방법은 무엇인가?

## 8. 8주차: 추론 최적화와 배포

8주차에서는 완성된 모델을 **실서비스에 배포**하기 위한 **추론 최적화 기법**을 다룬다. 우선 모델 파라미터를 8-bit 혹은 4-bit로 양자화하여 메모리 사용을 줄이고 CPU/GPU 추론 속도를 높이는 방법을 실습한다. HuggingFace **Transformers**와 **BitsAndBytes** 등을 이용해 INT8/INT4 양자화된 체크포인트를 생성하고, 응답 품질 저하가 최소화되는지 확인한다. 이어서 NVIDIA의 **TensorRT-LLM** 툴킷을 활용한 고속 추론 엔진 구축을 다룬다. TensorRT-LLM은 파이썬 API를 통해 LLM을 정의하면 자동으로 최적화된 TensorRT 엔진을 빌드해주며, NVIDIA GPU에서 효율적으로 추론을 수행한다. 실습으로 사전학습한 모델을 TensorRT-LLM으로 변환한 뒤, **Triton Inference Server**나 **Gradio** 인터페이스를 통해 배포한다. 이때 최적화 전후의 **레이턴시와 Throughput 변화**를 측정하여 성능 향상을 체감한다. 결과적으로 8주차에서는 **경량화된 LLM 서비스**를 구축하는 법을 배우며, 대용량 모델을 실사용 환경에 올릴 때 고려해야 할 최적화 기법들을 숙지한다.

### 8.1 모델 양자화

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

# 4비트 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 양자화된 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "korean_llm_finetuned",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 8.2 Gradio를 활용한 배포

```python
import gradio as gr

def generate_response(message, history):
    """사용자 입력에 대한 응답을 생성한다"""
    inputs = tokenizer(message, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio 인터페이스 생성
demo = gr.ChatInterface(
    fn=generate_response,
    title="한국어 LLM 챗봇",
    description="워크숍에서 구축한 한국어 LLM과 대화해보세요."
)

demo.launch()
```

### 체크포인트 질문

- 모델 양자화가 성능에 미치는 영향은 무엇인가?
- 실서비스 배포 시 고려해야 할 주요 요소들은 무엇인가?
- 추론 최적화 기법들의 장단점은 무엇인가?

## 9. 9주차: 모델 정렬(Alignment)

9주차에서는 **모델 정렬(Alignment)** 단계에 집중하여, LLM을 사용자 지침이나 가치에 맞게 튜닝하는 최신 기법들을 실습한다. 우선 Human Feedback 강화를 통한 지침 준수 모델 생성 개념을 설명하고, 대표적 방법인 **RLHF**(Reinforcement Learning from Human Feedback)의 절차를 알아본다. 여기에는 인간 피드백이 반영된 **보상모델** 학습과, PPO 알고리즘으로 언어모델을 최적화하는 과정이 포함된다. 다만 RLHF는 구현이 복잡하고 비용이 크므로, 대안으로 제시된 **DPO**(Direct Preference Optimization)를 직접 적용해 본다. DPO는 별도의 강화학습 없이도 인간 선호 데이터를 이용해 **모델을 직접 재학습**시키는 기법으로, RLHF에 준하는 성능을 보이면서도 구현이 단순한 장점이 있다. 실습에서는 오픈된 **선호도 데이터셋**(예: 인스트럭션 응답에 대한 랭킹)을 활용하여 DPO 알고리즘으로 우리 모델을 **지침 따라 대화하도록** 재튜닝한다. NVIDIA의 **NeMo-Aligner** 툴킷을 사용하면 RLHF 파이프라인과 DPO 알고리즘을 손쉽게 수행할 수 있으며, 수백 억 규모 모델도 효율적으로 정렬시킬 수 있다. 훈련 완료 후, 모델에게 민감한 질문이나 복합 지시를 프롬프트로 입력하여, **안전하고 도움되는 응답**을 생성하는지 확인한다. 9주차를 통해 참가자들은 **LLM Alignment의 중요성**과 구현 방법을 이해하고, 최종적으로 사용자 친화적인 **인스트럭트 모델**을 얻게 된다.

### 9.1 DPO 구현

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# DPO 설정
dpo_config = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=10,
)

# 선호도 데이터 준비
def prepare_dpo_data(examples):
    """DPO 훈련을 위한 데이터를 준비한다"""
    return {
        "prompt": examples["instruction"],
        "chosen": examples["chosen_response"],
        "rejected": examples["rejected_response"]
    }

# DPO 훈련
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
```

### 체크포인트 질문

- 모델 정렬(Alignment)이 왜 중요한가?
- RLHF와 DPO의 주요 차이점은 무엇인가?
- 안전한 AI 모델을 구축하기 위한 고려사항은 무엇인가?

## 10. 10주차: 통합 및 마무리

마지막 10주차에서는 그동안 다룬 내용을 **통합**하여 최종 결과물을 정리하고, 추가 발전 방향을 모색한다. 먼저 1주차부터 9주차까지의 과정을 하나의 파이프라인으로 연결하여 복습한다. 데이터 준비부터 토크나이징, 사전학습, 미세조정, 평가, 최적화, 정렬까지의 흐름을 정리하고, 각 단계에서 NeMo와 HuggingFace 도구들이 어떻게 협력했는지 되짚는다. 실습 결과 만들어진 **최종 한국어 LLM**을 HuggingFace Hub에 업로드하거나, 팀원들과 공유하여 실제 질의응답 데모를 실행해 본다. 또한 **Gradio** 등을 이용해 간단한 웹 인터페이스를 구성함으로써, 일반 사용자가 질문을 입력하고 모델이 응답하는 **챗봇 데모**를 완성한다. 이 과정에서 프롬프트 설계 최적화나 추가 미세조정을 통해 응답의 유용성과 안정성을 개선하는 마지막 튜닝을 시도할 수 있다. 마무리로, 최신 LLM 연구 동향인 멀티모달 통합, 지속적인 모델 모니터링과 피드백 루프 등의 주제를 짧게 토의하며 워크숍을 끝맺는다. 최종 주차를 통해 참가자들은 **LLM 개발의 전체 사이클**을 직접 경험한 것을 정리하고, 실무 응용 및 향후 학습에 대한 방향을 얻는다.

### 10.1 전체 파이프라인 통합

```python
# 전체 워크플로우 통합 예시
def complete_llm_pipeline():
    """LLM 개발의 전체 파이프라인을 실행한다"""
    
    # 1. 데이터 준비
    dataset = prepare_korean_corpus()
    
    # 2. 토크나이저 훈련
    tokenizer = train_korean_tokenizer(dataset)
    
    # 3. 모델 사전학습
    base_model = pretrain_llm(dataset, tokenizer)
    
    # 4. 미세조정
    finetuned_model = fine_tune_with_peft(base_model, task_data)
    
    # 5. 모델 정렬
    aligned_model = align_model_with_dpo(finetuned_model, preference_data)
    
    # 6. 최적화 및 배포
    optimized_model = optimize_for_inference(aligned_model)
    deploy_model(optimized_model)
    
    return optimized_model
```

### 10.2 최종 데모 구축

```python
import gradio as gr
from transformers import pipeline

# 최종 모델 로드
final_model = pipeline(
    "text-generation",
    model="korean_llm_final",
    tokenizer="korean_tokenizer"
)

def chat_with_model(message, history):
    """최종 모델과 대화하는 함수"""
    response = final_model(
        message,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )
    return response[0]['generated_text']

# 최종 데모 인터페이스
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="LLM From Scratch 워크숍 - 최종 데모",
    description="워크숍에서 처음부터 구축한 한국어 LLM과 대화해보세요!"
)

demo.launch()
```

### 체크포인트 질문

- LLM 개발 과정에서 가장 중요한 단계는 무엇인가?
- 워크숍을 통해 얻은 가장 큰 인사이트는 무엇인가?
- 향후 LLM 연구에서 주목해야 할 분야는 무엇인가?

---

## 참고자료

### 주요 논문 및 연구 자료

- Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint.
- Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." arXiv preprint.
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Liu, H., et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." arXiv preprint.
- Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv preprint.
- Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv preprint.

### 기술 문서 및 구현체

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- NVIDIA NeMo Documentation: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- Mamba GitHub Repository: https://github.com/state-spaces/mamba
- RWKV GitHub Repository: https://github.com/BlinkDL/RWKV-LM
- PEFT Library Documentation: https://huggingface.co/docs/peft
- TensorRT-LLM Documentation: https://docs.nvidia.com/tensorrt-llm/

### 온라인 리소스 및 블로그

- "A Visual Guide to Mamba and State Space Models" - Newsletter by Maarten Grootendorst
- "The RWKV language model: An RNN with the advantages of a transformer" - The Good Minima
- "Mamba Explained" - The Gradient
- "Introducing RWKV - An RNN with the advantages of a transformer" - Hugging Face Blog
- "Parameter-Efficient Fine-Tuning: A Comprehensive Guide" - Hugging Face Blog
- "DoRA: A High-Performing Alternative to LoRA" - NVIDIA Developer Blog
- "QLoRA: Making Large Language Models More Accessible" - Hugging Face Blog
