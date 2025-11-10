# LLM From Scratch Workshop

## Workshop Overview

Modern Large Language Models (LLMs) are often treated as 'black boxes' with their internal workings hidden. However, true expertise comes not just from using tools, but from understanding their fundamental principles. This workshop is based on this philosophy, aiming to achieve deep understanding beyond surface-level applications through the process of building LLMs 'from scratch'. Even though these may be small-scale models for educational purposes, the experience of building them directly provides unparalleled insights into the potential, limitations, and design choices that shape LLM behavior.

### Definition and Characteristics of LLMs

The Large Language Models covered in this workshop are concepts newly defined since the emergence of the Transformer architecture. This is not simply about size. Modern LLMs are distinguished from previous Natural Language Processing (NLP) models by three key characteristics:

1. **Scale**: Massive parameters ranging from billions to trillions
2. **Generative Pre-training**: Learning statistical patterns of language from large-scale text corpora before supervised learning for specific tasks
3. **Emergent Abilities**: Few-shot learning capabilities that can perform new tasks with just a few examples without separate fine-tuning

Even for educational small-scale models, the direct building experience provides unparalleled insights into the potential, limitations, and design choices that shape LLM behavior.

## Workshop Roadmap

| Week | Topic | Practical Goals | Tools Used | Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| 1 | LLM Overview and Environment Setup | Understanding LLM lifecycle, NeMo/HF practice environment setup | **NGC Container**, HF Transformers | Workshop environment preparation, simple model execution verification |
| 2 | Data Collection and Preprocessing | Korean corpus collection and preprocessing, quality improvement techniques | **NeMo Curator**, HF Datasets | Refined training corpus (Korean text) |
| 3 | Tokenizer Design and Construction | Korean tokenizer training, tokenization method comparison | **HF Tokenizers**, SentencePiece | Korean BPE tokenizer model |
| 4 | Model Architecture Exploration | Understanding Transformer and latest alternatives (Mamba, RWKV, etc.) | PyTorch (HF or NeMo AutoModel) | Small-scale model implementation and feature comparison |
| 5 | LLM Pre-training | Custom GPT model initialization and pre-training | **NeMo Run**, Megatron (AutoModel integration) | Korean-based LLM initial model |
| 6 | Fine-tuning and PEFT | Downstream task model fine-tuning, PEFT techniques application | **HF PEFT** (LoRA, WaveFT, DoRA, etc.) | Task-specific model (e.g., sentiment analyzer) |
| 7 | Model Evaluation and Prompt Utilization | Performance evaluation with KLUE benchmarks, prompt tuning practice | **HF Evaluation** (Metrics), generation output analysis | Evaluation report and response improvement tips |
| 8 | Inference Optimization and Deployment | Inference speed/memory optimization, production deployment environment setup | **TensorRT-LLM**, Triton, HF Pipelines | Lightweight model and demo service |
| 9 | Model Alignment | RLHF/DPO retraining for instruction-following models | **NeMo Aligner**, RLHF (DPO algorithm) | Instruction-following improved LLM |
| 10 | Integration and Conclusion | Full pipeline integration, model sharing and future task discussion | **NeMo & HF integration**, Gradio demo | Final demo and future development direction |

## Week 1: LLM Overview and Environment Setup

Week 1 provides an overview of the entire lifecycle of Large Language Models (LLMs) and prepares the practice environment. Using NVIDIA's **NGC Container**, we set up an environment that includes the NeMo framework and HuggingFace toolkit. We load example models using simple HuggingFace **Transformers** pipelines to verify operation and understand how NeMo and HF tools can work together. This ensures GPU acceleration environment and library compatibility needed for future practice, and helps understand the big picture of LLM workflows.

### Environment Setup Practice

```bash
# Run NVIDIA NGC container
docker run --gpus all -it --rm -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:23.10-py3

# Install required libraries
pip install transformers datasets accelerate
pip install nemo-toolkit[all]
```

### Basic Model Execution Example

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Simple text generation pipeline
generator = pipeline("text-generation", 
                    model="gpt2", 
                    device=0 if torch.cuda.is_available() else -1)

# Text generation test
prompt = "The future of artificial intelligence is"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

### Checkpoint Questions

- What are the most important stages in the entire LLM lifecycle?
- What are the main differences between NeMo and HuggingFace Transformers?
- What are the main considerations when running models in GPU environments?

## Week 2: Data Collection and Preprocessing

Week 2 covers **Korean corpus data collection and preprocessing**. Using **NeMo Curator**, we collect vast amounts of Korean text from Wikipedia, news, etc., and perform deduplication and filtering. For example, we load public datasets like KLUE corpus or NSMC sentiment corpus using **HuggingFace Datasets**, review quality, and add them to the training corpus. Curator's distributed processing filters out noisy data and builds homogeneous learning data. As a result, we secure **refined Korean text corpus** suitable for LLM pre-training and learn the considerations embedded in data composition.

### Korean Dataset Collection

```python
from datasets import load_dataset
import pandas as pd

# Load public Korean datasets
nsmc = load_dataset("nsmc")
klue_nli = load_dataset("klue", "nli")

# Korean Wikipedia data collection example
from datasets import load_dataset
wiki_ko = load_dataset("wikipedia", "20220301.ko", split="train[:10000]")

print(f"NSMC data: {len(nsmc['train'])} samples")
print(f"KLUE NLI data: {len(klue_nli['train'])} samples")
print(f"Wikipedia data: {len(wiki_ko)} samples")
```

### Data Cleaning and Preprocessing

```python
import re
from collections import Counter

def clean_korean_text(text):
    """Korean text cleaning function"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean special characters
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # Remove consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_by_length(texts, min_length=10, max_length=512):
    """Filter by length criteria"""
    return [text for text in texts 
            if min_length <= len(text) <= max_length]

# Apply data cleaning
cleaned_texts = [clean_korean_text(text) for text in raw_texts]
filtered_texts = filter_by_length(cleaned_texts)
```

### Checkpoint Questions

- What are the main quality indicators to consider when collecting Korean text data?
- How does NeMo Curator's distributed processing differ from existing data cleaning methods?
- What are the characteristics of data suitable for LLM pre-training?

## Week 3: Tokenizer Design and Construction

Week 3 involves directly building a **Korean tokenizer**. We train tokenizers based on **SentencePiece BPE** or WordPiece from the collected corpus and analyze whether tokenization results preserve Korean word units and context well. Using HuggingFace **🤗Tokenizers** library, we train custom tokenizers and perform **token segmentation comparison** with existing multilingual model tokenizers. For example, we check how sentences like "한국어 형태소" are tokenized and determine vocabulary size and tokenization strategies optimized for Korean. Through this practice, we build **custom tokenizer models** before LLM training and experience the importance of the tokenization stage.

### Korean Tokenizer Training

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Korean preprocessing setup
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Training configuration
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    min_frequency=2
)

# Train with Korean corpus
tokenizer.train_from_iterator(korean_texts, trainer)

# Post-processing setup
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", 2), ("</s>", 3)]
)
```

### Tokenizer Performance Comparison

```python
def compare_tokenizers(text, tokenizers):
    """Compare performance of multiple tokenizers"""
    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.encode(text)
        results[name] = {
            'tokens': tokens.tokens,
            'count': len(tokens.tokens),
            'ids': tokens.ids
        }
    return results

# Comparison example
text = "한국어 자연어 처리는 매우 흥미로운 분야입니다."
tokenizers = {
    'custom_korean': custom_tokenizer,
    'bert_multilingual': bert_tokenizer,
    'sentencepiece': sp_tokenizer
}

comparison = compare_tokenizers(text, tokenizers)
```

### Checkpoint Questions

- What are the main factors to consider when designing Korean tokenizers?
- What are the differences between BPE and WordPiece tokenizers in Korean processing?
- How does tokenizer vocabulary size affect model performance?

## Week 4: Model Architecture Exploration

Week 4 explores the diversity of **LLM model architectures**. We first review the core of Transformer structure (self-attention, feedforward, etc.) and examine latest alternative architectures. For example, **Mamba** is based on SSM (State Space Model) enabling linear inference for long sequences, achieving similar performance to transformers while dramatically improving inference latency and memory usage. Also, **RWKV** is an innovative LLM architecture 100% RNN-based, operating with linear time complexity without KV cache while achieving transformer-level performance. We also cover concepts of **DeepSeek**, the latest LLM from China. DeepSeek uses Mixture-of-Experts (MoE) structure to activate only some experts per input for efficiency, and introduces Multi-Head Latent Attention to show high performance with low resources. For practice, we implement small-scale Transformers and simple RNN models through PyTorch and compare **learning speed and memory usage** on the same data. Through this, we learn to understand various structural trade-offs and reflect latest research trends in LLM design.

### Transformer Architecture Implementation

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

### Mamba Architecture Implementation

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

# Mamba model composition
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

### Checkpoint Questions

- What are the time complexity differences between Transformer and Mamba architectures?
- How does RWKV combine the advantages of RNN and Transformer?
- What is the principle behind MoE (Mixture-of-Experts) structure improving efficiency?

## Week 5: LLM Pre-training

Week 5 involves **pre-training Korean LLMs** in earnest. Using the tokenizer and corpus prepared in previous weeks, we train GPT-series **basic language models** from scratch. We perform distributed training using NVIDIA's **NeMo Run** tool and Megatron-based recipes, and apply **NeMo AutoModel** functionality for HuggingFace integration. AutoModel allows loading HuggingFace model architectures directly in NeMo, with model parallelization and PyTorch JIT optimization supported by default. For example, we initialize custom GPT models with configured hidden size or layer count and train them in multi-GPU environments. We observe loss reduction trends through several epochs of training and evaluate initial model's language generation characteristics through **Korean sentence generation examples**. Through this week, we obtain **Korean-based LLM initial models** with our own corpus and practice large-scale pre-training processes and distributed training techniques.

### Pre-training Setup and Configuration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Load Korean tokenizer
tokenizer = AutoTokenizer.from_pretrained("custom_korean_tokenizer")

# Initialize GPT model
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",  # Use basic structure
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id
)

# Pre-training configuration
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

# Training data preparation
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
```

### Distributed Training Setup

```python
# Distributed training setup using NeMo
from nemo.collections.nlp.models.language_modeling import GPTModel

# NeMo GPT model configuration
nemo_model = GPTModel.from_pretrained(
    model_name="gpt2",
    trainer=Trainer(
        devices=4,  # Use 4 GPUs
        accelerator="gpu",
        strategy="ddp",  # Distributed data parallel
    )
)

# Train with Korean data
nemo_model.setup_training_data(
    train_file="korean_corpus.txt",
    validation_file="korean_validation.txt",
    tokenizer=tokenizer
)
```

### Checkpoint Questions

- What are the most important hyperparameters in LLM pre-training?
- What are the main factors to consider in distributed training?
- What are the differences between Korean and English pre-training?

## Week 6: Fine-tuning and PEFT

Week 6 involves **fine-tuning pre-trained models** for downstream tasks. We first specialize models to NSMC movie review sentiment analysis data through simple **supervised learning fine-tuning**. Instead of updating all parameters using HuggingFace's Trainer, we use PEFT techniques like **LoRA** to adjust only some weights for efficiency. LoRA application is easily done through HuggingFace **PEFT** library, and using **NeMo AutoModel**, we can attach LoRA adapters directly to pre-trained HF models for training. At this time, we also introduce latest techniques like **WaveFT** and **DoRA**. WaveFT achieves fine control and high-efficiency tuning superior to LoRA by learning only minimal parameters in the **wavelet domain** of weight residual matrices, experimentally showing that performance can be maintained with very few variables. **DoRA** (Weight-Decomposed LoRA) decomposes weight changes into magnitude and direction components for learning, achieving **accuracy closer to original full fine-tuning** than LoRA, which is NVIDIA's latest method. In practice, we perform the same sentiment analysis task with existing LoRA and DoRA and compare results. Through this, we learn **retraining techniques** that effectively retrain models with few resources and understand the advantages and disadvantages of each technique.

### LoRA Fine-tuning Implementation

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.1,
    bias="none"
)

# Apply LoRA to model
model = AutoModelForSequenceClassification.from_pretrained("korean_llm_base")
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
```

### DoRA Fine-tuning Implementation

```python
from peft import DoRAConfig, get_peft_model

# DoRA configuration
dora_config = DoRAConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["query", "value", "key", "dense"],
    lora_dropout=0.1,
    bias="none"
)

# Apply DoRA
model = get_peft_model(model, dora_config)
```

### Checkpoint Questions

- Why are PEFT techniques more efficient than full fine-tuning?
- What are the main differences between LoRA and DoRA?
- Which layers should be selected as targets during fine-tuning?

## Week 7: Model Evaluation and Prompt Utilization

Week 7 focuses on **model performance evaluation and utilization methods**. We first conduct quantitative evaluation using parts of **KLUE benchmark** on fine-tuned models. For example, we measure model accuracy using natural language inference (NLI) or question answering (MRC) data and calculate metrics like Accuracy, F1 using **HuggingFace's evaluate library**. We also manually review model responses to prepared prompts for **generative evaluation** or evaluate summary accuracy using metrics like BLEU/ROUGE. In this process, we cover considerations for Korean evaluation and introduce **model output rating evaluation** techniques using GPT-4 when necessary. We also conduct practice on **prompt optimization**. We adjust prompt phrases for the same question and observe changes in model response content, sharing **prompt engineering tips** to elicit desired output formats. For example, we give prompts that induce step-by-step thinking to make models answer reasoning processes in detail. Through this week, we learn to measure **objective model performance** and utilize models through **effective prompt design**.

### Model Performance Evaluation

```python
from evaluate import load
import torch

# KLUE benchmark evaluation
def evaluate_model(model, tokenizer, test_dataset):
    """Evaluate model performance"""
    
    # Accuracy evaluation
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

### Prompt Engineering

```python
def test_prompt_variations(model, tokenizer, question):
    """Test model responses with various prompts"""
    
    prompts = [
        f"Question: {question}\nAnswer:",
        f"Think step by step about the following question.\nQuestion: {question}\nAnswer:",
        f"You are a helpful AI assistant.\nQuestion: {question}\nAnswer:",
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

### Checkpoint Questions

- What special factors should be considered in Korean model evaluation?
- What are the core principles of prompt engineering?
- How can generative model quality be objectively evaluated?

## Week 8: Inference Optimization and Deployment

Week 8 covers **inference optimization techniques** for deploying completed models to production services. We first practice methods to reduce memory usage and increase CPU/GPU inference speed by quantizing model parameters to 8-bit or 4-bit. Using HuggingFace **Transformers** and **BitsAndBytes**, we generate INT8/INT4 quantized checkpoints and verify that response quality degradation is minimized. We then cover high-speed inference engine construction using NVIDIA's **TensorRT-LLM** toolkit. TensorRT-LLM automatically builds optimized TensorRT engines when LLMs are defined through Python API and efficiently performs inference on NVIDIA GPUs. For practice, we convert pre-trained models to TensorRT-LLM and deploy them through **Triton Inference Server** or **Gradio** interface. We measure **latency and throughput changes** before and after optimization to experience performance improvements. As a result, Week 8 teaches how to build **lightweight LLM services** and master optimization techniques to consider when deploying large models to real-world environments.

### Model Quantization

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "korean_llm_finetuned",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Deployment using Gradio

```python
import gradio as gr

def generate_response(message, history):
    """Generate response to user input"""
    inputs = tokenizer(message, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=generate_response,
    title="Korean LLM Chatbot",
    description="Chat with the Korean LLM built in this workshop."
)

demo.launch()
```

### Checkpoint Questions

- What impact does model quantization have on performance?
- What are the main factors to consider in production deployment?
- What are the advantages and disadvantages of inference optimization techniques?

## Week 9: Model Alignment

Week 9 focuses on the **Model Alignment** stage, practicing latest techniques to tune LLMs to user instructions or values. We first explain concepts of instruction-following model generation through Human Feedback reinforcement and learn procedures of representative method **RLHF** (Reinforcement Learning from Human Feedback). This includes learning **reward models** reflecting human feedback and optimizing language models through PPO algorithms. However, since RLHF is complex to implement and costly, we directly apply **DPO** (Direct Preference Optimization) presented as an alternative. DPO is a technique that **directly retrains models** using human preference data without separate reinforcement learning, showing performance comparable to RLHF while having the advantage of simple implementation. In practice, we retune our models to **follow instructions in conversation** using DPO algorithms with open **preference datasets** (e.g., rankings for instruction responses). Using NVIDIA's **NeMo-Aligner** toolkit, we can easily perform RLHF pipelines and DPO algorithms, and efficiently align models of hundreds of billions scale. After training completion, we input sensitive questions or complex instructions as prompts to verify that **safe and helpful responses** are generated. Through Week 9, participants understand the **importance of LLM Alignment** and implementation methods, and finally obtain user-friendly **instruction models**.

### DPO Implementation

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# DPO configuration
dpo_config = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    logging_steps=10,
)

# Prepare preference data
def prepare_dpo_data(examples):
    """Prepare data for DPO training"""
    return {
        "prompt": examples["instruction"],
        "chosen": examples["chosen_response"],
        "rejected": examples["rejected_response"]
    }

# DPO training
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
```

### Checkpoint Questions

- Why is model alignment important?
- What are the main differences between RLHF and DPO?
- What considerations are needed to build safe AI models?

## Week 10: Integration and Conclusion

The final Week 10 **integrates** the content covered so far to organize final deliverables and explore additional development directions. We first connect processes from Week 1 to Week 9 into one pipeline for review. We organize the flow from data preparation, tokenization, pre-training, fine-tuning, evaluation, optimization, to alignment, and review how NeMo and HuggingFace tools collaborated at each stage. We upload the **final Korean LLM** created through practice to HuggingFace Hub or share with team members to run actual Q&A demos. We also build simple web interfaces using **Gradio** to complete **chatbot demos** where general users input questions and models respond. In this process, we can attempt final tuning to improve response usefulness and stability through prompt design optimization or additional fine-tuning. Finally, we conclude the workshop by briefly discussing topics like multimodal integration, continuous model monitoring, and feedback loops, which are latest LLM research trends. Through the final week, participants organize their direct experience of **the entire LLM development cycle** and gain direction for practical applications and future learning.

### Full Pipeline Integration

```python
# Full workflow integration example
def complete_llm_pipeline():
    """Execute the entire LLM development pipeline"""
    
    # 1. Data preparation
    dataset = prepare_korean_corpus()
    
    # 2. Tokenizer training
    tokenizer = train_korean_tokenizer(dataset)
    
    # 3. Model pre-training
    base_model = pretrain_llm(dataset, tokenizer)
    
    # 4. Fine-tuning
    finetuned_model = fine_tune_with_peft(base_model, task_data)
    
    # 5. Model alignment
    aligned_model = align_model_with_dpo(finetuned_model, preference_data)
    
    # 6. Optimization and deployment
    optimized_model = optimize_for_inference(aligned_model)
    deploy_model(optimized_model)
    
    return optimized_model
```

### Final Demo Construction

```python
import gradio as gr
from transformers import pipeline

# Load final model
final_model = pipeline(
    "text-generation",
    model="korean_llm_final",
    tokenizer="korean_tokenizer"
)

def chat_with_model(message, history):
    """Function to chat with final model"""
    response = final_model(
        message,
        max_length=200,
        temperature=0.7,
        do_sample=True
    )
    return response[0]['generated_text']

# Final demo interface
demo = gr.ChatInterface(
    fn=chat_with_model,
    title="LLM From Scratch Workshop - Final Demo",
    description="Chat with the Korean LLM built from scratch in this workshop!"
)

demo.launch()
```

### Checkpoint Questions

- What is the most important stage in the LLM development process?
- What is the biggest insight gained through the workshop?
- What fields should be focused on in future LLM research?

---

## References

### Key Papers and Research Materials

- Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint.
- Peng, B., et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era." arXiv preprint.
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Liu, H., et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." arXiv preprint.
- Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv preprint.
- Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv preprint.

### Technical Documentation and Implementations

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- NVIDIA NeMo Documentation: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- Mamba GitHub Repository: https://github.com/state-spaces/mamba
- RWKV GitHub Repository: https://github.com/BlinkDL/RWKV-LM
- PEFT Library Documentation: https://huggingface.co/docs/peft
- TensorRT-LLM Documentation: https://docs.nvidia.com/tensorrt-llm/

### Online Resources and Blogs

- "A Visual Guide to Mamba and State Space Models" - Newsletter by Maarten Grootendorst
- "The RWKV language model: An RNN with the advantages of a transformer" - The Good Minima
- "Mamba Explained" - The Gradient
- "Introducing RWKV - An RNN with the advantages of a transformer" - Hugging Face Blog
- "Parameter-Efficient Fine-Tuning: A Comprehensive Guide" - Hugging Face Blog
- "DoRA: A High-Performing Alternative to LoRA" - NVIDIA Developer Blog
- "QLoRA: Making Large Language Models More Accessible" - Hugging Face Blog
