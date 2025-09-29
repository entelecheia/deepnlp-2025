# Deep Learning Natural Language Processing (131307379A) Syllabus

## Overview

In recent years, natural language processing (NLP) research has undergone a massive transformation. The emergence of large language models (LLMs) has dramatically improved the ability to generate and understand text, revolutionizing various application domains such as translation, question answering, and summarization. In 2024-2025, multimodal LLMs like GPT-5 and **Gemini 2.5 Pro** that can simultaneously process text, images, and audio have emerged, further expanding the scope of applications. Particularly noteworthy is the emergence of new architectures beyond **Transformer**. For example, **Mamba**, a state space model (SSM), can efficiently process up to millions of tokens with linear O(n) complexity, while **RWKV** can process conversational messages at 10x or more lower cost than existing methods in real-time.

This course reflects these latest developments to provide **hands-on** deep learning-based NLP techniques. Students first learn **core tool utilization methods** such as PyTorch and Hugging Face usage, then directly experience fine-tuning of **Transformer-based models and latest SSM architectures**, **prompt engineering**, **retrieval-augmented generation (RAG)**, **reinforcement learning from human feedback (RLHF)**, and **agent framework** implementation. Additionally, we cover latest **parameter-efficient fine-tuning (PEFT)** techniques (WaveFT, DoRA, VB-LoRA, etc.) and advanced **RAG architectures** (HippoRAG, GraphRAG), and practice cutting-edge concepts such as **multimodal LLMs** and **ultra-long context processing**. Finally, through team projects, students integrate learned content to implement **complete models and applications** that solve real problems.

This course is designed for third-year undergraduate level and assumes completion of prerequisite course _Language Models and Natural Language Processing (131107967A)_. Through team projects, students challenge real problem-solving using Korean corpora, and in the final project phase, we provide opportunities to work with industry datasets and receive feedback from industry experts, considering **industry-academia collaboration**.

## Learning Objectives

- Understand **the role and limitations of large language models** in modern NLP and utilize related tools such as PyTorch and Hugging Face.

- Understand the principles and trade-offs of **State Space Models** (e.g., Mamba, RWKV) along with **latest architectures**.

- Apply **fine-tuning** to pre-trained models or latest **parameter-efficient fine-tuning methods** like WaveFT, DoRA, VB-LoRA.

- Learn methods to systematically optimize prompts using **prompt engineering** techniques and DSPy framework.

- Understand **the evolution of evaluation metrics** (e.g., G-Eval, LiveCodeBench, etc.) and the importance of human evaluation, and learn latest alternatives to RLHF such as DPO (Direct Preference Optimization).

- Design and implement **advanced RAG** (Retrieval-Augmented Generation) architectures like **HippoRAG, GraphRAG** and hybrid search strategies.

- Understand AI regulatory frameworks like **EU AI Act** and acquire methodologies for implementing responsible AI systems.

- Track latest research trends to discuss **multimodal LLMs**, **small language models (SLM)**, **state space models (SSM)**, **multi-agent systems**, **mixture of experts (MoE)**, and other diverse latest technologies.

- Understand **the characteristics and challenges of Korean NLP** and develop application capabilities through hands-on practice using Korean corpora.

- Strengthen **collaboration and practical problem-solving capabilities** through team projects and gain project experience connected to industry.

## Course Schedule

| Week | Main Topics and Keywords                                                                                                                 | Key Hands-on/Assignments                                                                           |
| :--: | :---------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
|  1   | **Transformer and Next-Generation Architectures**<br/>• Self-Attention Mechanism and Limitations<br/>• **Mamba (Selective State Space Model)**<br/>• **RWKV (RNN-Transformer Hybrid)**<br/>• **Jamba (MoE-based Transformer+Mamba)** | **NVIDIA NGC Container Environment Setup**<br/>**Hugging Face Transformers Practice**<br/>**Mamba vs Transformer Performance Comparison Experiment** |
|  2   | **PyTorch 2.x and Latest Deep Learning Frameworks**<br/>• **torch.compile and Compiler Revolution**<br/>• **FlashAttention-3 Hardware Acceleration**<br/>• **AI Agent Frameworks** (DSPy, Haystack, CrewAI, LangGraph) | **torch.compile Performance Optimization Practice**<br/>**FlashAttention-3 Implementation and Comparison**<br/>**AI Agent System Construction** |
|  3   | **Modern PEFT Techniques for Efficient Fine-tuning**<br/>• **LoRA (Low-Rank Adaptation)**<br/>• **DoRA (Weight-Decomposed LoRA)**<br/>• **QLoRA (4-bit Quantization + LoRA)**<br/>• **VB-LoRA, WaveFT and Latest Techniques** | **PEFT Method Comparison Experiment**<br/>**LoRA/DoRA/QLoRA Performance Evaluation through Korean Sentiment Analysis**<br/>**Memory Efficiency and Inference Speed Analysis** |
|  4   | **Advanced Prompt Techniques and Optimization**<br/>• **Systematic Prompt Techniques** (Role Assignment, Structured Prompting)<br/>• **Self-Consistency and Tree of Thoughts**<br/>• **DSPy Framework** (Declarative Prompt Programming)<br/>• **Automatic Prompt Optimization (APE)** | **DSPy-based Automatic Prompt Optimization**<br/>**Self-Consistency Decoding Implementation**<br/>**Tree of Thoughts Problem Solving Practice** |
|  5   | **LLM Evaluation Paradigms and Benchmarks**<br/>• **Limitations of Traditional Metrics** (BLEU/ROUGE vs Meaning-based Evaluation)<br/>• **LLM-as-a-Judge** (GPTScore, G-Eval, FLASK)<br/>• **Specialized Benchmarks** (LiveCodeBench, MMLU-Pro)<br/>• **Domain-specific Benchmarks** (FinBen, AgentHarm, LEXam) | **BLEU/ROUGE vs G-Eval Comparison Experiment**<br/>**GPTScore Implementation and Evaluation**<br/>**FLASK Multi-dimensional Evaluation System Construction** |
|  6   | Seq2Seq Applications and **Multimodal Integration** – SmolVLM2, Qwen 2.5 Omni, Speech-Text Models                                                       | Multimodal Application Development Assignment 2                                                        |
|  7   | Large-scale Models and Few-shot Learning<br/>**Ultra-long Context Processing Technology** (1M+ tokens)                                                                | Long Context Processing Strategy Comparison Practice                                                               |
|  8   | **Next-generation PEFT** – WaveFT, DoRA, VB-LoRA, QLoRA, etc. Latest Techniques                                                                         | Performance Comparison Experiments of Various PEFT Techniques                                                          |
|  9   | **Advanced RAG Systems** – HippoRAG, GraphRAG, Hybrid Search Strategies                                                                      | Assignment 3: Building **Korean Enterprise Search System** based on GraphRAG                           |
|  10  | **Innovation in Alignment Techniques** – DPO, Constitutional AI, Process Reward Models                                                                | Comparison Practice between DPO and Existing RLHF Techniques                                                           |
|  11  | **Production Agent Systems** – CrewAI, Mirascope, Type-Safety Development                                                                | Multi-agent Orchestration Implementation                                                         |
|  12  | **AI Regulation and Responsible AI** – EU AI Act, Differential Privacy, Federated Learning                                                                  | Assignment for Designing Regulation-Compliant AI Systems                                                            |
|  13  | **Latest Research Trends** – Small Language Models (Gemma 3, Mistral NeMo), Enhanced Reasoning (Long CoT, PAL)                                               | Student Presentations of Latest Papers and Comprehensive Discussion                                                       |
|  14  | Final Project Development and MLOps                                                                                                         | Team Prototype Implementation and Feedback Sessions **(Industry Mentor Participation)**                                 |
|  15  | Final Project Presentations and Comprehensive Evaluation                                                                                                     | Team Presentations, Course Content Summary and Future Prospects Discussion                                            |

## Weekly Educational Content

### Week 1 – Latest Trends in Generative AI

#### Core Topics

- **Development History of LLMs and Latest Model Introduction**: Features and performance comparison of latest models such as GPT-5, Gemini 2.5 Pro, Claude 4.1 Opus
- **Limitations of Transformer Architecture**: O(n²) complexity problems and difficulties in long sequence processing
- **Overview of New Architectures**: Innovative approaches replacing Transformer such as Mamba, RWKV

#### Hands-on/Activities

- **Environment Setup**: PyTorch/Conda development environment configuration, Hugging Face Transformers installation
- **Key Hands-on**: Simple Q&A demo using Hugging Face pipeline
- **Comparison Experiment**: Response quality and speed comparison between Transformer-based and latest models

### Week 2 – Tool Learning for Deep Learning NLP

#### Core Topics

- **PyTorch Basics**: Core concepts of deep learning framework such as tensor operations and automatic differentiation
- **Hugging Face Transformers**: Usage of pre-trained models and pipeline usage
- **FlashAttention-3**: Large batch processing acceleration technique (~2× speed improvement on H100 GPU)
- **NLP Ecosystem Tools**: Introduction to specialized frameworks such as DSPy, Haystack, CrewAI

#### Hands-on/Activities

- **Key Hands-on**: Load pre-trained language models (BERT) and latest SSM (Mamba) models respectively, apply to Korean classification tasks
- **Performance Comparison**: Performance and efficiency comparison analysis on identical Korean datasets

### Week 3 – Efficient Fine-tuning (PEFT) Techniques

#### Core Topics

- **Parameter-Efficient Fine-tuning**: Lightweight techniques that achieve 95% or more performance with <1% parameters compared to full fine-tuning
- **Latest PEFT Methodologies**:
  - _WaveFT_: Improve efficiency by sparsifying parameter updates in **frequency domain (Wavelet)**
  - _DoRA_: Adaptive fine-tuning through **weight decomposition** (fine-grained representation learning)
  - _VB-LoRA_: **Vector bank-based LoRA** extension for multi-user·task environments
  - _QR-Adaptor_: Adapter technique that simultaneously optimizes **quantization (Q)** bitwidth and LoRA rank (R)
- **Model Lightweighting Trends**: 4-bit quantization format NF4 (NormalFloat4) becoming the de facto standard for QLoRA, reducing 7B models from 10GB→1.5GB memory

#### Hands-on/Assignment

- **Programming Assignment 1**: Perform fine-tuning experiments on the same Korean dataset using LoRA, DoRA, WaveFT methods respectively, and compare and analyze fine-tuning efficiency and performance retention rates

### Week 4 – Scientific Prompt Engineering

#### Core Topics

- **Systematic Prompt Design**: Systematically learn effective prompt design techniques
- **Various Prompt Strategies**: Core techniques that contributed to performance improvement such as role instruction and step-by-step questioning
- **Core Technique Deep Dive**:
  - _Self-Consistency_: Improve accuracy through **multiple solution path exploration** in math problem solving (+17%p improvement on GSM8K benchmark)
  - _Tree-of-Thoughts_: Solve difficult problems through **expansion of thinking** (24 game success rate 9%→74%)
  - _DSPy Framework_: Methodology that automatically generates/combines optimal prompts by "**programming** prompts like code"
  - _Automatic Prompt Engineering (APE)_: Cases such as achieving **93% accuracy** on GSM8K through algorithmic prompt optimization

#### Hands-on/Activities

- **Key Hands-on**: Build **prompt optimization pipeline** using **DSPy**
- **Comparison Analysis**: Automatically generate various prompts using DSPy for given problems and compare performance with manual prompts

### Week 5 – Latest AI Evaluation Systems

#### Core Topics

- **Paradigm Shifts in Evaluation**: Beyond traditional answer-matching evaluation, meta-evaluation using LLMs and experimental benchmarks have emerged
- **New Evaluation Techniques and Benchmarks**:
  - _G-Eval_: **GPT-4-based meta-evaluation** – Automated quality evaluation where LLMs evaluate other LLMs' responses using chain-of-thought
  - _LiveCodeBench_: Automatic code evaluation adopting **online code execution contest format** – Answer verification through test case execution (data contamination prevention)
  - _MMMU_: **Multimodal university-level exam** – **Large-scale multidisciplinary evaluation** set consisting of 11,500 problems across 6 fields and 30 subjects
  - _OmniBench_: **Triple multimodal evaluation** – First **Tri-modal integrated benchmark** measuring ability to understand and reason with images·audio·text **simultaneously**
  - _Humanity's Last Exam (HLE)_: **Comprehensive exam of 2500 questions** created by human experts – **Final exam** testing limitations of existing AI across broad fields including mathematics, humanities, and science
- **Domain-specific Specialized Benchmarks**: **SWE-Bench Verified** (500 verified problems for **software problem solving** based on actual GitHub issues), etc.

#### Hands-on/Activities

- **Key Hands-on**: Apply **LLM-based evaluation** techniques such as G-Eval to identical responses with existing automatic evaluation metrics (BLEU, ROUGE, etc.) and compare evaluation results

### Week 6 – Innovation in Multimodal NLP

#### Core Topics

- **"Any-to-Any" Multimodal Models**: Technology where single models receive various forms of input such as text, images, and audio and generate various forms of output
- **Representative Cases**:
  - _SmolVLM2_ (small 200M-2.2B parameters): Next-generation Vision-Language model that performs **video understanding** with lightweight models
  - _Qwen 2.5 Omni_: Alibaba's multimodal LLM that **integrates conversion** of text·image·audio (supporting all modal input/output with one model)
  - _QVQ-Max_ (formerly QVQ-72B): **Visual reasoning specialized ultra-large model** – 72B scale open-source vision-language model that understands image content and performs reasoning
  - _Real-time multimodal streaming_: Emergence of multimodal LLMs supporting **streaming input/output**
- **Integration of Speech Technology and LLM**:
  - _Voxtral_: Open-source **speech recognition** model with performance exceeding OpenAI Whisper (Realtime ASR)
  - _Orpheus_: TTS supporting **zero-shot speaker synthesis** – Learn speaker voice characteristics with one sentence, read arbitrary sentences

#### Hands-on/Assignment

- **Programming Assignment 2**: Develop **multimodal QA application** responding to image·text·audio mixed input. For example, implement so that when users ask questions with voice, the model finds related images and generates answers combining visual information and text

### Week 7 – Ultra-long Context Processing and Efficient Reasoning

#### Core Topics

- **Ultra-long Context (Long Context) Support**: Models capable of processing extremely long context windows (millions of tokens) have emerged, maintaining consistency in long document summarization and long-term conversations
- **Representative Cases**:
  - _Gemini 2.5 Pro_: Google's next-generation large multimodal model capable of processing up to **million-unit tokens** (enhanced reasoning ability and multimodal understanding compared to previous generation Gemini; research prototype targets 10 million tokens)
  - _Magic LTM-2-Mini_: Experimental model implementing **100 million token** scale context window with economical structure – Ultra-long context processing at 1/1000 cost level compared to Llama at same performance
- **Efficient Long Context Implementation Mechanisms**: Compare various techniques solving memory and speed problems such as Flash **Linear Attention**, **LongRoPE** (long context positional encoding)

#### Hands-on/Activities

- **Key Hands-on**: Implement **RAG-based summarization system** for long context scenarios and compare summarization accuracy and speed with ultra-long context LLMs (Gemini, etc.). (Example: Q&A or summarization of documents of dozens of pages)

### Week 8 – Core Review and Hands-on Reinforcement of Weeks 1-7

#### Core Topics

- Considering **midterm exam period**, organize and enhance understanding of core concepts learned in the previous 7 weeks
- **Key Topic Summary**: Organize key topics such as Transformer and SSM architectures, PyTorch utilization and FlashAttention optimization, latest PEFT techniques, prompt engineering, LLM evaluation methods, multimodal integration by team assignment in presentation format
- **Team-based Activities**:
  - _Quiz League_: Reconfirm key concepts by solving **review quizzes** alternately set by each team while competing and discussing
  - _Mini Project Redesign_: Select one of the assignments or hands-on activities performed in the first half and attempt **reimplementation with new approaches** or performance improvement (e.g., solving same task with different model architectures)

#### Hands-on/Activities

- **Key Hands-on**: Conduct team quiz solving and result sharing, **presentation and feedback sessions** on improved hands-on results
- **Midterm exam score feedback** and future learning direction review

### Week 9 – Advanced RAG Architectures

#### Core Topics

- **Next-generation Retrieval-Augmented Generation**: Structures of advanced RAG systems that integrate large-scale knowledge to improve response accuracy
- **Main Content**:
  - _HippoRAG_: RAG that mimics human **hippocampus** operation principles to reduce vector DB storage space by 25% and enhance **long-term memory** (persistent memory strengthening in information networks)
  - _GraphRAG_: Improve query response precision to 99% by explicitly modeling **associations** between contexts using **knowledge graphs**
  - _Hybrid search_: Multi-strategy search combining latest **dense embedding** techniques (NV-Embed-v2, etc.) and **sparse search techniques** (SPLADE) and graph exploration to secure both **accuracy and speed** in large-scale knowledge bases
- **Production Case Studies**: Analyze **large-scale RAG system** architectures that maintain P95 response latency within 100ms while processing tens of millions of tokens daily

#### Hands-on/Assignment

- **Assignment 3**: Build **Korean enterprise search system** based on GraphRAG. Create Q&A RAG system for given in-house wiki/document database and evaluate search accuracy and response speed

### Week 10 – Innovation in Alignment Techniques

#### Core Topics

- **LLM Output Control Techniques Emerging After RLHF**: New techniques for improving usefulness and safety of LLMs
- **Various Approaches**:
  - _DPO (Direct Preference Optimization)_: Method that directly learns user preferences without separate **reward models** (simplified pipeline compared to RLHF)
  - _Constitutional AI_: Technique that suppresses harmful content generation by AI self-correcting responses according to about 75 **constitutional principles** (applied to Anthropic Claude models)
  - _Process Supervision_: Reward model technique that gives granular feedback on **problem-solving process** (Chain-of-Thought) rather than final answer quality to strengthen correct reasoning process
  - _RLAIF (RL from AI Feedback)_: Approach where **AI evaluates AI** while learning using AI evaluators instead of humans (mimicking human-level evaluation)
- **Open-source Implementation Trends**: Public implementations such as TRL (Transformer Reinforcement Learning) library and OpenRLHF project have emerged, allowing anyone to experiment with latest alignment techniques (3-4× training speed improvement compared to existing DeepSpeed-Chat)

#### Hands-on/Activities

- **Key Hands-on**: Compare and evaluate responses of models fine-tuned with DPO and existing **RLHF** for identical prompts/instructions. (Comparison in aspects such as safety, content quality)

### Week 11 – Production Agent Systems

#### Core Topics

- **Agent Frameworks and Multi-agent Systems**: Technology that utilizes LLMs as multiple entities rather than single QA bots to handle complex tasks
- **Main Content**:
  - _CrewAI_: Role-based **multi-agent collaboration** framework – Assign different specialized roles to multiple LLMs to perform **team-like problem solving**
  - _Mirascope_: Agent development tool ensuring **type-safety** – Strictly manage format and type of prompt I/O through Pydantic data validation
  - _Haystack Agents_: Open-source agent framework specialized for document RAG pipelines – Easily configure search-comprehension chains to implement domain knowledge specialized agents
  - _Low-code integration platforms_: Environment where Flowise AI, LangFlow, n8n, etc. can design prompt workflows and visually integrate various tools through **GUI**
- **Toolformer and LLM Internal Tool Usage**: Approaches that internalize external tool usage capabilities in LLMs themselves: Train by inserting API call signals beforehand so models decide to use **tools** such as calculators or search at necessary moments during responses

#### Hands-on/Activities

- **Key Hands-on**: Implement **automated customer service system** prototype using multi-agent frameworks. For example, have one agent handle **FAQ Q&A**, another agent handle **database queries** or **ticket generation** to practice **orchestration** that handles users' complex demands through collaboration

### Week 12 – AI Regulation and Responsible AI

#### Core Topics

- **AI Governance and Ethical Issues**: Learn impact on industry and developer compliance requirements of world's first comprehensive AI legislation including **EU AI Act** implemented in August 2024
- **Privacy and Safety Enhancement Technologies**: Methodologies for responsible and regulation-compliant LLM service deployment:
  - _Differential privacy_: Prevent **personal information exposure** by introducing Differential Privacy to text embeddings, etc.
  - _Federated Learning_: Utilize frameworks for **collaborative learning locally** so user data doesn't gather at central servers
  - _Homomorphic encryption learning_: Protect sensitive information by performing model training with data itself encrypted
- **Industry-specific Regulation Response Cases**: **Domain-specific NLP solution design** cases such as HIPAA-compliant chatbots in healthcare, GDPR response examples in finance, FERPA-compliant tutor AI in education

#### Hands-on/Assignment

- **Assignment**: Write **suitable LLM service design** for given scenarios according to EU AI Act and other related regulations. Create checklist of measures to take from model development to deployment and present **regulatory compliance** by team

### Week 13 – Latest Research Trends and Paper Reviews

#### Core Topics

- **Survey of Latest Research Results**: Examine currently published latest models and techniques while discussing future directions in rapidly changing NLP field
- **Main Topics**:
  - **Development of ultra-large multimodal LLMs**: Analyze innovative features of cutting-edge models such as GPT-5, **Claude 4.1 Opus**, **Qwen 2.5 Omni**, **QVQ-Max**. For example, GPT-5 shows performance exceeding GPT-4 in **reasoning ability and context expansion**, and Claude 4.1 strengthens response consistency and safety by applying **constitutional AI principles**. Qwen 2.5 Omni and QVQ-Max pioneer new frontiers in multimodal **visual-language reasoning**, demonstrating ability to simultaneously perform image interpretation and complex reasoning.
  - **Renaissance of small language models**: Also cover advances of lightweight **small models (SLM)**. _Gemma 3_ (1B-4B scale) series are attracting attention as ultra-lightweight LLMs optimized to work smoothly on consumer devices, and _Mistral NeMo 12B_ shows specialized performance such as supporting **128K token** long context windows through NVIDIA NeMo optimization. Cases like _MathΣtral 7B_ specialized for specific areas (mathematics) achieving results comparable to GPT-4 are also introduced. These small models are being researched as **alternatives to large models** in terms of specialization and lightweighting.
  - **Evolution of reasoning capabilities**: Examine new attempts by LLMs for complex problem solving. _Long CoT_ reasons with very long **Chain-of-Thought** and performs **backtracking** and error correction when necessary, and _PAL (Program-Aided LM)_ improves numerical calculation or logical reasoning accuracy by combining code execution capabilities. _ReAct_ is a strategy that generates more accurate and factual answers by utilizing **external tools** (calculators, web search, etc.) during reasoning. Additionally, introduce _Thinking Mode_ concept – For example, Qwen series significantly improve performance in complex math·code problems by enabling **internal self-reasoning steps** in models through enable_thinking mode. Also cover cutting-edge approaches like Meta's _Toolformer_ that **embed tool usage capabilities in models** during pre-training so models call external APIs at necessary moments during responses to solve problems.
  - **Deployment and optimization frameworks**: Tools for efficiently **deploying** LLMs in actual service environments are also advancing. For example, _llama.cpp_ enabled execution of large models on CPU with single-file C++ implementation, and _MLC-LLM_ supports **LLM inference on mobile/browsers** using WebGPU. _PowerInfer-2_ is a framework that **maximizes power efficiency** for large model distributed inference, contributing to operational cost reduction.

#### Hands-on/Activities

- **Student latest paper presentations**: Review and present **latest NLP papers** selected by groups and discuss significance, limitations, and application possibilities of the research. For example, by selecting and discussing papers on new benchmarks (MMMU, HLE, etc.) or latest model techniques mentioned above, **comprehensively organize latest technology trends** _(Industry mentors or invited researchers participate in feedback)_

### Week 14 – Final Project Development and MLOps

#### Core Topics

- **Complete Prototype Implementation**: Complete **prototype implementation** of team projects and apply MLOps concepts
- **NLP Model MLOps Concepts**: Introduce model **version management** strategies, A/B testing techniques, **deployment pipeline** design, etc. Also cover methods for building **online learning pipelines** that continuously reflect user feedback in learning, real-time **monitoring and performance drift detection** systems
- **Team Prototype Development**: Each team implements **final models and application prototypes** for selected project topics. Reflect industry datasets or actual user scenarios to increase completeness and demonstrate intermediate results this week
- **Mentor Review Sessions**: Review project progress with invited industry mentors. Receive feedback on appropriateness of model architecture, utilization of latest technologies (e.g., multimodal integration, agent usage, etc.), practicality, etc., and reflect in final improvement direction

#### Hands-on/Activities

- **Key Activities**: Team prototype **demo presentations** (share current performance and remaining tasks) and mentor feedback reflection discussions

### Week 15 – Industry Application Case Analysis and Final Presentations

#### Core Topics

- **Industry Application Case Analysis**: Conclude the course by analyzing **industry cases where latest technologies are applied** and sharing final results of team projects
- **Industry-specific NLP Success Cases**: Introduce **latest application cases of LLM and NLP technologies** in each field such as healthcare, finance, and education. For example, in healthcare, cases where clinical record automation NLP reduced doctor documentation burden from 49% to 27%, in finance, cases where Morgan Stanley's contract analysis bot introduction saved 360,000 hours annually, in education, cases where **customized tutor AI** with multilingual support improved learning efficiency and increased student engagement by 30%. Through these cases, understand **practical impact of latest NLP technologies**
- **Final Project Result Presentations**: Each team presents final project outputs and demonstrates demos. Each team shares developed **model architecture**, core technology application content (e.g., ultra-long context support, multimodal input, agent collaboration, etc.), performance evaluation results and limitations. Receive feedback on practicality and improvement points through Q&A with industry mentors and students
- **Course Comprehensive Discussion**: Finally, **comprehensively organize** content covered in the course and conduct free discussion. Students **reflect on learning content** from week 1 to week 15 and share opinions about most impressive technologies or topics they want to study more. Faculty present **future prospects** (e.g., expected developments after GPT-5, direction of AI-human collaboration, etc.) and advise students to track and utilize latest NLP trends afterwards _(Collect course feedback through surveys)_

#### Hands-on/Activities

- **Final project result presentations**: Team project result presentations and demo demonstrations (sharing model architecture, demonstration results and limitations)
- **Course comprehensive discussion**: Overall summary of course content and Q&A, future prospects brainstorming (student feedback collection and future learning guidance)

## References (Selected Latest Papers and Materials)

### Latest Architectures and Models

- Gu & Dao (2023), _Mamba: Linear-Time Sequence Modeling with Selective State Spaces._
- Peng et al. (2023), _RWKV: Reinventing RNNs for the Transformer Era._
- Lieber et al. (2024), _Jamba: A Hybrid Transformer-Mamba Language Model._
- **(Multimodal LLM)** OpenAI (2025), _GPT-4 Technical Report (Augmentations for GPT-5 Preview)._
- Anthropic (2025), _Claude 4.1 Opus System Card._

### Parameter-Efficient Fine-tuning

- Zhang et al. (2024), _WaveFT: Wavelet-based Parameter-Efficient Fine-Tuning._
- Liu et al. (2024), _DoRA: Weight-Decomposed Low-Rank Adaptation._
- Chen et al. (2024), _VB-LoRA: Vector Bank for Efficient Multi-Task Adaptation._
- Dettmers et al. (2023), _QLoRA: Efficient Finetuning of Quantized LLMs._

### Prompt Engineering and Evaluation

- Khattab et al. (2023), _DSPy: Compiling Declarative Language Model Calls._
- Zhou et al. (2023), _Self-Consistency for Chain-of-Thought._
- Yao et al. (2023), _Tree of Thoughts: Deliberate Problem Solving with Large Language Models._
- Liu et al. (2023), _G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment._
- Jain et al. (2024), _LiveCodeBench: Holistic and Contamination-Free Code Evaluation._

### Knowledge Integration and RAG

- Zhang et al. (2024), _HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs._
- Edge et al. (2024), _GraphRAG: A Modular Graph-Based RAG Approach._
- Chen et al. (2024), _Hybrid Retrieval-Augmented Generation: Best Practices._

### Alignment and Responsible AI

- Rafailov et al. (2023), _Direct Preference Optimization: Your Language Model is Secretly a Reward Model._
- Bai et al. (2022), _Constitutional AI: Harmlessness from AI Feedback._
- OpenAI (2024), _SWE-bench Verified: Real-world Software Engineering Benchmark._
- Phan et al. (2025), _Humanity's Last Exam: The Ultimate Multimodal Benchmark at the Frontier of Knowledge._
- EU Commission (2024), _EU AI Act: Implementation Guidelines._

### Industry Applications and MLOps

- **Healthcare NLP** Market Report 2024–2028 (Markets&Markets).
- **Financial Services AI** Applications 2025 (McKinsey Global Institute).
- **State of AI in Education 2025** (Stanford HAI).
- Cremer & Liu (2025), _PowerInfer-2: Energy-Efficient LLM Inference at Scale._
- **Development Tools:** CrewAI Documentation – _Multi-agent Scenario Implementation Guide_
- DSPy Official Guide – _Prompt DSL Usage Guide_
- OpenRLHF Project – _Open-source RLHF Implementation_
