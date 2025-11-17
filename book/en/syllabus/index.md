# Syllabus

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

| Week | Main Topics and Keywords                                                                                                                                                     | Key Hands-on/Assignments                                                                                                  |
| :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
|  1   | Transformer and Next-Generation Architectures<br/>• Self-Attention Mechanism and Limitations<br/>• Mamba (SSM), RWKV, Jamba                                                  | Transformer Component Implementation<br/>Mamba vs Transformer Performance Comparison<br/>Architecture Complexity Analysis |
|  2   | PyTorch 2.x and Latest Deep Learning Frameworks<br/>• torch.compile Compiler Revolution<br/>• FlashAttention-3 Hardware Acceleration<br/>• AI Agent Frameworks               | torch.compile Performance Optimization<br/>FlashAttention-3 Implementation<br/>AI Agent Framework Comparison              |
|  3   | Modern PEFT Techniques for Efficient Fine-tuning<br/>• LoRA, DoRA, QLoRA<br/>• Advanced PEFT Techniques                                                                      | PEFT Method Comparison Experiment<br/>LoRA/DoRA/QLoRA Performance Evaluation<br/>Memory Efficiency Analysis               |
|  4   | Advanced Prompt Techniques and Optimization<br/>• Prompt Engineering Fundamentals<br/>• Self-Consistency, Tree of Thoughts<br/>• DSPy Framework                              | DSPy-based Automatic Prompt Optimization<br/>Self-Consistency Implementation<br/>Tree of Thoughts Problem Solving         |
|  5   | LLM Evaluation Paradigms and Benchmarks<br/>• Evaluation Paradigm Evolution<br/>• LLM-as-a-Judge (GPTScore, G-Eval, FLASK)<br/>• Specialized and Domain-specific Benchmarks  | G-Eval Implementation<br/>Benchmark Comparison Experiment<br/>Evaluation Bias Analysis                                    |
|  6   | Multimodal NLP Advancements<br/>• Vision-Language Models (LLaVA, MiniGPT-4, Qwen-2.5-Omni)<br/>• Visual Reasoning (QVQ-Max)<br/>• Speech Integration                         | Multimodal QA Application Development<br/>Vision-Language Model Comparison<br/>End-to-end Multimodal System               |
|  7   | Ultra-Long Context Processing and Efficient Inference<br/>• Context Window Revolution (1M+ tokens)<br/>• Attention Mechanism Optimization<br/>• LongRoPE and RAG Integration | FlashAttention-3 Integration<br/>Long Context Processing Comparison<br/>Performance Analysis                              |
|  8   | Core Review and Latest Trends<br/>• Architecture Review<br/>• Latest Model Trends (GPT-5, Gemini 2.5 Pro, Claude 4.1)<br/>• Industry Applications                            | Comprehensive Review<br/>Model Comparison<br/>Industry Case Analysis                                                      |
|  9   | Advanced RAG Systems – HippoRAG, GraphRAG, Hybrid Search Strategies                                                                                                          | Assignment 3: Building Korean Enterprise Search System based on GraphRAG                                                  |
|  10  | Innovation in Alignment Techniques – DPO, Constitutional AI, Process Reward Models                                                                                           | Comparison Practice between DPO and Existing RLHF Techniques                                                              |
|  11  | Production Agent Systems – CrewAI, Mirascope, Type-Safety Development                                                                                                        | Multi-agent Orchestration Implementation                                                                                  |
|  12  | AI Regulation and Responsible AI – EU AI Act, Differential Privacy, Federated Learning                                                                                       | Assignment for Designing Regulation-Compliant AI Systems                                                                  |
|  13  | Ontology and AI – Modeling Reality and Operating it with AI<br/>• Data Science to Decision Science<br/>• Semantic Ontology, GraphRAG<br/>• Kinetic Ontology, Closed-Loop Systems                                                      | Semantic Ontology Modeling<br/>GraphRAG Implementation<br/>Closed-Loop Simulation                                                      |
|  14  | Final Project Development and MLOps                                                                                                                                          | Team Prototype Implementation and Feedback Sessions (Industry Mentor Participation)                                       |
|  15  | Final Project Presentations and Comprehensive Evaluation                                                                                                                     | Team Presentations, Course Content Summary and Future Prospects Discussion                                                |

## Weekly Educational Content

### Week 1 – Transformer and Next-Generation Architectures

#### Core Topics

- **Transformer Architecture**: Self-attention mechanism, encoder-decoder structure, computational complexity $O(N^2)$
- **Mamba Architecture**: Selective State Space Model (SSM), linear time complexity $O(N)$, hardware optimization through selective mechanisms
- **RWKV Architecture**: RNN-Transformer hybrid, parallel training capability, infinite context processing
- **Jamba Architecture**: Hybrid Transformer-Mamba with Mixture-of-Experts (MoE), long context window support, efficiency optimization

#### Hands-on/Activities

- **Core Practice**: Implement basic Transformer components (multi-head self-attention, positional encoding) and compare with Mamba's selective state space mechanisms
- **Architecture Comparison**: Analyze computational complexity and memory usage differences between Transformer ($O(N^2)$) and Mamba ($O(N)$)
- **Performance Evaluation**: Benchmark different architectures on sequence modeling tasks, focusing on long-range dependency learning

### Week 2 – PyTorch 2.x and Latest Deep Learning Frameworks

#### Core Topics

- **PyTorch 2.x Revolution**: `torch.compile` compiler revolution, TorchDynamo, AOTAutograd, PrimTorch, TorchInductor
- **FlashAttention-3**: Hardware acceleration with tiling, TMA, WGMMA, FP8 support, ~2× speed improvement on H100 GPU
- **Hugging Face Transformers Ecosystem**: Model support, quantization, Zero-Build Kernels, `pipeline` API
- **AI Agent Frameworks**: LangGraph, CrewAI, LlamaIndex, Haystack, DSPy for building intelligent agent systems

#### Hands-on/Activities

- **Core Practice**: Implement `torch.compile` performance optimization and FlashAttention-3 integration
- **Framework Comparison**: Compare different AI agent frameworks (LangGraph vs CrewAI vs DSPy) for specific use cases
- **Performance Benchmarking**: Measure speed improvements and memory efficiency gains from latest optimizations

### Week 3 – Efficient Fine-tuning with Modern PEFT Techniques

#### Core Topics

- **PEFT Fundamentals**: Parameter-Efficient Fine-Tuning techniques that achieve 95%+ performance with <1% parameters
- **LoRA (Low-Rank Adaptation)**: Decompose weight matrices into low-rank form, learn only small rank matrices
- **DoRA (Weight-Decomposed LoRA)**: Adaptive fine-tuning through weight decomposition for fine-grained representation learning
- **QLoRA**: 4-bit quantization + LoRA, enabling 65B model fine-tuning on single 48GB GPU
- **Advanced PEFT**: NF4 quantization, double quantization, VB-LoRA, QR-Adaptor techniques

#### Hands-on/Activities

- **Core Practice**: Implement LoRA, DoRA, and QLoRA fine-tuning on Korean sentiment analysis dataset
- **Performance Comparison**: Compare memory usage, training speed, and final performance across different PEFT methods
- **Efficiency Analysis**: Measure parameter reduction ratios and performance retention rates

### Week 4 – Advanced Prompting Techniques and Optimization

#### Core Topics

- **Prompt Engineering Fundamentals**: Role prompting, structured prompting, few-shot vs zero-shot techniques
- **Self-Consistency**: Multiple solution path exploration for improved accuracy (+17% improvement on GSM8K)
- **Tree of Thoughts**: Deliberate problem solving through thought expansion (24 game success rate 9%→74%)
- **DSPy Framework**: Declarative Self-Improving Python, Signature, Module, Optimizer for automated prompt optimization
- **Automated Prompt Engineering**: APE, OPRO techniques for algorithmic prompt optimization

#### Hands-on/Activities

- **Core Practice**: Implement DSPy-based automatic prompt optimization pipeline
- **Technique Comparison**: Compare Self-Consistency, Tree of Thoughts, and automated prompt engineering approaches
- **Performance Evaluation**: Measure accuracy improvements across different prompting strategies on reasoning tasks

### Week 5 – LLM Evaluation Paradigms and Benchmarks

#### Core Topics

- **Evaluation Paradigm Evolution**: Traditional metrics (BLEU/ROUGE) vs meaning-based evaluation (BERTScore/BLEURT) vs LLM-as-a-Judge
- **LLM-as-a-Judge**: GPTScore, G-Eval, FLASK frameworks for automated evaluation using LLMs
- **Specialized Purpose Benchmarks**: LiveCodeBench, EvalPlus, HELM-Code, MMLU-Pro, GPQA, BBH
- **Domain-Specific Benchmarks**: FinBen, AgentHarm, LEXam, CSEDB, MATH, GSM8K
- **Evaluation Bias and Limitations**: Narcissistic bias, verbosity bias, inconsistency, differences from human evaluation

#### Hands-on/Activities

- **Core Practice**: Implement G-Eval and other LLM-based evaluation techniques
- **Benchmark Comparison**: Compare traditional metrics (BLEU/ROUGE) with LLM-as-a-Judge approaches on identical responses
- **Bias Analysis**: Analyze evaluation biases and limitations in different evaluation paradigms

### Week 6 – Multimodal NLP Advancements

#### Core Topics

- **Multimodal Integration**: Text, image, audio, and video processing in unified models
- **Vision-Language Models**: LLaVA, MiniGPT-4, Qwen-2.5-Omni for comprehensive multimodal understanding
- **Visual Reasoning**: QVQ-Max specialized for visual reasoning and logical context understanding
- **Speech Integration**: Voxtral for speech recognition, Orpheus for zero-shot speaker synthesis
- **Real-time Multimodal Streaming**: Streaming input/output capabilities in multimodal LLMs

#### Hands-on/Activities

- **Core Practice**: Implement multimodal QA application with image, text, and audio input
- **Model Comparison**: Compare different vision-language models (LLaVA vs MiniGPT-4 vs Qwen-2.5-Omni)
- **Integration Challenge**: Build end-to-end multimodal system with voice input, image analysis, and text generation

### Week 7 – Ultra-Long Context Processing and Efficient Inference

#### Core Topics

- **Context Window Revolution**: From kilobytes to megabytes - quantitative leap in context processing capabilities
- **2025 Flagship Models**: GPT-5, Gemini 2.5 Pro (1M tokens), Claude Sonnet 4 (1M tokens), Llama 4 (10M tokens), LTM-2-Mini (100M tokens)
- **Attention Mechanism Optimization**: FlashAttention I/O bottleneck optimization, Linear Attention approximation, Ring Attention distributed processing
- **Positional Encoding Extension**: LongRoPE for extending context windows beyond 2M tokens with minimal fine-tuning
- **RAG vs Ultra-Long Context**: Integration paradigms, HippoRAG as long-term memory system

#### Hands-on/Activities

- **Core Practice**: Implement FlashAttention-3 integration and LongRoPE context extension
- **RAG vs Long Context**: Compare RAG-based summarization with ultra-long context LLMs on long documents
- **Performance Analysis**: Measure cost, latency, and accuracy trade-offs in long context processing

### Week 8 – Core Review and Latest Trends

#### Core Topics

- **Architecture Review**: Transformer vs SSM architectures, computational complexity analysis, performance trade-offs
- **Optimization Techniques**: FlashAttention optimization, PEFT methods (LoRA, DoRA, QLoRA), efficiency improvements
- **Advanced Techniques**: Prompt engineering, LLM evaluation paradigms, multimodal integration, long context processing
- **Latest Model Trends**: GPT-5, Gemini 2.5 Pro, Claude 4.1, Qwen 2.5 series - comprehensive model comparison
- **Industry Applications**: Medical, legal, financial field applications, real-world deployment considerations

#### Hands-on/Activities

- **Core Practice**: Comprehensive review of key concepts through hands-on reinforcement
- **Model Comparison**: Compare latest models across different dimensions (performance, cost, capabilities)
- **Industry Case Analysis**: Analyze real-world applications and deployment strategies

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

### Week 13 – Ontology and AI: Modeling Reality and Operating it with AI

#### Core Topics

- **Paradigm Shift: From Data Science to Decision Science**:
  - The "Data-Rich, Decision-Poor" problem and the "last mile" gap
  - Limitations of Data Science (DS): Remaining at prediction and insight as "dashboard" builders
  - Goal of Decision Science (DSci): "Pilots" who prescribe optimal actions and create business impact
  - The need to convert expert "Tacit Knowledge" into "Explicit Models" that AI can understand
  - "Ontology-First" strategy: Modeling the semantics and logic of reality before data collection
- **Modeling Reality: Semantic Ontology (Semantic Layer)**:
  - Semantic Layer: A "Digital Twin" that reflects an organization's real world
  - Three core components of semantic ontology: Object Types, Properties, Link Types
  - Semantic Digital Twin: Modeling that integrates "meaning" and "context" beyond simple data replication
  - Root cause of LLM hallucinations: Limitations of "flat models" and the need for explicit semantics
- **Integrating AI: Grounding and GraphRAG**:
  - Two faces of AI: Symbolic AI (logical reasoning) vs Statistical AI (LLM, statistical prediction)
  - Neuro-Symbolic AI: Complementary combination of both approaches
  - Three-step "Grounding" governance: Data grounding (input control), Logic grounding (processing control), Action grounding (output control)
  - GraphRAG: Beyond standard RAG to knowledge graph-based multi-hop reasoning, improving precision by up to 35%
- **Operating Reality: Kinetic Ontology (Kinetic Layer)**:
  - Kinetic Ontology: Explicitly modeling "Verbs" (Actions) of reality in addition to semantic ontology ("Nouns")
  - "Writeback": Mechanism that reflects AI decisions into actual operational systems
  - Difference between analytical and operational systems: Automating the "last mile" through Writeback
  - "Closed-Loop" decision-making: Complete automation cycle of Read-Decide-Write-Feedback-Learn
  - AI Operating System: Enterprise-wide AI platform integrating semantic and kinetic layers

#### Hands-on/Activities

- **Core Practice**: Semantic ontology modeling exercise - Define object types, properties, and link types for a given business domain (e.g., university hospital, manufacturing) and create an ontology schema
- **GraphRAG Implementation**: Build a RAG system using knowledge graphs - Implement a hybrid search system combining vector search and graph traversal
- **Closed-Loop Simulation**: Implement a simple decision-making system prototype connecting semantic layer (read) and kinetic layer (write)

### Week 14 – Final Project Development and MLOps

#### Core Topics

- **Survey of Latest Research Results**: Examine currently published latest models and techniques while discussing future directions in rapidly changing NLP field
- **Main Topics**:
  - **Development of ultra-large multimodal LLMs**: Analyze innovative features of cutting-edge models such as GPT-5, **Claude 4.1 Opus**, **Qwen 2.5 Omni**, **QVQ-Max**. For example, GPT-5 shows performance exceeding GPT-4 in **reasoning ability and context expansion**, and Claude 4.1 strengthens response consistency and safety by applying **constitutional AI principles**. Qwen 2.5 Omni and QVQ-Max pioneer new frontiers in multimodal **visual-language reasoning**, demonstrating ability to simultaneously perform image interpretation and complex reasoning.
  - **Renaissance of small language models**: Also cover advances of lightweight **small models (SLM)**. _Gemma 3_ (1B-4B scale) series are attracting attention as ultra-lightweight LLMs optimized to work smoothly on consumer devices, and _Mistral NeMo 12B_ shows specialized performance such as supporting **128K token** long context windows through NVIDIA NeMo optimization. Cases like _MathΣtral 7B_ specialized for specific areas (mathematics) achieving results comparable to GPT-4 are also introduced. These small models are being researched as **alternatives to large models** in terms of specialization and lightweighting.
  - **Evolution of reasoning capabilities**: Examine new attempts by LLMs for complex problem solving. _Long CoT_ reasons with very long **Chain-of-Thought** and performs **backtracking** and error correction when necessary, and _PAL (Program-Aided LM)_ improves numerical calculation or logical reasoning accuracy by combining code execution capabilities. _ReAct_ is a strategy that generates more accurate and factual answers by utilizing **external tools** (calculators, web search, etc.) during reasoning. Additionally, introduce _Thinking Mode_ concept – For example, Qwen series significantly improve performance in complex math·code problems by enabling **internal self-reasoning steps** in models through enable*thinking mode. Also cover cutting-edge approaches like Meta's \_Toolformer* that **embed tool usage capabilities in models** during pre-training so models call external APIs at necessary moments during responses to solve problems.
  - **Deployment and optimization frameworks**: Tools for efficiently **deploying** LLMs in actual service environments are also advancing. For example, _llama.cpp_ enabled execution of large models on CPU with single-file C++ implementation, and _MLC-LLM_ supports **LLM inference on mobile/browsers** using WebGPU. _PowerInfer-2_ is a framework that **maximizes power efficiency** for large model distributed inference, contributing to operational cost reduction.

#### Hands-on/Activities

- **Student latest paper presentations**: Review and present **latest NLP papers** selected by groups and discuss significance, limitations, and application possibilities of the research. For example, by selecting and discussing papers on new benchmarks (MMMU, HLE, etc.) or latest model techniques mentioned above, **comprehensively organize latest technology trends** _(Industry mentors or invited researchers participate in feedback)_

### Week 15 – MLOps and Industry Application Case Analysis

#### Core Topics

- **NLP Model MLOps Concepts**: Introduce model **version management** strategies, A/B testing techniques, **deployment pipeline** design, etc. Also cover methods for building **online learning pipelines** that continuously reflect user feedback in learning, real-time **monitoring and performance drift detection** systems
- **Industry Application Case Analysis**: Conclude the course by analyzing **industry cases where latest technologies are applied** and sharing final results of team projects
- **Industry-specific NLP Success Cases**: Introduce **latest application cases of LLM and NLP technologies** in each field such as healthcare, finance, and education. For example, in healthcare, cases where clinical record automation NLP reduced doctor documentation burden from 49% to 27%, in finance, cases where Morgan Stanley's contract analysis bot introduction saved 360,000 hours annually, in education, cases where **customized tutor AI** with multilingual support improved learning efficiency and increased student engagement by 30%. Through these cases, understand **practical impact of latest NLP technologies**
- **Course Comprehensive Discussion**: Finally, **comprehensively organize** content covered in the course and conduct free discussion. Students **reflect on learning content** from week 1 to week 15 and share opinions about most impressive technologies or topics they want to study more. Faculty present **future prospects** (e.g., expected developments after GPT-5, direction of AI-human collaboration, etc.) and advise students to track and utilize latest NLP trends afterwards _(Collect course feedback through surveys)_

#### Hands-on/Activities

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
