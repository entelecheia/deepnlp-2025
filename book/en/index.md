# Deep Learning for Natural Language Processing (131307379A)

## Overview

In recent years, natural language processing (NLP) research has undergone a massive transformation. Large language models (LLMs) have dramatically improved the ability to generate and understand text, revolutionizing various application domains such as translation, question answering, and summarization. In 2024-2025, multimodal models like GPT-4o and Gemini 1.5 Pro that can process text, images, and audio have emerged, greatly expanding the scope of LLM applications. Particularly noteworthy is the emergence of **new architectures beyond Transformer**. For example, **Mamba**, a state space model (SSM), can process up to 1 million tokens with linear O(n) complexity and provides 5x faster inference speed than Transformer.

This course reflects these latest developments to learn **hands-on deep learning and NLP techniques**. Students first learn to use PyTorch and Hugging Face tools, then directly experience fine-tuning of Transformer-based models and **latest SSM architectures**, prompt engineering, retrieval-augmented generation (RAG), reinforcement learning from human feedback (RLHF), and agent framework implementation. Additionally, we cover **latest parameter-efficient fine-tuning techniques** (WaveFT, DoRA, VB-LoRA) and **advanced RAG architectures** (HippoRAG, GraphRAG), and finally complete a model that solves real problems by integrating learned content through team projects.

This course is designed for third-year undergraduate level and assumes completion of prerequisite course _Language Models and Natural Language Processing (131107967A)_. Through team projects, students challenge real problem-solving using **Korean corpora**, and in the final project phase, we provide opportunities to work with actual industry datasets and receive feedback from industry experts, considering **industry-academia collaboration**.

### Learning Objectives

- Understand the role and limitations of large language models in modern NLP and utilize related tools (PyTorch, Hugging Face, etc.).

- Understand the principles and trade-offs of **State Space Models** (e.g., Mamba, RWKV) along with Transformer and other latest architectures.

- Apply fine-tuning to pre-trained models or latest **parameter-efficient fine-tuning** methods like **WaveFT, DoRA, VB-LoRA**.

- Learn methods to systematically optimize prompts using **prompt engineering techniques** and **DSPy framework**.

- Understand the evolution of evaluation metrics (G-Eval, LiveCodeBench, etc.) and the importance of human evaluation, and learn latest alternatives to RLHF such as **DPO (Direct Preference Optimization)**.

- Design and implement **advanced RAG (Retrieval-Augmented Generation)** architectures like **HippoRAG, GraphRAG** and hybrid search strategies.

- Understand AI regulatory frameworks like **EU AI Act** and acquire methodologies for implementing responsible AI systems.

- Track latest research trends to discuss **advantages and disadvantages of latest technologies** such as multimodal LLMs, small language models (SLM), state space models (SSM), and mixture of experts (MoE).

- Understand the characteristics and challenges of Korean NLP and develop application capabilities through hands-on practice using **Korean corpora**.

- Strengthen collaboration and practical problem-solving capabilities through team projects and gain project experience connected to industry.

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

## Table of Contents

```{tableofcontents}

```
