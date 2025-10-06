# Deep Learning for Natural Language Processing (131307379A)

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

| Week | Main Topics and Keywords                                                                                                                                                                     | Key Hands-on/Assignments                                                                                                              |
| :--: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
|  1   | **Transformer and Next-Generation Architectures**<br/>• Self-Attention Mechanism and Limitations<br/>• **Mamba (SSM)**, **RWKV**, **Jamba**                                                  | **Transformer Component Implementation**<br/>**Mamba vs Transformer Performance Comparison**<br/>**Architecture Complexity Analysis** |
|  2   | **PyTorch 2.x and Latest Deep Learning Frameworks**<br/>• **torch.compile Compiler Revolution**<br/>• **FlashAttention-3 Hardware Acceleration**<br/>• **AI Agent Frameworks**               | **torch.compile Performance Optimization**<br/>**FlashAttention-3 Implementation**<br/>**AI Agent Framework Comparison**              |
|  3   | **Modern PEFT Techniques for Efficient Fine-tuning**<br/>• **LoRA**, **DoRA**, **QLoRA**<br/>• **Advanced PEFT Techniques**                                                                  | **PEFT Method Comparison Experiment**<br/>**LoRA/DoRA/QLoRA Performance Evaluation**<br/>**Memory Efficiency Analysis**               |
|  4   | **Advanced Prompt Techniques and Optimization**<br/>• **Prompt Engineering Fundamentals**<br/>• **Self-Consistency**, **Tree of Thoughts**<br/>• **DSPy Framework**                          | **DSPy-based Automatic Prompt Optimization**<br/>**Self-Consistency Implementation**<br/>**Tree of Thoughts Problem Solving**         |
|  5   | **LLM Evaluation Paradigms and Benchmarks**<br/>• **Evaluation Paradigm Evolution**<br/>• **LLM-as-a-Judge** (GPTScore, G-Eval, FLASK)<br/>• **Specialized and Domain-specific Benchmarks**  | **G-Eval Implementation**<br/>**Benchmark Comparison Experiment**<br/>**Evaluation Bias Analysis**                                    |
|  6   | **Multimodal NLP Advancements**<br/>• **Vision-Language Models** (LLaVA, MiniGPT-4, Qwen-2.5-Omni)<br/>• **Visual Reasoning** (QVQ-Max)<br/>• **Speech Integration**                         | **Multimodal QA Application Development**<br/>**Vision-Language Model Comparison**<br/>**End-to-end Multimodal System**               |
|  7   | **Ultra-Long Context Processing and Efficient Inference**<br/>• **Context Window Revolution** (1M+ tokens)<br/>• **Attention Mechanism Optimization**<br/>• **LongRoPE and RAG Integration** | **FlashAttention-3 Integration**<br/>**Long Context Processing Comparison**<br/>**Performance Analysis**                              |
|  8   | **Core Review and Latest Trends**<br/>• **Architecture Review**<br/>• **Latest Model Trends** (GPT-5, Gemini 2.5 Pro, Claude 4.1)<br/>• **Industry Applications**                            | **Comprehensive Review**<br/>**Model Comparison**<br/>**Industry Case Analysis**                                                      |
|  9   | **Advanced RAG Systems** – HippoRAG, GraphRAG, Hybrid Search Strategies                                                                                                                      | Assignment 3: Building **Korean Enterprise Search System** based on GraphRAG                                                          |
|  10  | **Innovation in Alignment Techniques** – DPO, Constitutional AI, Process Reward Models                                                                                                       | Comparison Practice between DPO and Existing RLHF Techniques                                                                          |
|  11  | **Production Agent Systems** – CrewAI, Mirascope, Type-Safety Development                                                                                                                    | Multi-agent Orchestration Implementation                                                                                              |
|  12  | **AI Regulation and Responsible AI** – EU AI Act, Differential Privacy, Federated Learning                                                                                                   | Assignment for Designing Regulation-Compliant AI Systems                                                                              |
|  13  | **Latest Research Trends** – Small Language Models (Gemma 3, Mistral NeMo), Enhanced Reasoning (Long CoT, PAL)                                                                               | Student Presentations of Latest Papers and Comprehensive Discussion                                                                   |
|  14  | Final Project Development and MLOps                                                                                                                                                          | Team Prototype Implementation and Feedback Sessions **(Industry Mentor Participation)**                                               |
|  15  | Final Project Presentations and Comprehensive Evaluation                                                                                                                                     | Team Presentations, Course Content Summary and Future Prospects Discussion                                                            |

## Table of Contents

```{tableofcontents}

```
