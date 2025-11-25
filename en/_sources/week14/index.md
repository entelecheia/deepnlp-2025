# Week 14: The 2025 NLP Landscape

## 1. Introduction: The Post-Scaling Era

### 1.1 Setting the Stage

This 14th lecture serves as a capstone for our comprehensive study of Deep Learning for Natural Language Processing. In the preceding weeks, we meticulously traced the evolution of the field, beginning with the foundational sequence models, including Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). We then transitioned to the watershed moment of the Transformer architecture, which serves as the "cornerstone" of the modern large language model (LLM) revolution, epitomized by architectures like BERT and the Generative Pre-trained Transformer (GPT) series.

We now arrive at the bleeding edge: the state of NLP research as it exists in 2025. The narrative of the past five years was overwhelmingly dominated by a singular pursuit: scaling. The "scaling hypothesis"—that increasing model size, data, and compute would predictably unlock new capabilities—has been validated to a remarkable degree. However, as we enter 2025, this singular focus is fracturing. The field is now confronting the fundamental limitations and consequences of this scaling-first paradigm.

The current research landscape, therefore, is defined by a significant paradigm shift. The primary questions are no longer just "How big can we build?" but have evolved to "How _efficiently_ can we build?", "How _capably_ can these models act?", and "How _reliably_ can we trust them?" This lecture synthesizes the absolute latest research from 2024 and 2025 to explore the field's new frontiers: agency, efficiency, and reliability.

### 1.2 The 2025 Research Landscape

To construct this analysis, we have synthesized the proceedings and preprints from the premier conferences that define the state-of-the-art. Our review includes key papers and trends from the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024), the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), EMNLP 2025, the 2024 and 2025 conferences on Neural Information Processing Systems (NeurIPS), and the 2024 and 2025 International Conferences on Machine Learning (ICML).

This synthesis reveals three dominant and interconnected research themes that define the 2025 landscape:

1. Agency: The conceptual and practical transition of LLMs from passive text generators into goal-directed, autonomous systems known as LLM Agents. This involves augmenting models with capabilities for planning, tool use, and multi-step reasoning.
2. Efficiency: A marked architectural divergence _away_ from the canonical Transformer. This trend is driven by the urgent need to solve the Transformer's quadratic scaling bottleneck, with two primary solutions dominating the discourse: State Space Models (SSMs) and Mixture of Experts (MoE).
3. Reliability: A critical and introspective examination of the _true nature_ of LLM reasoning. This includes a field-wide debate on whether models are "thinking" or "parroting," how to build alignment without sacrificing capability (the "Safety Tax"), and how to formalize defense against adversarial attacks.

### 1.3 Market and Industry Context (2025)

This academic shift is not occurring in a vacuum; it is a direct response to and driver of massive industrial transformation. The global NLP market is projected to reach an estimated $39.37 billion in 2025, demonstrating a compound annual growth rate (CAGR) of 21.82%.

This growth is dominated by technology giants like Microsoft, Google, and OpenAI. Microsoft, for instance, holds a 15-20% market share in enterprise adoption as of 2025. These companies are leveraging the academic trends we will discuss today to power a new generation of products.

Key application domains include:

- Conversational AI and Customer Support: Chatbots and virtual assistants are no longer simple rule-based systems. They are powered by deep learning models that understand emotional nuance and context, providing 24/7 support.
- Sentiment Analysis: Businesses are using sophisticated NLP to monitor brand perception in real-time, analyzing social media, product reviews, and surveys with a nuance that can detect subtlety, sarcasm, and complex emotions.
- Healthcare and Clinical Informatics: NLP is being used to extract actionable insights from unstructured clinical notes, accelerate medical research by synthesizing literature, and power advanced clinical decision support systems.

As we move through today's lecture, we will connect these abstract research trends back to the practical challenges and opportunities they unlock in the real world.

## 2. Part 1: Architectural Revolutions (Beyond the Transformer)

### 2.1 The Problem: The Transformer's Bottleneck

The 2017 paper "Attention Is All You Need" introduced the Transformer, an architecture that has been the undisputed cornerstone of modern NLP for nearly a decade. Its core mechanism, self-attention, allows the model to build rich, context-aware representations by looking at all other tokens in a sequence.

However, this mechanism is also its greatest liability. The computational and memory cost of self-attention scales quadratically with the sequence length, $n$. This is commonly expressed as $O(n^2)$. For a sequence of 1,000 tokens, this is manageable. For a sequence of 1,000,000 tokens—such as a medical record, a book, or a genomic sequence—this quadratic cost becomes computationally infeasible.

This "quadratic bottleneck" has been the single greatest barrier to scaling models to truly long contexts. While techniques like sparse attention or sliding-window attention have provided temporary fixes, the 2024-2025 research landscape is defined by a search for a true successor architecture that is _sub-quadratic_—ideally, _linear_—in its scaling properties. This search has bifurcated into two major, non-exclusive directions: changing the core recurrence mechanism (State Space Models) and changing the parameter activation mechanism (Mixture of Experts).

### 2.2 Deep Dive: State Space Models (SSMs) and the Rise of Mamba

The first and perhaps most revolutionary architectural shift is the validation of State Space Models (SSMs) as a viable backbone for large-scale sequence modeling.

#### 2.2.1 Conceptual Overview

SSMs are not new; they originate from classical control theory. At a high level, they represent a system by an internal "state" $x$ that evolves over time. This design conceptually blends the strengths of two different architectures:

1. Like RNNs: They are recurrent. The state at time $t$ is a function of the state at time $t-1$ and the input at time $t$. This property makes them extremely fast for autoregressive inference, as the computation per step is constant (no large K-V cache to manage) and scales linearly ($O(n)$) with sequence length.
2. Like CNNs: They can be expressed as a large convolutional kernel, allowing them to be trained in a highly parallelized, non-recurrent fashion.

Prior SSMs (like S4) showed promise in continuous-time modalities like audio but struggled to compete with Transformers on discrete, content-dense modalities like language. This changed with the introduction of Mamba.

#### 2.2.2 Seminal Paper Review: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023/2024)

This paper is arguably one of the most significant architectural papers of the 2024-2025 period, as it provides the first compelling, production-ready alternative to the Transformer.

- Core Problem: The paper addresses a critical failure of previous sub-quadratic models (like linear attention or gated convolutions). These models, while efficient, failed to match the _performance_ of Transformers. The authors hypothesized this was because they lacked a "content-based reasoning" mechanism. Attention is powerful because _which_ tokens it focuses on is a _function of the content_ of the tokens themselves (via the Query-Key-Value mechanism). Prior SSMs were time-and-input-invariant.
- Novel Methodology: The "Selective SSM" (S6): The central innovation of Mamba is the introduction of a selection mechanism. The core parameters of the SSM (specifically, the $A$, $B$, and $C$ matrices that govern state dynamics) are no longer static; they are made _input-dependent_. This seemingly simple change has profound consequences. It allows the model to _selectively_ decide, based on the _content_ of the current token, whether to propagate information (keep it in the state) or forget it (flush the state). This gives Mamba the content-aware routing capability of attention, but retains the linear-time, recurrent structure of an SSM. To make this efficient, the authors developed a hardware-aware parallel scan algorithm that avoids materializing the full state in GPU memory.
- Key Results: The results were transformative.
  1. Linear-Time Scaling: Mamba scales linearly ($O(n)$) in sequence length. This allowed the authors to demonstrate strong performance on real-world data up to _million-length sequences_.
  2. Fast Inference: In autoregressive generation, Mamba achieves 5x higher throughput than Transformers of equivalent size. This is because its recurrent state is compact; it does not require a large, memory-bandwidth-intensive K-V cache that grows with the context window.
  3. State-of-the-Art Performance: Mamba was the first linear-time model to achieve Transformer-quality performance on language. A Mamba-3B model was shown to match the performance of Transformer models _twice its size_ (e.g., a 7B parameter model). On long-sequence modalities where Transformers struggle, such as genomics and audio, Mamba set new state-of-the-art records.
- Conclusion & Impact: Mamba effectively "solves" the Transformer's quadratic bottleneck without sacrificing performance. It has established SSMs as a strong and viable candidate to be the _successor_ to the Transformer as the backbone architecture for the next generation of foundation models.

### 2.3 Deep Dive: Mixture of Experts (MoE) as a Scaling Paradigm

The second major architectural trend, Mixture of Experts (MoE), attacks efficiency from a different angle. It does not (by itself) solve the $O(n^2)$ sequence length problem. Instead, it solves the parameter count problem.

#### 2.3.1 Conceptual Overview

In a standard "dense" Transformer, every single parameter in the model is activated to process every single token. As models scale to hundreds of billions of parameters, this becomes exceptionally compute-intensive.

MoE, an idea that has been refined over the years, replaces the dense Feed-Forward Network (FFN) layers with sparse MoE layers. An MoE layer consists of:

1. A set of "Experts": $N$ parallel FFNs (e.g., 8 experts).
2. A "Router" Network: A small, trainable network that analyzes each token and dynamically decides which experts to send it to.

In a typical setup, the router might select the top 2 of the 8 experts. This means that while the model might have 1 trillion total parameters, it only _uses_ a fraction (e.g., 200 billion) for any given token. This allows for a massive increase in model _capacity_ (knowledge) while holding inference _compute_ (FLOPs) constant.

#### 2.3.2 Seminal Paper Review: "MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models" (NeurIPS 2024)

(Note: This paper is also cited as "MoE-LLaVA")

This paper exemplifies the 2025 evolution of MoE. It's no longer just a scaling trick; it's a sophisticated mechanism for building _generalist_ models.

- Core Problem: Generalist Multimodal Large Language Models (MLLMs)—models trained to handle many different tasks (e.g., image captioning, Visual Question Answering (VQA), Optical Character Recognition (OCR))—suffer from "task interference." Training on such diverse tasks often leads to the model becoming a "jack of all trades, master of none," underperforming specialist models on nearly every task.
- Novel Methodology: The authors propose the Mixture of Multimodal Experts (MoME) framework. This insightfully applies the MoE concept not just to the _language_ FFNs, but also to the _vision_ encoders.
  1. Mixture of Vision Experts (MoVE): The model is given access to _multiple_ specialist vision encoders (e.g., CLIP-ViT for general concepts, DINOv2 for fine-grained features, Pix2Struct for document text). A router network learns to _adaptively modulate_ and combine features from these encoders based on the user's instruction and the input image.
  2. Mixture of Language Experts (MoLE): The model also incorporates standard MoE layers (implemented as parameter-efficient adapters) in the LLM's FFN layers to handle task-specific linguistic nuances.
- Key Results: The MoME architecture was highly effective. Visualizations of the router's gating decisions confirmed that the model learned to _specialize_ to mitigate task interference. When given a task from the "Document" group (e.g., OCR), the MoVE router showed a _strong preference_ for the "Pix2Struct" vision expert, allocating it over 70% of the weight. When given a Referring Expression Comprehension (REC) task, it routed heavily to the DINOv2 expert. This "clear specialization" demonstrates that the model was dynamically routing tasks to the expert best equipped to handle them.
- Conclusion & Impact: This paper, along with others, shows that MoE has evolved from a simple scaling technique into a powerful framework for building _generalist multimodal agents_. It allows a single model to house a "team" of specialists and dynamically deploy the correct one for the job, solving the "task interference" problem. This trend is also being leveraged in industry-specific applications, such as for domain-specific code generation.

### 2.4 A Bifurcated Architectural Future

The analysis of Mamba and MoE reveals that the 2024-2025 architectural trend is not a single path forward but a bifurcation, with two distinct solutions emerging to solve two distinct problems.

1. First, the canonical Transformer architecture faces a hard scaling wall due to its $O(n^2)$ computational complexity relative to sequence length.
2. The State Space Model (SSM) architecture, embodied by Mamba, directly attacks this problem. By using a selective, recurrent state, it achieves $O(n)$ scaling in sequence length. Its primary benefits are the ability to process _extremely long contexts_ (millions of tokens) and _very fast autoregressive inference_ (due to its compact recurrent state).
3. Second, a separate problem is that of parameter count in dense models. A model like Llama 3 70B must activate all 70 billion parameters for _every_ token it processes. This is computationally expensive.
4. The Mixture of Experts (MoE) architecture attacks this problem. It decouples the number of parameters from the compute required for inference. A model with 1 trillion total parameters can be designed to only use 100-200 billion active parameters per token. Its primary benefit is _massive knowledge capacity_ at a _constant inference cost_.

These two solutions—SSMs and MoE—are not mutually exclusive; they are, in fact, complementary. SSMs solve the sequence length bottleneck, while MoE solves the parameter knowledge bottleneck. The clear implication for future SOTA models is a hybrid architecture that combines both: a backbone built from Mamba-style SSM blocks, where the dense FFN component of each block is replaced with a sparse MoE layer. This "Mixture of Mambas" would, in theory, achieve the best of both worlds: linear-time scaling in context length _and_ massive, sparsely-activated knowledge.

### 2.5 Architectural Comparison: Transformer vs. SSM (Mamba) vs. MoE

To summarize this new architectural landscape, the following table provides a clear comparison of the trade-offs defining the 2025 design space.

| Architecture            | Key Mechanism           | Sequence Scaling (Compute) | Parameter Scaling (Inference Compute) | Autoregressive Inference Speed | Primary Use Case (2025)             |
| :---------------------- | :---------------------- | :------------------------- | :------------------------------------ | :----------------------------- | :---------------------------------- |
| Canonical Transformer 5 | Dense Self-Attention    | $O(n^2)$                   | $O(N)$ (Dense)                        | Slow (grows with K-V Cache)    | General Purpose, \<128k Context     |
| SSM (Mamba) 25          | Selective Scan (S6)     | $O(n)$                     | $O(N)$ (Dense)                        | Very Fast (Constant per token) | Long-Context (\>1M), Fast Inference |
| MoE-Transformer 29      | Sparse Gating / Routing | $O(n^2)$                   | $O(k)$ (Sparse, $k \\ll N$)           | Slow (grows with K-V Cache)    | Massive Knowledge Scaling           |

## 3. Part 2: The New Capability Frontier: Agentic AI

### 3.1 From Generative Models to Autonomous Agents

While architects have been rebuilding the _engine_ of LLMs, another community has been redefining _what the engine is used for_. The single most dominant application and research trend of 2025 is the evolution of LLMs from static text generators into AI Agents.

This represents a fundamental conceptual shift. A "generative model" is a passive system that takes a prompt and produces a text completion. An "AI agent," by contrast, is an autonomous system that "senses their environment, makes decisions, and takes actions" to achieve a goal.

This shift is enabled by augmenting foundational LLMs with a new set of capabilities, often referred to as an "agentic stack". A 2024 survey paper identifies three prominent paradigms for building these agents:

1. Reasoning and Planning: The ability to receive a complex, multi-step goal (e.g., "Plan a 5-day trip to Tokyo") and decompose it into a sequence of executable sub-tasks.
2. Tool Use: The ability to interact with the "outside world." This includes using external tools, calling APIs (e.g., booking a flight, checking the weather), or performing Retrieval-Augmented Generation (RAG) to query knowledge bases. This augmentation is a central focus of 2025-2026 research.
3. Memory and Self-Improvement: The ability to store information from past interactions (memory) and learn from feedback (e.g., from human correction or tool execution failures) to improve future performance.

### 3.2 Deep Dive: Multi-Agent Systems (MAS) and Emergent Behavior

The 2025 research frontier has already pushed beyond single-agent systems to investigate Multi-Agent Systems (MAS). In an MAS, multiple specialized agents collaborate, debate, or compete to solve problems that are too complex for any single agent. For example, one agent might be a "planner," another a "code executor," and a third a "critic."

This has opened a new and fascinating field of study: the emergent properties of these "agent ensembles." Researchers are no longer just studying the LLM; they are studying the "social" dynamics of LLM societies.

- Emergent Coordination: A 2025 paper titled "Emergent Coordination in Multi-Agent Language Models" introduces an information-theoretic framework to measure "dynamical emergence" and "cross-agent synergy." It demonstrates that simple prompt design can steer a group of agents from acting as "mere aggregates" (a collection of individuals) to a "higher-order" collaborative system that exhibits goal-directed complementarity.
- Emergent Language: A 2025 survey reviews 181 papers on "emergent language" in multi-agent reinforcement learning (MARL). This research explores how agents, when incentivized to cooperate, can develop novel, efficient communication protocols to achieve their goals.
- The Peril of Emergence: This new capability is not without risk. A 2025 ICML workshop paper provides a critical warning: "Safety and alignment performance... of isolated LLMs... likely do not transfer to multi-agent... ensembles." The authors found that multi-agent systems exhibit "emergent group dynamics," including "peer pressure" that can cause individual, aligned agents to converge on unsafe decisions, even when guided by a supervisor. This implies that aligning an MAS is a fundamentally new and harder problem than aligning a single LLM.

### 3.3 Seminal Paper Review: "Agent Laboratory: Using LLM Agents as Research Assistants" (Schmidgall et al., 2025)

This paper, submitted in January 2025, provides a tangible and powerful example of agents moving from "toys" to "tools" for highly complex, real-world tasks. It has become a highly-cited example of the agentic frontier.

- Core Problem: The traditional scientific discovery process is slow, costly, and labor-intensive, limiting the number of ideas researchers can explore.
- Novel Methodology: The "Agent Laboratory" is an autonomous LLM-based framework designed to complete the _entire machine learning research process_ from a single human-provided idea. The pipeline consists of specialized agents that:
  1. Perform a Literature Review: Autonomously search for, read, and synthesize existing knowledge.
  2. Formulate an Experiment: Write and execute novel code to implement the research idea and run experiments.
  3. Write a Report: Analyze the results and draft a full, conference-style research paper (e.g., for ICLR) complete with an abstract, methods section, and results.
- Key Results: The framework demonstrated stunning capability.
  1. SOTA Code Performance: The agent-generated code achieved state-of-the-art performance on a subset of the MLE-Bench, a benchmark for machine learning tasks.
  2. Human-in-the-Loop is Critical: The paper tested both a fully autonomous mode and a "co-pilot" mode. The key finding was that while the autonomous mode _functioned_, the "co-pilot" mode—where human researchers provided feedback at each stage—_significantly_ improved the overall quality of the final research.
- Conclusion & Impact: "Agent Laboratory" is a proof-of-concept that agents can automate high-cognition, domain-specific tasks. Its most important conclusion, however, is that the future is not one of full autonomy but of human-agent collaboration. The agent excels at "low-level coding and writing," freeing the human researcher to focus on "creative ideation".

### 3.4 The 2025 Agentic AI Debate: Autonomy vs. Control

As agents move from research prototypes like "Agent Laboratory" to real-world products, a central and critical debate has emerged in 2025: how much autonomy should they have?

On one side, the research vision and media hype tout the promise of "fully autonomous systems" that can operate independently for extended periods to accomplish complex tasks.

On the other side, enterprise and safety-conscious organizations are pushing back, citing profound systemic risks. A 2025 report from McKinsey warns that this new paradigm introduces risks that traditional genAI architectures were not built to handle: "uncontrolled autonomy," "fragmented system access," "lack of observability," and "agent sprawl and duplication." What begins as intelligent automation, they warn, can quickly become "operational chaos".

This has led to a more pragmatic, industry-driven approach. Anthropic, for example, published a post in 2025 advising developers to make a crucial architectural distinction:

- Workflows: Systems where LLMs and tools are "orchestrated through _predefined_ code paths." The human defines the workflow, and the LLM executes the steps.
- Agents: Systems where the LLM "dynamically directs its own processes and tool usage."

Anthropic concludes that for most real-world applications, "Workflows" are superior because they offer "predictability" and "control". This "autonomy vs. control" spectrum represents the defining challenge for the _deployment_ of agentic AI in 2025. Researchers are pushing the boundaries of what is _possible_, while industry is trying to build guardrails to make it _reliable_ and _safe_.

### 3.5 The Evaluation Crisis: How to Benchmark Agents?

This new agentic paradigm has created an evaluation crisis. As a 2025 ArXiv paper points out, traditional LLM benchmarks—which typically test static knowledge or text generation quality—are "insufficient" for evaluating agents.

The paper offers a powerful analogy: "LLM evaluation is like examining the performance of an engine. In contrast, agent evaluation assesses a car's performance comprehensively, as well as under various driving conditions".

An agent's performance is not just about its "engine" (the LLM) but about its _interaction_ with a dynamic environment. It is a probabilistic system that must deal with API failures, ambiguous instructions, and long-term planning.

The 2025 solution, as detailed in a new survey, is the development of entirely new evaluation frameworks. This is a major theme at EMNLP 2025. These new benchmarks are moving away from simple accuracy to measure a new set of "agentic" dimensions:

- Task Completion: Did the agent achieve the multi-step goal?
- Memory and Context Retention: Does the agent remember instructions and findings from previous steps?
- Planning and Tool Integration: Can the agent correctly choose and use tools to accomplish its plan?
- User Experience: How efficient and intuitive is the human-agent collaboration?

## 4. Part 3: The New Domains: True Multimodality

The third major trend of 2025 is the rapid maturation of Multimodal Large Language Models (MLLMs), moving them from simple image-captioners to true "any-to-any" systems that can fluently reason across text, images, audio, and video.

### 4.1 Beyond Fused Encoders: Towards "Any-to-Any" MLLMs

The first generation of MLLMs (circa 2023-2024), such as LLaVA, were primarily "input-side" only. They fused a pre-trained vision encoder to an LLM, giving the LLM the ability to _understand_ images and _produce text_ about them.

The 2024-2025 paradigm shift is toward "any-to-any" models. The goal is a single model that can accept _any_ combination of modalities as input (e.g., a video and an audio question) and _produce_ output in any modality (e.g., an edited video with a text explanation).

### 4.2 Seminal Paper Review: "NExT-GPT: Any-to-Any Multimodal LLM" (Wu et al., ICML 2024)

This ICML 2024 paper provides the architectural blueprint for this "any-to-any" vision.

- Core Problem: To bridge the gap from "input-side" multimodal understanding to "any-to-any" multimodal _generation_.
- Novel Methodology: NExT-GPT's architecture is elegant. It positions the LLM as a central cognitive router. The system connects three off-the-shelf, pre-trained components:
  1. Multimodal Encoders: Existing models to "perceive" inputs (images, video, audio).
  2. A Central LLM: The "brain" that performs reasoning and, crucially, "modality switching."
  3. Multimodal Diffusion Decoders: Existing models (like Stable Diffusion) that can generate content (images, audio) based on instructions from the LLM.  
     The system is trained with only 1% of its parameters (the small adaptors connecting the components) using a novel "modality-switching instruction tuning (MosIT)" dataset.
- Key Results: The MosIT dataset successfully "empowered NExT-GPT with complex cross-modal semantic understanding and content generation". The model can, for example, take an image and a text prompt as input and generate a new, edited image as output.
- Conclusion & Impact: This paper showcases a "unified AI agent capable of modeling universal modalities". This architecture—using the LLM as a central reasoning layer to coordinate existing encoders and decoders—has become a key trend in 2024-2025 for building complex, generative multimodal systems.

### 4.3 Deep Dive: The Video-Language Frontier

While image-language tasks are maturing, the video-language frontier is where the most active research is happening.66 Video introduces the fundamental complexity of time. The key 2025 challenge, as highlighted in numerous EMNLP 2025 papers 70, is "fine-grained temporal grounding"—the ability to link a natural language description to _specific, precise moments_ in a video.

### 4.4 Seminal Paper Review: "Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models" (Wang et al., EMNLP 2025)

This EMNLP 2025 _Findings_ paper is a perfect example of research attacking this new frontier.

- Core Problem: Existing Video-LLMs struggle with fine-grained temporal grounding. They can provide a coarse-grained summary of a video but cannot answer questions like, "What did the person say _between 0:31 and 0:34_?" The paper identifies the causes as "ineffective temporal modeling and inadequate timestamp representations".
- Novel Methodology: The authors propose Grounded-VideoLLM, which introduces several innovations:
  1. Two-Stream Encoder: A novel encoder that explicitly captures _inter-frame relationships_ (e.g., motion) in one stream while preserving _intra-frame visual details_ in another.
  2. Progressive Training: A multi-stage training strategy that ensures a "smooth learning curve." The model is first trained on simple video-caption tasks, then "progressively introduced to complex video temporal grounding tasks".
  3. Synthetic Data: To strengthen temporal reasoning, the authors constructed a new "VideoQA dataset with grounded information using an automated annotation pipeline". This is another example of the synthetic data trend we will discuss shortly.
- Key Results: "Extensive experiments demonstrate that Grounded-VideoLLM not only surpasses existing models in fine-grained grounding tasks but also exhibits strong potential as a general video understanding assistant".
- Conclusion & Impact: This paper is representative of the 2025 multimodal trend: moving beyond "whole video" understanding to precise, "in-the-moment" temporal reasoning. This is a critical capability for applications like video analysis, robotics, and human-computer interaction.

## 5. Part 4: The Great Debates: Reasoning, Reliability, and Safety

As models have become more capable, the 2025 research landscape is dominated by a critical, introspective debate about the _reliability_ of these capabilities. Can these models _truly_ reason? Can they be trusted? And what are the hidden trade-offs of making them "safe"?

### 5.1 The Reasoning Debate (2025): Parrot or Thinker?

This debate has been simmering for years, but 2025 has brought new, nuanced evidence.

- Background: The debate was ignited by the 2021 "Stochastic Parrots" paper 75, which argued that LLMs are simply large-scale pattern matchers with no "understanding." The counter-argument came from the "emergent abilities" demonstrated by scaled models, most notably Chain-of-Thought (CoT) reasoning 77 and its successors like Tree of Thoughts (ToT) 78, which showed that prompting models to "think step-by-step" allowed them to solve complex reasoning problems they would otherwise fail.
- The Nuanced View (2025): The debate in 2025 is far more sophisticated. Recent papers, such as Apple's controversial "Illusion of Thinking" 83, claimed that Large Reasoning Models (LRMs) fail at complex, out-of-distribution reasoning tasks (e.g., Towers of Hanoi), suggesting they are still just "stochastic parrots".85
  - However, follow-up research published in 2025 provided a critical rebuttal.85 This new study found that the "Illusion of Thinking" experiments were _fundamentally flawed_. For example, they tested the models on _unsolvable_ configurations of the River Crossing puzzle. When the researchers tested the _same models_ on _solvable_ puzzles, the LRMs "solved instances with 100+ agents effortlessly".85
  - This leads to the 2025 consensus: the truth is nuanced and in the middle. LRMs are not just "pattern-matching parrots," but they are also not "human-level reasoners." A more accurate description is that they are "stochastic, RL-tuned searchers" in a high-dimensional latent space.85 Some tasks are trivial for their search mechanism (even at large scale), while others (like complex Towers of Hanoi) consistently break them.

### 5.2 Seminal Report Review: "The Decreasing Value of Chain of Thought in Prompting" (Meincke et al., 2025)

This technical report from June 2025 adds a critical, practical dimension to the reasoning debate. It challenges the universal assumption that CoT prompting is always a superior method.

- Core Problem: The report investigates the common wisdom that CoT prompting ("think step by step") is a universally beneficial practice.
- Methodology: The researchers tested a suite of modern 2025 models on PhD-level multiple-choice questions. They crucially distinguished between:
  1. "Non-reasoning models" (e.g., GPT-4o, Sonnet 3.5)
  2. "Reasoning models" (e.g., o4-mini, Flash 2.5)  
     They compared a "Direct" prompt against a "Step by step" (CoT) prompt, running each question 25 times to measure performance and consistency.
- Key Findings: The results were striking and challenged the "CoT is always better" dogma.
  1. For Reasoning Models: CoT showed "diminishing returns." The models showed only "marginal benefits" in accuracy, which "rarely" justified the "substantial time costs" (a 20-80% increase in response time).
  2. For Non-Reasoning Models: The results were mixed. While CoT produced "modest average improvements" on some models, it also "increased variability". This is a key finding: CoT caused the models to _change their answers_ on problems they previously answered _correctly_, leading to new errors.
- Conclusion & Impact: This report suggests that as models (especially "reasoning models") internalize complex reasoning capabilities, explicit CoT prompting may become an unreliable crutch. It is not a magical "thinking" button. It simply forces the model down a different, longer "stochastic search" path, which is not guaranteed to be better and can, in fact, be worse. This suggests that the value of CoT prompting will _decrease_ as model architectures improve.

### 5.3 Automating Oversight: LLM-as-a-Judge

As LLM-generated outputs (especially from agents) become longer and more complex, human evaluation has become a critical bottleneck. This has led to the 2024-2025 trend of using LLM-as-a-Judge—using a powerful LLM to evaluate, score, and even provide feedback on the outputs of other models. However, this simply pushes the problem up one level: how do we ensure the _judge_ is reliable, fair, and transparent?

### 5.4 Seminal Paper Review: "EvalPlanner: A Preference Optimization Algorithm for Thinking-LLM-as-a-Judge" (Saha et al., 2025)

This 2025 paper (accepted to ACL 2025) directly tackles the problem of building better LLM judges.

- Core Problem: Previous judge models were limited. Their reasoning was often "constrained" to "hand-designed components" (e.g., a fixed list of criteria) and they "intertwined planning with the reasoning for evaluation". They were following a fixed, human-provided rubric.
- Novel Methodology: EvalPlanner proposes a more powerful, decoupled CoT. The model is trained via preference optimization (learning from pairs of good/bad evaluations) to perform a two-stage process:
  1. Generate an Evaluation Plan: First, the model generates an _unconstrained_ "recipe" for _how_ it will evaluate the given response. This plan is tailored to the specific question and answer.
  2. Execute the Plan: Second, the model follows its _own_ plan step-by-step to arrive at the final verdict.
- Key Results: EvalPlanner achieved new state-of-the-art performance on RewardBench and PPE, despite being trained on _fewer_ synthetically generated preference pairs. The key was that _learning to plan an evaluation_ was a more robust and generalizable strategy than simply _following_ a fixed evaluation template.
- Conclusion & Impact: This paper demonstrates a significant step in meta-reasoning: teaching a model to _plan how to plan_. This capability is essential for creating the robust, transparent, and scalable automated evaluation systems required for the agentic era.

### 5.5 The Alignment Trade-off: Safety vs. Capability

Perhaps the most critical and contentious debate in 2025 surrounds AI safety and alignment. The standard industry pipeline for building a state-of-the-art model has become: (1) Pre-training, (2) Instruction Fine-Tuning (SFT) and/or Reasoning Fine-Tuning, and (3) Safety Alignment, often using Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO).

A critical discovery, highlighted in 2025 preprints, is that this final safety step (Step 3) may be actively _damaging_ the model's reasoning capability (from Step 2). This phenomenon has been dubbed the "Safety Tax".

Research investigating this pipeline found that:

1. Fine-tuning a base model for reasoning (e.g., on math CoT data) significantly _improves_ its reasoning scores but can also _degrade_ its safety, making it more vulnerable to misuse.
2. Then, applying safety alignment (e.g., fine-tuning on harmful-question/polite-refusal pairs) successfully _restores_ the model's safety and makes it harmless.
3. However, this safety alignment simultaneously _degrades_ the model's reasoning capabilities, with one paper reporting a 7% to 30% drop in reasoning accuracy.

This implies a fundamental, unresolved trade-off. Current alignment methods appear to force a choice between a _smart_ model and a _safe_ model. This is one of the most significant open problems in the field, as it suggests that making models safe may be making them "less reasonable".

### 5.6 Seminal Paper Review: "Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable" (2025)

This March 2025 preprint provides the most direct evidence for this trade-off.

- Core Problem: To systematically investigate the impact of safety alignment (SFT on refusal pairs) when applied to a Large Reasoning Model (LRM) that has already been fine-tuned for high reasoning performance.
- Methodology: The authors used a clear two-stage pipeline: (1) Reasoning Training, followed by (2) Safety Alignment. They measured reasoning accuracy (on benchmarks like GPQA) and harmfulness (on BeaverTails) after each stage. They tested two types of safety datasets: one with long "COT Refusal" traces and one with "DirectRefusal" answers.
- Key Findings:
  - The safety alignment _worked_ to make the model safe, successfully reducing its harmfulness score.
  - However, this came at the direct _cost_ of "downgrading... reasoning capability."
  - The trade-off was explicit: safety alignment with "DirectRefusal" data was _most effective_ at restoring safety but also _most damaging_ to reasoning, causing a 30.91% drop in accuracy.
- Conclusion & Impact: The paper concludes that this sequential pipeline presents an "unavoidable trade-off," which they term the "Safety Tax." This finding presents a critical challenge to the entire alignment research program and is part of a broader conversation about the "alignment tax".

### 5.7 Proactive Defense: The Formalization of Red Teaming

Given the high stakes of model failure, the field has moved from reactive patching to proactive defense. This is the goal of Red Teaming: a structured process of adversarially testing a model to find its vulnerabilities, biases, and safety flaws _before_ deployment. This has become a formalized practice at all major labs.

However, manual red teaming by human experts is "costly" and slow. This has created a need for _automated_ red-teaming systems, which themselves are a major 2025 research topic.

### 5.8 Seminal Paper Review: "MART: Improving LLM Safety with Multi-round Automatic Red-Teaming" (Zhu et al., NAACL 2024)

This NAACL 2024 paper proposes a scalable solution to the red-teaming bottleneck.

- Core Problem: Manual red teaming is not scalable. Existing automatic methods are good at _discovering_ safety risks but do not _address_ them.
- Novel Methodology (MART): The Multi-round Automatic Red-Teaming (MART) method proposes an iterative "game" between two LLMs:
  1. An Adversarial LLM is prompted to generate "challenging prompts" (jailbreaks) to elicit an unsafe response from a Target LLM.
  2. The Target LLM's unsafe responses are collected.
  3. This new dataset of failures is used to safety fine-tune the Target LLM, creating an "updated" and safer version.
  4. The loop repeats, with the Adversarial LLM now crafting _new, harder_ attacks against the _improved_ Target LLM.
- Key Results: The method was highly effective. The violation rate of a target LLM (with limited initial alignment) was reduced by 84.7% after just 4 rounds of MART.
- Crucial Finding: Most importantly, "model helpfulness on non-adversarial prompts remains stable throughout iterations".
- Conclusion & Impact: MART provides a scalable, automated framework for safety alignment. This finding provides a crucial counter-point to the "Safety Tax" paper. It suggests that _how_ you safety-tune matters. An iterative, adversarial fine-tuning loop (like MART) may be able to improve safety _without_ the catastrophic degradation of general helpfulness. This, along with a new wave of 2025 papers on adaptive adversarial attacks and defenses, marks the maturation of adversarial robustness as a formal sub-field.

## 6. Part 5: The Data Engine: Self-Supervision and Synthetic Generation

### 6.1 The Role of Self-Supervised Learning (SSL)

Underpinning every single model discussed today is the paradigm of Self-Supervised Learning (SSL). SSL is the machine learning technique that enables models to learn meaningful representations from massive, _unlabeled_ datasets (like the text of the internet).

In SSL, implicit labels are generated _from the data itself_. The "pretext task" for an LLM like GPT, for example, is "predict the next word." The "label" is simply the next word in the text, requiring no human annotation. This ability to learn from web-scale unlabeled data is what enables the creation of foundation models. SSL remains the foundational paradigm for the pre-training stage of all large-scale models.

### 6.2 The 2025 Trend: LLMs as Data Generators

While SSL is used for _pre-training_, the _fine-tuning_ and _alignment_ stages require high-quality, labeled data, which is expensive and time-consuming for humans to create.

This bottleneck has led to one of the most significant—and controversial—trends of 2025: using LLMs themselves to generate synthetic data. This is a central topic at ACL 2025. The promise is that LLMs can create vast, diverse, and cheap datasets for tasks like classification, question answering, and instruction tuning.

However, this practice is highly contentious. A 2025 review of ACL paper submissions noted a "concerning trend" of researchers "using LLMs to generate so-called benchmark datasets and then claiming that these datasets can be used for training/fine-tuning." The reviewer criticized this approach, noting that these datasets are of "unknown quality and representativeness" and that researchers are "relying on LLMs to generate data because it is easy and convenient," not because it is rigorous.

### 6.3 Survey Review: "A Survey on LLM-driven Synthetic Data Generation" (2025)

This 2025 survey paper provides a systematic overview of this new and controversial sub-field.

- Key Techniques: The survey outlines the primary methods for generating synthetic data:
  1. Prompt-Based Generation: Using zero-shot or few-shot prompts to instruct an LLM to generate new labeled examples for a target task.
  2. Retrieval-Augmented Pipelines (RAG): Grounding the LLM's generation in real documents (retrieved from a corpus) to improve the factual accuracy of the synthetic data.
  3. Iterative Self-Refinement: Using a model's own outputs (e.g., generated code) to iteratively fine-tune and improve itself.
- Key Challenges: The survey also highlights the profound risks:
  1. Factual Inaccuracies: The LLM "hallucinates" and generates data that is factually incorrect, which can poison the dataset.
  2. Bias Amplification: The inherent societal biases (e.g., gender, cultural) present in the parent LLM are "baked into" the synthetic data, potentially amplifying them.
  3. Model Collapse: This is identified as the most critical long-term risk. "Model collapse" or "model deterioration" occurs when models are trained on the synthetic output of _other models_. This creates a "self-consuming loop" where, over successive generations, the data ecosystem loses diversity, "forgets" the real-world distribution, and irreversibly degrades in quality.
- Mitigation Strategies:
  - Filtering and Blending: The most common mitigation is to never train on _purely_ synthetic data. Instead, high-quality synthetic data is "blended" with a (smaller) set of real-world data to "anchor" the model in reality.
  - Reinforcement Learning from Execution Feedback (RLEF): A key 2025 technique, RLEF is particularly powerful for _code_ generation. Synthetic code has a unique property: it can be _automatically verified_ by an external, objective "truth" (the code interpreter or compiler). By generating code, _running it_, and using the pass/fail signal as execution feedback, researchers can filter for _functionally correct_ synthetic data. This provides a robust, non-human reward signal that breaks the "hallucination" loop.

### 6.4 The Self-Consuming Loop

The rise of synthetic data presents a fundamental challenge for the future of AI. The field is actively building a "self-consuming" data loop.

The logical chain is as follows:

1. Training modern AI models requires massive, high-quality datasets.
2. Acquiring this data from humans is the primary bottleneck: it is slow, expensive, and difficult to scale.
3. Therefore, researchers and labs are increasingly using their best-performing models to generate _synthetic data_ to train the _next_ generation of models.
4. This creates a closed, autoregressive loop, where models are trained on the output of other models.

This problem is so significant that it was the focus of a dedicated workshop at NeurIPS 2025, "AI in the Synthetic Data Age," which explicitly aims to study "AI model deterioration due to synthetic data training".

This does not mean synthetic data is useless. Instead, it implies that as generative models become more common, the importance of data curation, data filtering, and "data-centric" AI is becoming _more_ critical, not less. The RLEF technique is a powerful example of a successful mitigation strategy because the code interpreter acts as an _external, objective filter_ that prevents the "hallucination" loop from taking over. Finding similar, automated "truth" filters for natural language remains a massive open challenge.

## 7. Part 6: Concluding Lecture: The Frontiers of 2026 and Beyond

As this lecture concludes, we look to the immediate future. The trends of 2025 point directly to the open problems and research frontiers that will define 2026 and beyond.

### 7.1 Efficiency and Ubiquity

The push for efficiency is not just about training SOTA models; it's about _deploying_ them.

- On-Device AI: A major 2025-2026 trend is running foundation models _on-device_ (e.g., mobile phones, laptops). This is critical for privacy (data never leaves the device) and latency (no network round-trip). This requires new techniques in model compression, quantization, and efficient architecture design.
- Efficient Code Generation: As agents write more code, the _quality_ of that code is coming under scrutiny. New 2024-2025 benchmarks like ENAMEL are being designed to evaluate LLM-generated code not just for _functional correctness_ (does it run?), but for _algorithmic efficiency_ (does it run _fast_?). This is a much higher bar for reasoning.
- Hardware Co-design: The field is moving beyond algorithmic optimization and looking at the hardware itself. New research is conducting "limit studies" to identify the fundamental bottlenecks in LLM inference, focusing on memory bandwidth and chip-to-chip interconnects. This signals a new era of co-design, where future hardware will be built specifically for new architectures like SSMs and MoEs.

### 7.2 The Quantum Leap: An Introduction to QNLP

Further on the horizon lies a paradigm that could fundamentally rewrite the rules of computation: Quantum Natural Language Processing (QNLP).

- 7.2.1 The Concept: QNLP is an emerging, interdisciplinary field that seeks to apply the principles of quantum computing to natural language processing. The core idea is that the rich, compositional structures of language (e.g., grammar, semantics) are difficult to model with classical statistics but may be naturally represented by the mathematics of quantum mechanics, such as quantum entanglement, superposition, and interference.
- 7.2.2 The 2025 Status: This field is highly experimental.
  - Tools: The lambeq toolkit, an open-source Python library from Quantinuum, exists to convert natural language sentences into parameterized quantum circuits, ready to be run on quantum hardware.
  - Theory: A wave of 2024-2025 surveys are mapping the theoretical landscape, designing "quantum embeddings," "quantum attention mechanisms," and hybrid classical-quantum models.
  - Practice: All current research readily admits that the field is severely constrained by "hardware limitations," noise, and decoherence. A 2025 paper, for example, demonstrates a QNLP model on a _few-shot_ Natural Language Inference task. All current work is limited to "small data sets".
- 7.2.3 Future Promise: While still in its infancy, QNLP holds the long-term _potential_ to achieve a "quantum advantage"—solving NLP tasks more efficiently or accurately than any classical model ever could. The most promising near-term applications are in specialized scientific domains like bioinformatics, drug discovery, and protein structure prediction, where quantum simulation can be applied to complex biological "languages".

### 7.3 Final Summary: Open Research Questions for 2026

As we conclude this course, the key open questions—many of which can form the basis of a thesis or future research career—are no longer "Can we scale?" but "What have we built, and how do we control it?"

Based on our review of the 2024-2025 landscape, the open research questions for 2026 are:

1. Architecture: Can we successfully combine SSMs (like Mamba) and MoE into a single hybrid architecture? Can such a model achieve _both_ linear-time context scaling and massive, sparsely-activated knowledge?
2. Agency: How do we solve the "autonomy vs. control" dilemma? How do we build agents that are both capable and verifiably safe? And, critically, how do we solve the _new_ alignment problem for _multi-agent systems_?
3. Reliability: How do we solve the "Safety Tax"? Can we develop new alignment techniques (like MART) that verifiably preserve reasoning and other capabilities?
4. Data: How do we build a _sustainable_ data ecosystem? How do we break the "self-consuming loop" and prevent "model collapse"? What other objective, external filters (like RLEF for code) can we discover for natural language?
5. Interdisciplinarity: The future of NLP is "beyond scaling". The official theme of EMNLP 2025 is "Interdisciplinary Recontextualization of NLP", and the theme of AAAI-26 is "collaborative bridges". The most significant open questions now lie at the intersection of NLP and other fields: science, medicine, education, and social science.

## References

1. A Comprehensive Review of Deep Learning: Architectures, Recent Advances, and Applications \- MDPI, [https://www.mdpi.com/2078-2489/15/12/755](https://www.mdpi.com/2078-2489/15/12/755)
2. Deep learning for natural language processing: advantages and challenges | National Science Review | Oxford Academic, [https://academic.oup.com/nsr/article/5/1/24/4107792](https://academic.oup.com/nsr/article/5/1/24/4107792)
3. Deep Learning for Natural Language Processing: A Review of Models and Applications, [https://www.researchgate.net/publication/395057230_Deep_Learning_for_Natural_Language_Processing_A_Review_of_Models_and_Applications](https://www.researchgate.net/publication/395057230_Deep_Learning_for_Natural_Language_Processing_A_Review_of_Models_and_Applications)
4. Natural Language Processing (NLP) in Artificial Intelligence \- | World Journal of Advanced Research and Reviews, [https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-0275.pdf](https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-0275.pdf)
5. Transformer Architecture Evolution in Large Language Models: A Survey \- ResearchGate, [https://www.researchgate.net/publication/394522965_Transformer_Architecture_Evolution_in_Large_Language_Models_A_Survey](https://www.researchgate.net/publication/394522965_Transformer_Architecture_Evolution_in_Large_Language_Models_A_Survey)
6. A Survey of Large Language Models: Evolution, Architectures, Adaptation, Benchmarking, Applications, Challenges, and Societal Implications \- MDPI, [https://www.mdpi.com/2079-9292/14/18/3580](https://www.mdpi.com/2079-9292/14/18/3580)
7. A Survey of Large Language Models \- arXiv, [https://arxiv.org/html/2303.18223v16](https://arxiv.org/html/2303.18223v16)
8. The 2024 Conference on Empirical Methods in Natural Language ..., [https://2024.emnlp.org/](https://2024.emnlp.org/)
9. ACL 2025 Highlights: Direction of NLP & AI | by Megagon Labs, [https://megagonlabs.medium.com/acl-2025-highlights-direction-of-nlp-ai-e9478c0b4ccf](https://megagonlabs.medium.com/acl-2025-highlights-direction-of-nlp-ai-e9478c0b4ccf)
10. Accepted Main Conference Papers \- ACL 2025, [https://2025.aclweb.org/program/main_papers/](https://2025.aclweb.org/program/main_papers/)
11. The 2025 Conference on Empirical Methods in Natural Language ..., [https://2025.emnlp.org/](https://2025.emnlp.org/)
12. NeurIPS/ICLR/ICML Journal-to-Conference Track, [https://neurips.cc/public/JournalToConference](https://neurips.cc/public/JournalToConference)
13. Workshops \- NeurIPS 2025, [https://neurips.cc/virtual/2025/events/workshop](https://neurips.cc/virtual/2025/events/workshop)
14. NeurIPS 2025 Papers, [https://neurips.cc/virtual/2025/papers.html](https://neurips.cc/virtual/2025/papers.html)
15. NeurIPS 2024 Papers, [https://nips.cc/virtual/2024/papers.html](https://nips.cc/virtual/2024/papers.html)
16. Proceedings of the 2025 Conference on Empirical ... \- ACL Anthology, [https://aclanthology.org/2025.emnlp-tutorials.pdf](https://aclanthology.org/2025.emnlp-tutorials.pdf)
17. ICML 2024 Papers, [https://icml.cc/virtual/2024/papers.html](https://icml.cc/virtual/2024/papers.html)
18. Top 10 NLP Trends to Watch in 2025 – Future of AI & Language Processing | Shaip, [https://www.shaip.com/blog/nlp-trends-2025/](https://www.shaip.com/blog/nlp-trends-2025/)
19. Natural Language Processing Statistics By Market, Revenue And Trends (2025) \- ElectroIQ, [https://electroiq.com/stats/natural-language-processing-statistics/](https://electroiq.com/stats/natural-language-processing-statistics/)
20. Natural language processing (NLP) Decade Long Trends, Analysis and Forecast 2025-2033, [https://www.archivemarketresearch.com/reports/natural-language-processing-nlp-559381](https://www.archivemarketresearch.com/reports/natural-language-processing-nlp-559381)
21. Future of Natural Language Processing: Key Trends in 2025 \- IABAC, [https://iabac.org/blog/the-future-of-natural-language-processing](https://iabac.org/blog/the-future-of-natural-language-processing)
22. The Future of Natural Language Processing: Trends to Watch in 2025 and Beyond, [https://www.tekrevol.com/blogs/natural-language-processing-trends/](https://www.tekrevol.com/blogs/natural-language-processing-trends/)
23. Natural language processing models in 2025 | Pre-trained NLP models | NLP solutions for businesses | Lumenalta, [https://lumenalta.com/insights/7-of-the-best-natural-language-processing-models-in-2025](https://lumenalta.com/insights/7-of-the-best-natural-language-processing-models-in-2025)
24. Transformer: A Novel Neural Network Architecture for Language Understanding, [https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)
25. Mamba: Linear-Time Sequence Modeling with Selective ... \- arXiv, [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
26. From S4 to Mamba: A Comprehensive Survey on Structured State Space Models \- arXiv, [https://arxiv.org/abs/2503.18970](https://arxiv.org/abs/2503.18970)
27. Mamba State-Space Models Are Lyapunov-Stable Learners \- arXiv, [https://arxiv.org/pdf/2406.00209](https://arxiv.org/pdf/2406.00209)
28. Mamba: Linear-Time Sequence Modeling with Selective State Spaces \- arXiv, [https://arxiv.org/pdf/2312.00752](https://arxiv.org/pdf/2312.00752)
29. Mixture-of-Experts in the Era of LLMs A New Odyssey, [https://icml.cc/media/icml-2024/Slides/35222_1r94S59.pdf](https://icml.cc/media/icml-2024/Slides/35222_1r94S59.pdf)
30. Mixture of Demonstrations for In-Context Learning \- NIPS, [https://proceedings.neurips.cc/paper_files/paper/2024/file/a0da098e0031f58269efdcba40eedf47-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/a0da098e0031f58269efdcba40eedf47-Paper-Conference.pdf)
31. MoME: Mixture of Multimodal Experts for Generalist ... \- NIPS papers, [https://proceedings.neurips.cc/paper_files/paper/2024/file/4a3a14b9536806a0522930007c5512f7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/4a3a14b9536806a0522930007c5512f7-Paper-Conference.pdf)
32. A Perspective on LLM Data Generation with Few-shot Examples: from Intent to Kubernetes Manifest \- ACL Anthology, [https://aclanthology.org/2025.acl-industry.27.pdf](https://aclanthology.org/2025.acl-industry.27.pdf)
33. AI Agents in 2025: Expectations vs. Reality \- IBM, [https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality](https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality)
34. 5 Cutting-Edge Natural Language Processing Trends Shaping 2026 \- KDnuggets, [https://www.kdnuggets.com/5-cutting-edge-natural-language-processing-trends-shaping-2026](https://www.kdnuggets.com/5-cutting-edge-natural-language-processing-trends-shaping-2026)
35. Large Language Models: A Survey \- arXiv, [https://arxiv.org/html/2402.06196v3](https://arxiv.org/html/2402.06196v3)
36. \[2406.05804\] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning \- arXiv, [https://arxiv.org/abs/2406.05804](https://arxiv.org/abs/2406.05804)
37. Plan Then Action: High-Level Planning Guidance Reinforcement Learning for LLM Reasoning \- arXiv, [https://arxiv.org/html/2510.01833v1](https://arxiv.org/html/2510.01833v1)
38. Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study \- arXiv, [https://arxiv.org/html/2507.23589v1](https://arxiv.org/html/2507.23589v1)
39. Talk: Beyond Scaling: Frontiers of Retrieval-Augmented Language Models, [https://today.wisc.edu/events/view/206147](https://today.wisc.edu/events/view/206147)
40. Beyond Scaling: Frontiers of Retrieval-Augmented Language Models \- Harvard SEAS, [https://events.seas.harvard.edu/event/beyond-scaling-frontiers-of-retrieval-augmented-language-models](https://events.seas.harvard.edu/event/beyond-scaling-frontiers-of-retrieval-augmented-language-models)
41. Large Language Model Agent: A Survey on Methodology, Applications and Challenges \- arXiv, [https://arxiv.org/pdf/2503.21460](https://arxiv.org/pdf/2503.21460)
42. Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research \- arXiv, [https://arxiv.org/html/2506.01839v1](https://arxiv.org/html/2506.01839v1)
43. Multi-Agent Systems Powered by Large Language Models: Applications in Swarm Intelligence \- arXiv, [https://arxiv.org/html/2503.03800v1](https://arxiv.org/html/2503.03800v1)
44. LLM Multi-Agent Systems: Challenges and Open Problems \- arXiv, [https://arxiv.org/html/2402.03578v2](https://arxiv.org/html/2402.03578v2)
45. \[2510.05174\] Emergent Coordination in Multi-Agent Language Models \- arXiv, [https://arxiv.org/abs/2510.05174](https://arxiv.org/abs/2510.05174)
46. \[2409.02645\] Emergent Language: A Survey and Taxonomy \- arXiv, [https://arxiv.org/abs/2409.02645](https://arxiv.org/abs/2409.02645)
47. MAEBE: Multi-Agent Emergent Behavior Framework \- arXiv, [https://arxiv.org/pdf/2506.03053](https://arxiv.org/pdf/2506.03053)
48. Agent Laboratory: Using LLM Agents as Research Assistants \- arXiv, [https://arxiv.org/abs/2501.04227](https://arxiv.org/abs/2501.04227)
49. A Review of Large Language Models as Autonomous Agents and Tool Users \- arXiv, [https://arxiv.org/html/2508.17281v1](https://arxiv.org/html/2508.17281v1)
50. Trending Papers \- Hugging Face, [https://huggingface.co/papers/trending](https://huggingface.co/papers/trending)
51. Levels of Autonomy for AI Agents Working Paper \- arXiv, [https://arxiv.org/html/2506.12469v1](https://arxiv.org/html/2506.12469v1)
52. Building Effective AI Agents \- Anthropic, [https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)
53. Seizing the agentic AI advantage \- McKinsey, [https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage)
54. The State of AI Agent Platforms in 2025: Comparative Analysis \- Ionio, [https://www.ionio.ai/blog/the-state-of-ai-agent-platforms-in-2025-comparative-analysis](https://www.ionio.ai/blog/the-state-of-ai-agent-platforms-in-2025-comparative-analysis)
55. Evaluation and Benchmarking of LLM Agents: A Survey \- arXiv, [https://arxiv.org/html/2507.21504v1](https://arxiv.org/html/2507.21504v1)
56. \[2503.22458\] Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey \- arXiv, [https://arxiv.org/abs/2503.22458](https://arxiv.org/abs/2503.22458)
57. The 2025 Conference on Empirical Methods in Natural Language Processing, [https://aclanthology.org/events/emnlp-2025/](https://aclanthology.org/events/emnlp-2025/)
58. \[2408.01319\] A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks \- arXiv, [https://arxiv.org/abs/2408.01319](https://arxiv.org/abs/2408.01319)
59. survey on multimodal large language models | National Science Review \- Oxford Academic, [https://academic.oup.com/nsr/article/11/12/nwae403/7896414](https://academic.oup.com/nsr/article/11/12/nwae403/7896414)
60. ICML 2024 NExT-GPT: Any-to-Any Multimodal LLM Oral \- ICML 2025, [https://icml.cc/virtual/2024/oral/35529](https://icml.cc/virtual/2024/oral/35529)
61. Apple Machine Learning Research at NeurIPS 2024, [https://machinelearning.apple.com/research/neurips-2024](https://machinelearning.apple.com/research/neurips-2024)
62. Code and models for ICML 2024 paper, NExT-GPT: Any-to-Any Multimodal Large Language Model \- GitHub, [https://github.com/NExT-GPT/NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)
63. NExT-GPT: Any-to-Any Multimodal LLM \- GitHub, [https://raw.githubusercontent.com/mlresearch/v235/main/assets/wu24e/wu24e.pdf](https://raw.githubusercontent.com/mlresearch/v235/main/assets/wu24e/wu24e.pdf)
64. NExT-GPT: Any-to-Any Multimodal LLM \- Proceedings of Machine Learning Research, [https://proceedings.mlr.press/v235/wu24e.html](https://proceedings.mlr.press/v235/wu24e.html)
65. \[2309.05519\] NExT-GPT: Any-to-Any Multimodal LLM \- arXiv, [https://arxiv.org/abs/2309.05519](https://arxiv.org/abs/2309.05519)
66. A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges \- arXiv, [https://arxiv.org/html/2501.02189v5](https://arxiv.org/html/2501.02189v5)
67. Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions \- arXiv, [https://arxiv.org/html/2404.07214v3](https://arxiv.org/html/2404.07214v3)
68. A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges \- arXiv, [https://arxiv.org/html/2501.02189v6](https://arxiv.org/html/2501.02189v6)
69. Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM, [https://arxiv.org/html/2505.18110v2](https://arxiv.org/html/2505.18110v2)
70. Findings of the Association for Computational Linguistics: EMNLP 2025 \- ACL Anthology, [https://aclanthology.org/volumes/2025.findings-emnlp/](https://aclanthology.org/volumes/2025.findings-emnlp/)
71. Grounded-VideoLLM: Sharpening Fine-grained Temporal ..., [https://aclanthology.org/2025.findings-emnlp.50/](https://aclanthology.org/2025.findings-emnlp.50/)
72. Findings of the Association for Computational Linguistics: EMNLP 2025 \- ACL Anthology, [https://aclanthology.org/2025.findings-emnlp.0.pdf](https://aclanthology.org/2025.findings-emnlp.0.pdf)
73. Zhiyang Xu \- OpenReview, [https://openreview.net/profile?id=\~Zhiyang_Xu1](https://openreview.net/profile?id=~Zhiyang_Xu1)
74. Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models \- ACL Anthology, [https://aclanthology.org/2025.findings-emnlp.50.pdf](https://aclanthology.org/2025.findings-emnlp.50.pdf)
75. Parrot or pilot? how llms 'think' when the geometry clicks | by BuildShift \- Level Up Coding, [https://levelup.gitconnected.com/parrot-or-pilot-how-llms-think-when-the-geometry-clicks-1ca58d307b8e](https://levelup.gitconnected.com/parrot-or-pilot-how-llms-think-when-the-geometry-clicks-1ca58d307b8e)
76. Proof of Thought : Neurosymbolic Program Synthesis allows Robust and Interpretable Reasoning \- arXiv, [https://arxiv.org/html/2409.17270v2](https://arxiv.org/html/2409.17270v2)
77. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models \- arXiv, [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
78. Tree of Thoughts: Deliberate Problem Solving with Large Language Models, [https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf)
79. What is Tree Of Thoughts Prompting? \- IBM, [https://www.ibm.com/think/topics/tree-of-thoughts](https://www.ibm.com/think/topics/tree-of-thoughts)
80. From Chains to Trees: Revolutionizing AI Reasoning with Tree-of-Thought Prompting” | by Jacky Hsiao | Medium, [https://medium.com/@jacky0305/from-chains-to-trees-revolutionizing-ai-reasoning-with-tree-of-thought-prompting-ff0afb566dce](https://medium.com/@jacky0305/from-chains-to-trees-revolutionizing-ai-reasoning-with-tree-of-thought-prompting-ff0afb566dce)
81. Tree of Thoughts (ToT) \- Prompt Engineering Guide, [https://www.promptingguide.ai/techniques/tot](https://www.promptingguide.ai/techniques/tot)
82. ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving \- arXiv, [https://arxiv.org/html/2505.12717v1](https://arxiv.org/html/2505.12717v1)
83. Top AI Research Papers of 2025: From Chain-of-Thought Flaws to Fine-Tuned AI Agents, [https://www.aryaxai.com/article/top-ai-research-papers-of-2025-from-chain-of-thought-flaws-to-fine-tuned-ai-agents](https://www.aryaxai.com/article/top-ai-research-papers-of-2025-from-chain-of-thought-flaws-to-fine-tuned-ai-agents)
84. The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity \- Apple Machine Learning Research, [https://machinelearning.apple.com/research/illusion-of-thinking](https://machinelearning.apple.com/research/illusion-of-thinking)
85. New Research Challenges Apple's "AI Can't Really Reason" Study \- Finds Mixed Results : r/OpenAI \- Reddit, [https://www.reddit.com/r/OpenAI/comments/1lqjw0n/new_research_challenges_apples_ai_cant_really/](https://www.reddit.com/r/OpenAI/comments/1lqjw0n/new_research_challenges_apples_ai_cant_really/)
86. Technical Report: The Decreasing Value of Chain of Thought in ..., [https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/](https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/)
87. Chain of Thought Prompting: Enhance AI Reasoning & LLMs \- Future AGI, [https://futureagi.com/blogs/chain-of-thought-prompting-ai-2025](https://futureagi.com/blogs/chain-of-thought-prompting-ai-2025)
88. Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge \- arXiv, [https://arxiv.org/html/2501.18099v2](https://arxiv.org/html/2501.18099v2)
89. ryokamoi/llm-self-correction-papers: List of papers on Self-Correction of LLMs. \- GitHub, [https://github.com/ryokamoi/llm-self-correction-papers](https://github.com/ryokamoi/llm-self-correction-papers)
90. Learning to Plan & Reason for Evaluation with Thinking-LLM ... \- arXiv, [https://arxiv.org/abs/2501.18099](https://arxiv.org/abs/2501.18099)
91. NeurIPS 2024 Spotlight Posters, [https://neurips.cc/virtual/2024/events/spotlight-posters-2024](https://neurips.cc/virtual/2024/events/spotlight-posters-2024)
92. Datasets Benchmarks 2024 \- NeurIPS 2025, [https://neurips.cc/virtual/2024/events/datasets-benchmarks-2024](https://neurips.cc/virtual/2024/events/datasets-benchmarks-2024)
93. A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications \- arXiv, [https://arxiv.org/html/2410.15595v3](https://arxiv.org/html/2410.15595v3)
94. Safety Alignment Makes Your Large Reasoning Models Less Reasonable \- arXiv, [https://arxiv.org/html/2503.00555v1](https://arxiv.org/html/2503.00555v1)
95. Safety Tax: Safety Alignment Makes Your Large Reasoning ... \- arXiv, [https://arxiv.org/abs/2503.00555](https://arxiv.org/abs/2503.00555)
96. \[2507.19672\] Alignment and Safety in Large Language Models: Safety Mechanisms, Training Paradigms, and Emerging Challenges \- arXiv, [https://arxiv.org/abs/2507.19672](https://arxiv.org/abs/2507.19672)
97. NLP for Social Good: A Survey of Challenges, Opportunities, and Responsible Deployment, [https://arxiv.org/html/2505.22327v1](https://arxiv.org/html/2505.22327v1)
98. Security Concerns for Large Language Models: A Survey \- arXiv, [https://arxiv.org/html/2505.18889v1](https://arxiv.org/html/2505.18889v1)
99. Red Teaming AI Red Teaming \- arXiv, [https://arxiv.org/html/2507.05538v1](https://arxiv.org/html/2507.05538v1)
100. MART: Improving LLM Safety with Multi-round Automatic Red ..., [https://aclanthology.org/2024.naacl-long.107/](https://aclanthology.org/2024.naacl-long.107/)
101. From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs, [https://arxiv.org/html/2506.13434v1](https://arxiv.org/html/2506.13434v1)
102. What Can Generative AI Red-Teaming Learn from Cyber Red-Teaming? \- Software Engineering Institute, [https://www.sei.cmu.edu/documents/6301/What_Can_Generative_AI_Red-Teaming_Learn_from_Cyber_Red-Teaming.pdf](https://www.sei.cmu.edu/documents/6301/What_Can_Generative_AI_Red-Teaming_Learn_from_Cyber_Red-Teaming.pdf)
103. From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation \- ACL Anthology, [https://aclanthology.org/2025.findings-emnlp.1244.pdf](https://aclanthology.org/2025.findings-emnlp.1244.pdf)
104. From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation \- ACL Anthology, [https://aclanthology.org/2025.findings-emnlp.1244/](https://aclanthology.org/2025.findings-emnlp.1244/)
105. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks \- arXiv, [https://arxiv.org/html/2507.06489v1](https://arxiv.org/html/2507.06489v1)
106. Libr-AI/OpenRedTeaming: Papers about red teaming LLMs and Multimodal models., [https://github.com/Libr-AI/OpenRedTeaming](https://github.com/Libr-AI/OpenRedTeaming)
107. What Is Self-Supervised Learning? \- IBM, [https://www.ibm.com/think/topics/self-supervised-learning](https://www.ibm.com/think/topics/self-supervised-learning)
108. Consequential Advancements of Self-Supervised Learning (SSL) in Deep Learning Contexts \- MDPI, [https://www.mdpi.com/2227-7390/12/5/758](https://www.mdpi.com/2227-7390/12/5/758)
109. 5th Workshop on Self-Supervised Learning: Theory and Practice \- NeurIPS 2025, [https://neurips.cc/virtual/2024/workshop/84703](https://neurips.cc/virtual/2024/workshop/84703)
110. Synthetic Data in the Era of LLMs, [https://synth-data-acl.github.io/](https://synth-data-acl.github.io/)
111. Synthetic Data Generation Using Large Language Models: Advances in Text and Code \- arXiv, [https://arxiv.org/pdf/2503.14023](https://arxiv.org/pdf/2503.14023)
112. \[D\] Reviewed several ACL papers on data resources and feel that LLMs are undermining this field : r/MachineLearning \- Reddit, [https://www.reddit.com/r/MachineLearning/comments/1jihs98/d_reviewed_several_acl_papers_on_data_resources/](https://www.reddit.com/r/MachineLearning/comments/1jihs98/d_reviewed_several_acl_papers_on_data_resources/)
113. Synthetic Data Generation Using Large Language Models ... \- arXiv, [https://arxiv.org/abs/2503.14023](https://arxiv.org/abs/2503.14023)
114. Synthetic Data Generation Using Large Language Models: Advances in Text and Code, [https://arxiv.org/html/2503.14023v2](https://arxiv.org/html/2503.14023v2)
115. Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls \- arXiv, [https://arxiv.org/html/2510.01631v1](https://arxiv.org/html/2510.01631v1)
116. FASTGEN: Fast and Cost-Effective Synthetic Tabular Data Generation with LLMs \- arXiv, [https://arxiv.org/html/2507.15839v1](https://arxiv.org/html/2507.15839v1)
117. NeurIPS 2025 Workshop on AI in the Synthetic Data Age: Challenges and Solutions \- DIGITAL SIGNAL PROCESSING AT RICE UNIVERSITY, [https://dsp.rice.edu/neurips-2025-workshop-on-ai-in-the-synthetic-data-age-challenges-and-solutions/](https://dsp.rice.edu/neurips-2025-workshop-on-ai-in-the-synthetic-data-age-challenges-and-solutions/)
118. Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use \- arXiv, [https://arxiv.org/html/2504.04736v1](https://arxiv.org/html/2504.04736v1)
119. NeurIPS 2025 Papers with Code & Data, [https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/](https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/)
120. Updates to Apple's On-Device and Server Foundation Language Models, [https://machinelearning.apple.com/research/apple-foundation-models-2025-updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
121. Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices \- arXiv, [https://arxiv.org/html/2504.00002v1](https://arxiv.org/html/2504.00002v1)
122. How efficient is LLM-generated code? A rigorous & high-standard benchmark \- arXiv, [https://arxiv.org/html/2406.06647v4](https://arxiv.org/html/2406.06647v4)
123. Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need \- arXiv, [https://arxiv.org/html/2507.14397v1](https://arxiv.org/html/2507.14397v1)
124. \[2505.13840\] EfficientLLM: Efficiency in Large Language Models \- arXiv, [https://arxiv.org/abs/2505.13840](https://arxiv.org/abs/2505.13840)
125. Natural Language Processing in 2025: Technologies, Trends & Business Impact \- Aezion, [https://www.aezion.com/blogs/natural-language-processing/](https://www.aezion.com/blogs/natural-language-processing/)
126. Quantum Natural Language Processing: A Comprehensive Review of Models, Methods, and Applications \- arXiv, [https://arxiv.org/html/2504.09909v2](https://arxiv.org/html/2504.09909v2)
127. \[2504.09909\] Quantum Natural Language Processing: A Comprehensive Review of Models, Methods, and Applications \- arXiv, [https://arxiv.org/abs/2504.09909](https://arxiv.org/abs/2504.09909)
128. Quantum Natural Language Processing: Challenges and Opportunities \- MDPI, [https://www.mdpi.com/2076-3417/12/11/5651](https://www.mdpi.com/2076-3417/12/11/5651)
129. Quantum natural language processing and its applications in bioinformatics: a comprehensive review of methodologies, concepts, and future directions \- Frontiers, [https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1464122/full](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1464122/full)
130. Design and analysis of quantum machine learning: a survey \- Taylor & Francis Online, [https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2312121](https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2312121)
131. Comparative Study of Traditional Machine Learning and Quantum Computing in Natural Language Processing: A Case Study on Sentiment Analysis \- IEEE Xplore, [https://ieeexplore.ieee.org/document/10791272/](https://ieeexplore.ieee.org/document/10791272/)
132. Quantinuum Announces Updates to Quantum Natural Language Processing Toolkit λambeq, Enhancing Accessibility, [https://www.quantinuum.com/press-releases/quantinuum-announces-updates-to-quantum-natural-language-processing-toolkit-lambeq-enhancing-accessibility](https://www.quantinuum.com/press-releases/quantinuum-announces-updates-to-quantum-natural-language-processing-toolkit-lambeq-enhancing-accessibility)
133. Natural Language, AI, and Quantum Computing in 2024 \- arXiv, [https://arxiv.org/html/2403.19758v1](https://arxiv.org/html/2403.19758v1)
134. A Survey on Quantum Machine Learning: Basics, Current Trends, Challenges, Opportunities, and the Road Ahead \- arXiv, [https://arxiv.org/html/2310.10315v3](https://arxiv.org/html/2310.10315v3)
135. A Survey on Quantum Machine Learning: Basics, Current Trends, Challenges, Opportunities, and the Road Ahead \- arXiv, [https://arxiv.org/html/2310.10315v4](https://arxiv.org/html/2310.10315v4)
136. \[2510.15972\] Quantum NLP models on Natural Language Inference \- arXiv, [https://www.arxiv.org/abs/2510.15972](https://www.arxiv.org/abs/2510.15972)
137. Main Technical Track: Call for Papers \- AAAI \- The Association for the Advancement of Artificial Intelligence, [https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/](https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/)
138. Natural Language Processing Projects 2026-27, [https://www.kcl.ac.uk/nmes/assets/informatics-pdfs-2026-27/natural-language-processing-projects-2026-27.pdf](https://www.kcl.ac.uk/nmes/assets/informatics-pdfs-2026-27/natural-language-processing-projects-2026-27.pdf)
