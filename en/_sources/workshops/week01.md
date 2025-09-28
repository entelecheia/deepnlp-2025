# Week 1 Workshop: LLM Overview and Development Environment Setup

## **Workshop Introduction: Exploring the Journey of Large Language Models**

Welcome to this LLM workshop. The objectives for Week 1 are as follows:

1. Establish a comprehensive understanding of the complete lifecycle of Large Language Model (LLM) projects, from ideation to deployment and maintenance.
2. Build a powerful GPU-accelerated development environment using industry-standard tools.
3. Successfully perform the first text generation task using pre-trained LLMs.

This workshop focuses on hands-on practice with two core tools. The first is **Hugging Face Transformers**. This library serves as the de facto standard for accessing vast pre-trained models and datasets, acting as the "lingua franca" of the NLP community. The second is

**NVIDIA NeMo Framework**. NeMo is an enterprise-grade cloud-native framework designed for building, customizing, and deploying large-scale generative AI models. The core value of this framework lies in its performance and scalability, deeply optimized for the NVIDIA hardware ecosystem.

This workshop combines Hugging Face's extensive model and data ecosystem with NeMo's powerful and scalable training and customization capabilities, presenting an approach that leverages the strengths of both worlds in modern LLM development.

---

## **Part 1: In-Depth Analysis of the Complete LLM Lifecycle**

LLM projects encompass more than simple model training, involving a complex process from clear goal setting to continuous maintenance. This process goes through multiple stages, with each stage having a significant impact on subsequent stages.

### **Stage 1: Scope Definition and Problem Formulation**

- **Objective:** Transform vague ideas into well-defined projects with clear business objectives and technical constraints.
- **Key Activities:**
  - **Problem Understanding:** Clearly articulate the problem to be solved. It should be specific, such as "reducing customer support workload" or "automated marketing copy generation."
  - **Feasibility Analysis:** Review available technology, data, and resources, and evaluate whether LLMs are suitable tools for solving the problem.
  - **Success Metrics Definition:** Establish quantifiable key performance indicators (KPIs). Examples include "30% reduction in support workload," "maintaining customer satisfaction above 4.5/5," and "response latency under 2 seconds."
  - **Constraint Identification:** List all operational, legal, and ethical boundaries, including data privacy protection (GDPR, etc.), latency requirements, and conditions requiring human agent handoff.

### **Stage 2: Data Collection and Refinement**

- **Objective:** Build high-quality, diverse, and representative datasets that will serve as the foundation for the model's knowledge and behavior.
- **Key Activities:**
  - **Data Sourcing:** Collect vast amounts of text from various sources such as books, websites, articles, and code.
  - **Preprocessing and Refinement:** Perform crucial steps including noise removal, normalization, deduplication, and filtering of low-quality or harmful content.
  - **Tokenization:** Convert raw text into numerical format (tokens) that models can process.
  - **Bias and Privacy Mitigation:** Actively work to mitigate biases present in source data and address privacy issues such as personally identifiable information (PII) removal.

### **Stage 3: Pre-training**

- **Objective:** Train a general-purpose "Foundation Model" based on large-scale unlabeled corpora. This model learns broad understanding of grammar, facts, reasoning abilities, and language.
- **Key Activities:**
  - **Unsupervised Learning:** Typically uses next-token prediction objectives. The model learns by predicting the next word in sentences.
  - **Large-scale Computing:** This is the most expensive stage financially and computationally, requiring millions of GPU hours.
- **Challenges:** Massive costs, overfitting risks, and the need for careful regularization techniques.

### **Stage 4: Supervised Fine-Tuning (SFT)**

- **Objective:** Adapt general-purpose foundation models to specific tasks or domains using smaller, labeled datasets.
- **Key Activities:**
  - **High-quality Dataset Generation:** Curate or directly generate examples of desired input-output pairs (e.g., instruction-response pairs).
  - **Task-specific Training:** Further train the model with labeled data to improve performance on specific tasks such as summarization, question answering, and instruction following.
- **Challenges:** Catastrophic forgetting phenomenon where models lose general capabilities while specializing in specific tasks, and difficulties in securing substantial amounts of high-quality labeled data.

### **Stage 5: Alignment and Safety Tuning (RLHF/DPO)**

- **Objective:** Align model behavior with human values, preferences, and safety guidelines to make them more useful, harmless, and honest.
- **Key Activities:**
  - **Reinforcement Learning from Human Feedback (RLHF):** A three-stage process involving (1) human preference data collection (ranking various model outputs), (2) training a "reward model" that predicts human preferences, and (3) fine-tuning LLMs using reinforcement learning (PPO, etc.) to maximize reward scores.
  - **Direct Preference Optimization (DPO):** A newer and more direct method that directly fine-tunes LLMs using the same preference data without training a separate reward model.
- **Challenges:** Reward hacking where models find loopholes to receive high rewards without being truly useful, and the high cost and complexity of scaling human feedback.

### **Stage 6: Evaluation and Benchmarking**

- **Objective:** Rigorously test model performance through extensive academic benchmarks and custom task-specific test sets.
- **Key Activities:**
  - **Automated Metrics:** Use indicators such as ROUGE scores for summarization tasks or accuracy for classification tasks.
  - **Human Evaluation:** Assess qualities such as usefulness, consistency, and safety that are difficult to capture with automated metrics.
  - **Red Teaming:** Actively seek out and attempt to exploit model vulnerabilities, biases, or safety failures.

### **Stage 7: Deployment and Inference Optimization**

- **Objective:** Enable trained and aligned models to be used in real applications in a reliable, scalable, and cost-effective manner.
- **Key Activities:**
  - **Model Optimization:** Apply techniques such as quantization (reducing weight precision), pruning (removing unnecessary weights), and distillation (training smaller models to mimic larger models) to reduce model size and improve latency.
  - **Serving Infrastructure:** Deploy models to cloud infrastructure, on-premises servers, or edge devices using inference servers such as Triton or vLLM.

### **Stage 8: Monitoring and Maintenance**

- **Objective:** Ensure model performance remains high over time and continuously improve based on real usage data.
- **Key Activities:**
  - **Performance Monitoring:** Track metrics such as latency, error rates, and user satisfaction.
  - **Drift Detection:** Identify when model performance degrades due to changes in input data patterns ("data drift").
  - **Continuous Improvement Loop (Data Flywheel):** Generate new training data for additional fine-tuning and model updates using actual inference data and user feedback.

### **Checkpoint Question 1: What is the most important stage in the LLM lifecycle?**

While all stages are interdependent, **scope definition (Stage 1) and data refinement (Stage 2)** can be considered the most important. Poorly defined problems lead to developing wrong solutions, wasting all subsequent efforts. Similarly, low-quality, biased, or irrelevant data fundamentally limits model potential regardless of how much computing resources are invested in pre-training or how sophisticated alignment techniques are. The principle of "garbage in, garbage out" applies thoroughly. Errors in these early stages are very difficult to fix later. Models pre-trained on flawed data require much more effort to align and may never reach desired performance or safety levels.

While it's easy to understand these stages as a linear process, actual LLM development is much more dynamic and iterative. For example, when performance degradation is detected in the monitoring stage (Stage 8), it doesn't simply end with redeploying the model. This feedback becomes an opportunity to build new fine-tuning datasets (Stage 4) or even return to previous stages to address fundamental data quality issues (Stage 2). Therefore, the LLM lifecycle should be understood not as a straight line but as a cyclical system like a "Data Flywheel," where each stage provides feedback to others and continuously drives improvement.

---

## **Part 2: Development Environment Setup: NVIDIA NGC Hands-on Guide**

This section provides step-by-step guidance on building a development environment for the workshop.

### **Prerequisites Checklist**

- **Hardware:** NVIDIA GPU (Pascal architecture or higher recommended for WSL2).
- **Software:**
  - Latest NVIDIA drivers for your operating system
  - Docker Engine installation and execution
  - Windows users: Windows 10 (21H2 or higher) or Windows 11, WSL2 activation and Linux distribution installation (e.g., Ubuntu).

### **Step-by-Step Installation Guide**

1. **NGC Access:**
   - Go to the NGC website (ngc.nvidia.com) and log in or create a free account.
2. **API Key Generation:**
   - Navigate to Setup > API Keys and click Generate Personal Key. This key serves as a password for programmatic access to NGC services.
   - **Important:** Copy the generated key immediately and store it securely. NGC does not store keys.
3. **Docker and NGC Registry Authentication:**
   - Open a terminal (PowerShell for Windows).
   - Run the `docker login nvcr.io` command.
   - Enter `$oauthtoken` as the username.
   - Paste the NGC API key you just generated as the password.
4. **Workshop Container Download:**
   - Use a specific version of the NVIDIA PyTorch container to ensure reproducibility.
   - Run the `docker pull nvcr.io/nvidia/pytorch:23.10-py3` command. This container includes all essential libraries optimized for NVIDIA GPUs, including PyTorch, CUDA, and cuDNN.
5. **Interactive Container Execution:**

   - Run the following command. The meaning of each flag is as follows:

     ```bash
     docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.10-py3
     ```

   - `--gpus all`: Exposes all available host GPUs to the container, forming the core of the GPU acceleration environment.
   - `-it`: Runs the container in interactive mode, allowing access to the container's internal shell.
   - `--rm`: Automatically deletes the container when terminated to keep the system clean.
   - `-v $(pwd):/workspace`: Mounts the current directory of the host machine to the `/workspace` directory inside the container. This is essential for saving work and accessing local data.

6. **GPU Access Verification:**
   - Run `nvidia-smi` inside the container shell.
   - If a table containing GPU, driver version, and CUDA version is output, the container is properly accessing the hardware.

### **Essential Python Library Installation**

Run the following pip commands inside the running container to install the NeMo and Hugging Face ecosystems:

```bash
# Install Hugging Face libraries
pip install transformers datasets accelerate

# Install full NVIDIA NeMo toolkit
pip install nemo-toolkit[all]
```

### **Troubleshooting Guide**

It's important to resolve the most common installation issues in advance for smooth workshop progress.

| Error Message                                                                                                  | Common Causes                                                                                                                                                   | Solutions                                                                                                                                                                                                                                                                                                                                                                                 |
| :------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| docker: Error response from daemon: OCI runtime create failed... nvidia-container-cli: initialization error... | Host NVIDIA driver issues; NVIDIA Container Toolkit not configured; Docker daemon restart needed.                                                               | 1. Check host driver: Verify that the latest NVIDIA driver is properly installed on the host machine. nvidia-smi should work on the host. 2. Configure toolkit: Run `sudo nvidia-ctk runtime configure --runtime=docker` followed by `sudo systemctl restart docker`. 3. WSL2 specifics: Ensure Linux GPU drivers are not installed inside WSL2. Windows drivers automatically interface. |
| Failed to initialize NVML: Driver/library version mismatch                                                     | Mismatch between NVIDIA driver version loaded in kernel and user-space library (libnvidia-ml.so) version. Often occurs when not rebooting after driver updates. | 1. Easiest solution: Reboot. System reboot is the most reliable way to synchronize kernel modules and user-space libraries. 2. Manual reload (advanced): Unload all NVIDIA kernel modules (sudo rmmod...) and reload them. Complex and risky, so rebooting is recommended. 3. Clean reinstall: Completely remove all NVIDIA drivers and install the latest version fresh.                 |
| WSL2 nvidia-smi execution error: couldn't communicate with the NVIDIA driver                                   | (Legacy issue) Early WSL2 versions didn't fully support NVML.                                                                                                   | 1. Update WSL kernel: Run `wsl --update` in PowerShell to get the latest kernel. 2. Windows update: Ensure you're using the latest build of Windows 10/11. 3. Update NVIDIA drivers: Install the latest drivers designed for WSL2 support.                                                                                                                                                |

Using pre-configured NGC containers goes beyond simple convenience; it's a fundamental practice that ensures scientific reproducibility and mitigates dependency issues. Attempts to manually install drivers, toolkits, and libraries in different environments for each developer easily lead to failures and time waste. NGC containers encapsulate the perfect combination of NVIDIA-optimized and tested drivers, CUDA libraries, cuDNN, and PyTorch. By having all participants start with the same container (nvcr.io/nvidia/pytorch:23.10-py3), we fundamentally block "it works on my machine" type problems. This containerized approach transforms complex and error-prone environment setup tasks into a single docker run command, providing all participants with a consistent baseline. This is both a cornerstone of modern MLOps and an important lesson in itself.

---

## **Part 3: First Encounter: Running LLMs with Hugging Face Transformers**

This section conducts the first hands-on coding practice that demonstrates the power of the built environment.

### **The Power of Simplicity: Pipeline API**

Hugging Face pipeline is the highest-level, easy-to-use API for inference. This API abstracts and provides three core stages: (1) tokenization (preprocessing), (2) model inference (forward pass), and (3) output decoding (postprocessing).

### **Practice 1: First Text Generation**

Here is the provided Python script:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Simple text generation pipeline
generator = pipeline("text-generation",
                     model="gpt2",
                     device=0 if torch.cuda.is_available() else -1)

# Text generation test
prompt = "The future of Artificial Intelligence is"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result['generated_text'])
```

- **Code Analysis:**
  - `pipeline("text-generation", model="gpt2",...)`: Instantiates the pipeline.
    - `task="text-generation"`: Specifies the desired task. The complete list of available tasks can be found in the documentation.
    - `model="gpt2"`: Specifies the model to use from Hugging Face Hub. gpt2 is a good starting point due to its small size and familiarity.
    - `device=0`: This is the core of GPU acceleration. It instructs the pipeline to load the model and data onto the first GPU (CUDA device 0).

### **Practice 2: Korean Text Generation**

Let's apply what we've learned to a Korean model. EleutherAI/polyglot-ko-1.3b is a good choice as a well-documented open-source (Apache 2.0 license) Korean LLM.

```python
# Korean text generation pipeline
korean_generator = pipeline("text-generation",
                           model="EleutherAI/polyglot-ko-1.3b",
                           device=0 if torch.cuda.is_available() else -1)

prompt = "대한민국 인공지능의 미래는"
result = korean_generator(prompt, max_length=50, num_return_sequences=1)
print(result['generated_text'])
```

### **Output Control: Generation Parameter Guide**

Basic generation results may be repetitive or meaningless. Using parameters that control decoding strategies, we can go beyond just "making it work" to "making it work well."

| Parameter                   | Description                                                                                                                                               | Impact on Output                                                                                                             | Recommended Use Cases                                                                                 |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| max_length / max_new_tokens | Maximum total length of output sequence or maximum number of new tokens to generate.                                                                      | Controls response length and prevents infinite loops. Set to appropriate values for application requirements.                |                                                                                                       |
| temperature                 | Real number greater than 0. Controls next token probability. Low values (<1.0) make the model more deterministic, high values (>1.0) make it more random. | Low: Predictable, conservative, repetitive. High: Creative, diverse, increased risk of meaningless results.                  | **Creative tasks:** 0.7 - 1.0. **Fact-based tasks:** 0.2 - 0.5.                                       |
| top_k                       | Integer. Filters vocabulary to the k most likely next tokens. The model samples from this reduced set.                                                    | Limits the selection pool to prevent very low probability tokens from being selected. Setting too low can hinder creativity. | Good starting point is 50. Use when you want to avoid strange word choices.                           |
| top_p (Nucleus Sampling)    | Real number between 0 and 1. Filters vocabulary to the smallest token set whose cumulative probability exceeds p.                                         | More dynamic than top_k. Adjusts vocabulary size based on next token predictability. Better general-purpose sampling method. | Good starting point is 0.9 to 0.95. OpenAI recommends changing either temperature or top_p, not both. |
| num_return_sequences        | Integer. Number of sequences to generate independently.                                                                                                   | Can generate multiple different completion sentences from the same prompt.                                                   | Useful for brainstorming or providing multiple options to users.                                      |
| no_repeat_ngram_size        | Integer. Prevents n-grams of this size from appearing more than once.                                                                                     | Directly addresses repetitive phrase problems, a common failure mode of greedy/beam search.                                  | Set to 2 or 3 to improve fluency and reduce obvious repetition.                                       |

Various decoding strategies represent approaches to managing the fundamental tension between generating creative and diverse text versus generating consistent and predictable text, rather than simply adjusting arbitrary values. The core of LLMs is the probability distribution of vocabulary for the next token. Pure greedy approaches that select only the most likely token are deterministic but often generate monotonous and unnatural text. To introduce creativity, we must sample from the distribution, but unconstrained sampling produces meaningless results. Therefore,

parameters like top_k, top_p, and temperature are tools for exploring the spectrum between pure exploitation (greedy search) and pure exploration (unconstrained sampling). Finding the right balance for a given task is the core skill of inference configuration.

---

## **Part 4: The Two Frameworks Story: NeMo and Hugging Face**

This section compares the two core frameworks of the workshop and provides an in-depth answer to the second checkpoint question.

### **Philosophical Deep Dive**

- **Hugging Face:**
  - **Core Philosophy:** Democratization and collaboration. The goal is to make cutting-edge ML accessible to everyone.
  - **Implementation:** Appears as a massive community-based Model Hub with hundreds of thousands of models and datasets, easy-to-use high-level APIs (pipeline), and open-source libraries that have become industry standards (transformers, datasets, accelerate).
- **NVIDIA NeMo:**
  - **Core Philosophy:** Performance, scalability, and enterprise readiness. The goal is to provide an end-to-end platform for efficiently building, customizing, and deploying generative AI models on NVIDIA hardware, from single GPU experiments to large-scale multi-node training clusters.
  - **Implementation:** Appears as deep integration with NVIDIA hardware and software stacks (CUDA, Transformer Engine, Megatron-Core), support for advanced parallel processing techniques (tensor, pipeline, data), and focus on the entire MLOps lifecycle including data curation (NeMo Curator) and optimized deployment.

### **Comparative Analysis: NeMo vs Hugging Face Transformers**

| Feature            | Hugging Face Transformers                                                                                                          | NVIDIA NeMo Framework                                                                                                                                           |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Primary Goal**   | Accessibility, community, ease of use                                                                                              | Performance, scalability, enterprise-grade training                                                                                                             |
| **Architecture**   | High-level API (pipeline) and modular Auto\* classes based on PyTorch/TensorFlow/JAX. Fundamentally framework-agnostic.            | Tightly integrated with NVIDIA's own high-performance backends like PyTorch Lightning and Megatron-Core, Transformer Engine.                                    |
| **Core Strengths** | **Ecosystem:** Unparalleled collection of models and datasets in the Hub. Very easy to start with inference and basic fine-tuning. | **Large-scale Performance:** Optimized for large-scale model training on thousands of GPUs. Provides excellent throughput and efficiency on NVIDIA hardware.    |
| **Ease of Use**    | **Very High.** Pipeline API is beginner-friendly. Extensive documentation and community support.                                   | **Intermediate to Advanced.** Steeper learning curve with emphasis on distributed training configuration (improved with NeMo 2.0's Python-based configuration). |
| **Configuration**  | Primarily programmatic through Python code. Uses TrainingArguments class for Trainer API.                                          | Historically YAML-based. NeMo 2.0 transitions to more flexible and powerful Python-based configuration system (Fiddle).                                         |
| **Use Cases**      | Rapid prototyping, inference applications, academic research, small to medium-scale model fine-tuning.                             | Foundation model pre-training from scratch, large-scale SFT/RLHF on proprietary data, enterprise R&D.                                                           |

### **Bridging the Gap: Interoperability and Coexistence**

Developers are not forced to choose "either this or that." The two frameworks are increasingly working in a complementary manner.

- **NeMo's AutoModel Feature:** This is a game-changing feature. It's a high-level interface within NeMo designed to directly load and fine-tune Hugging Face models without manual conversion steps. This enables "Day-0 support" for new models by combining Hugging Face Hub's vast ecosystem with NeMo's high-performance training environment.
- **Checkpoint Conversion Scripts:** When not using AutoModel, NeMo provides explicit scripts like convert_llama_hf_to_nemo.py to convert model weights between Hugging Face and NeMo's .nemo format.
- **Data Integration through HFDatasetDataModule:** NeMo provides a dedicated data module (HFDatasetDataModule) that can directly wrap and use datasets from the Hugging Face datasets library within NeMo training pipelines, simplifying the data loading process.

The increasing interoperability between NeMo and Hugging Face shows that the AI development ecosystem is maturing. This means that rather than competing as closed single platforms, specialized tools are being designed to work together. Hugging Face serves as a universal "marketplace" for sharing and discovering models and data, while NeMo serves as a specialized "high-performance engine." This modular, interoperable approach allows developers to use the most suitable tools for each stage of the LLM lifecycle, accelerating innovation for everyone.

---

## **Conclusion and Week 1 Team Challenge**

### **Week 1 Summary**

This week, we built a solid conceptual model of the LLM lifecycle, set up a fully functional GPU-based development environment, and gained hands-on experience directly running pre-trained models with configurable output.

### **Week 2 Preview**

Next week, we will cover in-depth analysis of data preparation using datasets and NeMo Curator, and performing supervised fine-tuning (SFT) on custom datasets.

### **Week 1 Team Challenge (Recommended)**

- **Objective:** Apply the techniques learned this week to a real team-based exploration task.
- **Tasks:**
  1. **Exploration:** Explore Hugging Face Hub together with team members.
  2. **Task Selection:** Choose a task of interest (e.g., summarization, sentiment analysis).
  3. **Find Korean Datasets:** Find Korean datasets suitable for the selected task.
     - _Sentiment Analysis Recommendation:_ e9t/nsmc (Naver Movie Review Corpus).
     - _Summarization Recommendation:_ Explore models like nglaura/koreascience-summarization or gogamza/kobart-summarization.
  4. **Find Korean Models:** Find pre-trained models suitable for the task. This could be general models like EleutherAI/polyglot-ko-1.3b or task-specific fine-tuned models.
  5. **Experiment:** Load models using the pipeline API in the NGC container environment and run inference on 3-5 examples from the selected dataset.
  6. **Discussion:** Analyze results with team members. Are the results good? How can they be improved? This discussion will be a perfect entry point for next week's fine-tuning topic.

## **References**

1. What are the features of Hugging Face's Transformers? - Milvus, accessed September 28, 2025, [https://milvus.io/ai-quick-reference/what-are-the-features-of-hugging-faces-transformers](https://milvus.io/ai-quick-reference/what-are-the-features-of-hugging-faces-transformers)
2. Hugging Face Transformers: Leverage Open-Source AI in Python, accessed September 28, 2025, [https://realpython.com/huggingface-transformers/](https://realpython.com/huggingface-transformers/)
3. Introduction to Hugging Face Transformers - GeeksforGeeks, accessed September 28, 2025, [https://www.geeksforgeeks.org/artificial-intelligence/Introduction-to-hugging-face-transformers/](https://www.geeksforgeeks.org/artificial-intelligence/Introduction-to-hugging-face-transformers/)
4. NVIDIA NeMo Framework, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/index.html](https://docs.nvidia.com/nemo-framework/index.html)
5. What Is The Difference Between NVIDIA NeMo Framework & NeMo Microservices?, accessed September 28, 2025, [https://cobusgreyling.medium.com/what-is-the-difference-between-nvidia-nemo-framework-nemo-microservices-9339dbb62226](https://cobusgreyling.medium.com/what-is-the-difference-between-nvidia-nemo-framework-nemo-microservices-9339dbb62226)
6. LLM Development: Step-by-Step Phases of LLM Training - lakeFS, accessed September 28, 2025, [https://lakefs.io/blog/llm-development/](https://lakefs.io/blog/llm-development/)
7. LLM Project Lifecycle: Revolutionized by Generative AI - Data Science Dojo, accessed September 28, 2025, [https://datasciencedojo.com/blog/llm-project-lifecycle/](https://datasciencedojo.com/blog/llm-project-lifecycle/)
8. Large Language Model Lifecycle: An Examination Challenges, accessed September 28, 2025, [https://www.computer.org/publications/tech-news/trends/large-language-model-lifecycle/](https://www.computer.org/publications/tech-news/trends/large-language-model-lifecycle/)
9. What are the Stages of the LLMOps Lifecycle? - Klu.ai, accessed September 28, 2025, [https://klu.ai/glossary/llm-ops-lifecycle](https://klu.ai/glossary/llm-ops-lifecycle)
10. The LLM Project Lifecycle: A Practical Guide | by Tony Siciliani - Medium, accessed September 28, 2025, [https://medium.com/@tsiciliani/the-llm-project-lifecycle-a-practical-guide-9117228664d4](https://medium.com/@tsiciliani/the-llm-project-lifecycle-a-practical-guide-9117228664d4)
11. LLM Development Life Cycle, accessed September 28, 2025, [https://muoro.io/llm-development-life-cycle](https://muoro.io/llm-development-life-cycle)
12. Analysis of the Hugging Face Transformers Library: Purpose and Component Classes : 1, accessed September 28, 2025, [https://medium.com/@danushidk507/analysis-of-the-hugging-face-transformers-library-purpose-and-component-classes-1-8f5bdc7a3b17](https://medium.com/@danushidk507/analysis-of-the-hugging-face-transformers-library-purpose-and-component-classes-1-8f5bdc7a3b17)
13. LLM Post-Training: A Deep Dive into Reasoning Large Language Models - arXiv, accessed September 28, 2025, [https://arxiv.org/pdf/2502.21321](https://arxiv.org/pdf/2502.21321)
14. Understanding the Effects of RLHF on LLM Generalisation and Diversity - arXiv, accessed September 28, 2025, [https://arxiv.org/html/2310.06452v2](https://arxiv.org/html/2310.06452v2)
15. Reinforcement Learning From Human Feedback (RLHF) For LLMs - Neptune.ai, accessed September 28, 2025, [https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms](https://neptune.ai/blog/reinforcement-learning-from-human-feedback-for-llms)
16. Summarization - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/tasks/summarization](https://huggingface.co/docs/transformers/tasks/summarization)
17. Run Hugging Face Models Instantly with Day-0 Support from NVIDIA ..., accessed September 28, 2025, [https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework/](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework/)
18. CUDA on WSL User Guide — CUDA on WSL 13.0 documentation, accessed September 28, 2025, [https://docs.nvidia.com/cuda/wsl-user-guide/index.html](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
19. Enable NVIDIA CUDA on WSL 2 - Microsoft Learn, accessed September 28, 2025, [https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
20. GPU support - Docker Docs, accessed September 28, 2025, [https://docs.docker.com/desktop/features/gpu/](https://docs.docker.com/desktop/features/gpu/)
21. 1. Overview — NVIDIA GPU Cloud Documentation, accessed September 28, 2025, [https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
22. Containers For Deep Learning Frameworks User Guide - NVIDIA Docs, accessed September 28, 2025, [https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
23. Troubleshooting — NVIDIA Container Toolkit - NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html)
24. Docker Nvidia Runtime error : r/archlinux - Reddit, accessed September 28, 2025, [https://www.reddit.com/r/archlinux/comments/1mqlv5k/docker_nvidia_runtime_error/](https://www.reddit.com/r/archlinux/comments/1mqlv5k/docker_nvidia_runtime_error/)
25. NVIDIA Docker - initialization error: nvml error: driver not loaded · Issue #1393 - GitHub, accessed September 28, 2025, [https://github.com/NVIDIA/nvidia-docker/issues/1393](https://github.com/NVIDIA/nvidia-docker/issues/1393)
26. Docker Fails to Launch GPU Containers with NVIDIA Runtime, but Podman Works, accessed September 28, 2025, [https://forums.docker.com/t/docker-fails-to-launch-gpu-containers-with-nvidia-runtime-but-podman-works/147966](https://forums.docker.com/t/docker-fails-to-launch-gpu-containers-with-nvidia-runtime-but-podman-works/147966)
27. GPU Troubleshooting Guide: Resolving Driver/Library Version Mismatch Errors, accessed September 28, 2025, [https://support.exxactcorp.com/hc/en-us/articles/32810166604183-GPU-Troubleshooting-Guide-Resolving-Driver-Library-Version-Mismatch-Errors](https://support.exxactcorp.com/hc/en-us/articles/32810166604183-GPU-Troubleshooting-Guide-Resolving-Driver-Library-Version-Mismatch-Errors)
28. How to resolve "Failed to initialize NVML: Driver/library version mismatch" error - D2iQ, accessed September 28, 2025, [https://support.d2iq.com/hc/en-us/articles/4409480561300-How-to-resolve-Failed-to-initialize-NVML-Driver-library-version-mismatch-error](https://support.d2iq.com/hc/en-us/articles/4409480561300-How-to-resolve-Failed-to-initialize-NVML-Driver-library-version-mismatch-error)
29. Failed to initialize NVML: Driver/library version mismatch - Reddit, accessed September 28, 2025, [https://www.reddit.com/r/freebsd/comments/18zhf55/failed_to_initialize_nvml_driverlibrary_version/](https://www.reddit.com/r/freebsd/comments/18zhf55/failed_to_initialize_nvml_driverlibrary_version/)
30. Nvidia NVML Driver/library version mismatch [closed] - Stack Overflow, accessed September 28, 2025, [https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch)
31. Hiccups setting up WSL2 + CUDA - NVIDIA Developer Forums, accessed September 28, 2025, [https://forums.developer.nvidia.com/t/hiccups-setting-up-wsl2-cuda/128641](https://forums.developer.nvidia.com/t/hiccups-setting-up-wsl2-cuda/128641)
32. Installing the NVIDIA Container Toolkit, accessed September 28, 2025, [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
33. Install NeMo Framework - NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
34. NVIDIA-NeMo/NeMo: A scalable generative AI framework built for researchers and developers working on Large Language Models, Multimodal, and Speech AI (Automatic Speech Recognition and Text-to-Speech) - GitHub, accessed September 28, 2025, [https://github.com/NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)
35. Quickstart - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour)
36. Pipelines - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/main_classes/pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
37. The pipeline API - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers.js/v3.0.0/pipelines](https://huggingface.co/docs/transformers.js/v3.0.0/pipelines)
38. pipelines - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers.js/api/pipelines](https://huggingface.co/docs/transformers.js/api/pipelines)
39. Pipeline - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/pipeline_tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)
40. EnverLee/polyglot-ko-1.3b-Q4_0-GGUF - Hugging Face, accessed September 28, 2025, [https://huggingface.co/EnverLee/polyglot-ko-1.3b-Q4_0-GGUF](https://huggingface.co/EnverLee/polyglot-ko-1.3b-Q4_0-GGUF)
41. EleutherAI/polyglot-ko-1.3b at main - Hugging Face, accessed September 28, 2025, [https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/tree/main)
42. EleutherAI/polyglot-ko-1.3b · Hugging Face, accessed September 28, 2025, [https://huggingface.co/EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
43. How to run a Hugging Face text generation AI model Locally? step-by-step tutorial, accessed September 28, 2025, [https://www.youtube.com/watch?v=Ez_bHdET0iw](https://www.youtube.com/watch?v=Ez_bHdET0iw)
44. Generation - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
45. How to generate text: using different decoding methods for language generation with Transformers - Hugging Face, accessed September 28, 2025, [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)
46. Temperature, top_p and top_k for chatbot responses - OpenAI Developer Community, accessed September 28, 2025, [https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542)
47. Utilities for Generation - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/en/internal/generation_utils](https://huggingface.co/docs/transformers/en/internal/generation_utils)
48. Transformers - Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
49. Using transformers at Hugging Face, accessed September 28, 2025, [https://huggingface.co/docs/hub/transformers](https://huggingface.co/docs/hub/transformers)
50. Hugging Face pitches HUGS as an alternative to Nvidia's NIM for open models - Reddit, accessed September 28, 2025, [https://www.reddit.com/r/AMD_Stock/comments/1gct7mt/hugging_face_pitches_hugs_as_an_alternative_to/](https://www.reddit.com/r/AMD_Stock/comments/1gct7mt/hugging_face_pitches_hugs_as_an_alternative_to/)
51. Master Generative AI with NVIDIA NeMo, accessed September 28, 2025, [https://resources.nvidia.com/en-us-ai-large-language-models/watch-78](https://resources.nvidia.com/en-us-ai-large-language-models/watch-78)
52. NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support, accessed September 28, 2025, [https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/](https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/)
53. Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities, accessed September 28, 2025, [https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/](https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/)
54. NVIDIA NeMo Framework Developer Docs, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/index.html)
55. Advantage of NEMO models & Toolkit #7073 - GitHub, accessed September 28, 2025, [https://github.com/NVIDIA/NeMo/discussions/7073](https://github.com/NVIDIA/NeMo/discussions/7073)
56. Configuring Nemo-Guardrails Your Way: An Alternative Method for Large Language Models, accessed September 28, 2025, [https://towardsdatascience.com/configuring-nemo-guardrails-your-way-an-alternative-method-for-large-language-models-c82aaff78f6e/](https://towardsdatascience.com/configuring-nemo-guardrails-your-way-an-alternative-method-for-large-language-models-c82aaff78f6e/)
57. Configure NeMo-Run — NVIDIA NeMo Framework User Guide - NVIDIA Docs Hub, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html)
58. RunBot's math-to-text on NVIDIA NeMo Framework AutoModel - LoRA - vLLM Forums, accessed September 28, 2025, [https://discuss.vllm.ai/t/runbots-math-to-text-on-nvidia-nemo-framework-automodel/637](https://discuss.vllm.ai/t/runbots-math-to-text-on-nvidia-nemo-framework-automodel/637)
59. Community Model Converter User Guide — NVIDIA NeMo ..., accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/checkpoints/user_guide.html](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/checkpoints/user_guide.html)
60. Checkpoint Conversion — NVIDIA NeMo Framework User Guide 24.07 documentation, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/checkpointconversion.html](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/checkpointconversion.html)
61. Checkpoint conversion - NeMo-Skills, accessed September 28, 2025, [https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion/](https://nvidia.github.io/NeMo-Skills/pipelines/checkpoint-conversion/)
62. HFDatasetDataModule — NVIDIA NeMo Framework User Guide, accessed September 28, 2025, [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/automodel/codedocs/hf_dataset_data_module.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/automodel/codedocs/hf_dataset_data_module.html)
63. e9t/nsmc · Datasets at Hugging Face, accessed September 28, 2025, [https://huggingface.co/datasets/e9t/nsmc](https://huggingface.co/datasets/e9t/nsmc)
64. nglaura/koreascience-summarization · Datasets at Hugging Face, accessed September 28, 2025, [https://huggingface.co/datasets/nglaura/koreascience-summarization](https://huggingface.co/datasets/nglaura/koreascience-summarization)
65. gogamza/kobart-summarization - Hugging Face, accessed September 28, 2025, [https://huggingface.co/gogamza/kobart-summarization](https://huggingface.co/gogamza/kobart-summarization)
