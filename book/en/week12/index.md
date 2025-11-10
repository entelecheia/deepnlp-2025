# Week 12: AI Regulation and Responsible AI

## Lecture Overview

Welcome to the 12th-week lecture on Deep Learning for Natural Language Processing. Today, we will cover the most critical and complex intersection where the NLP technologies we have learned meet the real world: the issues of regulation and accountability. As of November 2025, we are in the midst of an inflection point, where the "Wild West" era of AI technology is ending and the "Age of Law" is dawning.

The goals of this lecture are twofold. First, to dissect the global regulatory framework centered on the EU AI Act—which was enacted on August 1, 2024, and began full-scale implementation in August 2025—to clearly understand what you, as developers, must comply with. Second, to review the latest research (2025) on Privacy-Enhancing Technologies (PETs) to draw a blueprint for how to technically implement responsible AI that complies with these laws.

You are required to complete the "EU AI Act Compliant LLM Service Design" assignment this semester. This lecture will provide the legal, technical, and architectural foundation necessary to complete that assignment.

---

## Module 1: The 2025 AI Governance & Regulatory Landscape

2025 is the first year that AI regulation has transitioned from abstract ethical guidelines to binding law. At the center of this change is the EU AI Act.

### 1.1. The New Global Standard: The EU AI Act's Structure and Core

The EU AI Act, which entered into force on August 1, 2024, and is being implemented in phases as of 2025, is the world's first comprehensive AI regulation. This law, much like the "CE mark," applies to all "Providers" and "Deployers" who intend to deploy or place on the market AI systems in the EU. This applies regardless of whether the AI system was developed within the EU or in a third country, and it also applies if the "output" is used within the EU.

#### 1.1.1. Core Architecture: The 4-Tier Risk-Based Approach

The most prominent feature of the EU AI Act is its risk-based approach, which classifies all AI systems into four tiers based on their risk level. The stringency of the regulation is directly proportional to the risk level.

1. **Unacceptable Risk:** These are AI systems that pose a clear threat to the values and fundamental rights of the EU. Such systems are entirely banned from being placed on the market, put into service, or used.
2. **High Risk:** These are AI systems that could have a significant impact on human fundamental rights, health, safety, or the core functions of society. The majority of the Act deals with the obligations for these high-risk systems.
3. **Limited Risk:** This applies to systems where users must be aware that they are interacting with an AI. Systems like chatbots or deepfakes fall into this category, and light transparency obligations are imposed.
4. **Minimal Risk:** This includes the majority of AI applications with little to no risk, such as AI-based video games or spam filters. These systems are effectively unregulated, and only voluntary adherence to codes of conduct is recommended.

#### 1.1.2. In Effect Since Feb 2025: "Unacceptable Risk" and its NLP Relevance

The prohibition clause was the first to take legal effect. As of February 2, 2025, the use of "Unacceptable Risk" AI became illegal in the EU. This has an immediate impact on NLP researchers and developers.

The main prohibitions directly related to NLP are as follows:

- **Social Scoring:** Prohibits systems used by public authorities to evaluate or classify individuals based on their social behavior, trustworthiness, or personal characteristics, leading to detrimental treatment (e.g., restricted access to services).
- **Manipulative AI:** Prohibits AI systems that use subliminal techniques beyond a person's consciousness or exploit the vulnerabilities of specific groups (e.g., based on age, disability, socioeconomic status) to materially distort a person's behavior.
- **Emotion Recognition in the Workplace and Education:** According to European Commission (EC) guidelines released in early February 2025, AI systems that infer emotions of individuals in workplaces or educational institutions are prohibited. Except for very specific medical or safety reasons (e.g., driver fatigue detection), this effectively blocks the commercialization of NLP-based "employee emotion analysis solutions," "student stress monitoring," or "job candidate emotion analysis" products within the EU.

This prohibited list is like a clear legal and social "death sentence" for certain directions in NLP research. While technically possible, the EU has declared that such technologies are socially unacceptable.

#### 1.1.3. Key Obligations for "High-Risk AI Systems" (HRAIS)

This is the category where most commercial NLP systems you will design (especially in finance, HR, and education) are likely to fall. If classified as HRAIS, the provider (i.e., the developer) must pass a strict ex-ante conformity assessment before market release and fulfill the following key obligations.

The 7 Key Obligations for HRAIS under the EU AI Act (Art. 8–17):

1. **Risk Management System (Art. 9):** A continuous process must be established and documented to identify, evaluate, and mitigate risks throughout the AI system's entire lifecycle.
2. **Data and Data Governance (Art. 10):** High-quality training, validation, and testing datasets must be used to minimize discriminatory outcomes. Datasets must be relevant, sufficiently representative, and, to the extent possible, free of errors and complete for the intended purpose. (We will discuss in Module 3.2 how this provision conflicts with GDPR.)
3. **Technical Documentation (Art. 11):** Detailed technical documentation must be drawn up and kept up-to-date, containing all information necessary for authorities to assess the system's compliance (e.g., architecture, performance, dataset specifications).
4. **Record-Keeping / Logs (Art. 12):** The system's operation must be automatically recorded, and logs (e.g., the basis for decisions) must be generated and stored to ensure traceability of results.
5. **Transparency / Info for Deployers (Art. 13):** Clear and adequate information regarding the system's capabilities, limitations, correct usage, and interpretation methods must be provided to the "Deployer" who will actually operate the system.
6. **Human Oversight (Art. 14):** The system must be designed to allow for appropriate human-in-the-loop intervention and oversight while in use. A human must be able to interrupt, disregard, or reverse the AI's decision.
7. **Accuracy, Robustness, and Cybersecurity (Art. 15):** The system must demonstrate a high level of accuracy appropriate for its intended purpose, be robust against errors or external adversarial attacks, and possess an appropriate level of cybersecurity.

#### 1.1.4. [Critical] NLP-Specific "High-Risk" Use Cases (Annex III)

So, which NLP systems are "High-Risk (HRAIS)"? Annex III of the AI Act lists 8 specific use cases that are considered high-risk by default. This list clearly shows that NLP and profiling technologies are a core target of the legislation.

- **Education and Vocational Training:**
  - Systems that determine access, admission, or assignment to educational institutions (e.g., AI admissions officers, AI application screeners).
  - Systems that evaluate learning outcomes of students (e.g., AI-based automated grading, AI tutor analysis of student performance).
- **Employment, Workers Management, and Access to Self-Employment:**
  - AI systems for recruitment or selection (e.g., targeting job ads, analyzing and filtering CVs, evaluating interview candidates).
  - Systems for making promotion and termination decisions, task allocation based on personal traits or behavior, and performance monitoring and evaluation.
- **Access to Essential Private and Public Services:**
  - AI systems for Credit Scoring or evaluating creditworthiness (excluding for financial fraud detection).
  - Systems used by public authorities to evaluate, reduce, or revoke eligibility for public benefits and services (e.g., social security, welfare).
  - Systems used for risk assessment and pricing for life and health insurance.
- **Law Enforcement:**
  - Polygraphs (lie detectors) and similar tools.
  - Systems to evaluate the reliability of evidence during criminal investigations or prosecutions.
  - Systems for assessing the risk of crime or profiling individuals based on personality traits or past criminal behavior.

As is evident from this list, the high-risk and prohibited provisions of the EU AI Act are less focused on physical AI (robots, drones) and more on profiling and automated decision-making systems that evaluate and predict human language, behavior, and characteristics, and as a result, determine an individual's access to opportunities (admission, hiring, loans). This means the AI Act fundamentally has a very strong character of being an "NLP Regulation Act."

### 1.2. The 2025 Flashpoint: Regulating General-Purpose AI (GPAI)

On August 2, 2025, the most controversial provisions of the EU AI Act—the obligations for General-Purpose AI (GPAI) models—officially took effect. These provisions place direct responsibility on the companies (e.g., OpenAI, Google, Anthropic, Meta) that develop and provide large-scale models like GPT-4, Llama 3, and Claude 3, also known as Foundation Models. As of November 2025, this is the hottest regulatory issue in the AI industry.

#### 1.2.1. July 2025 Guidelines: Defining "GPAI"

Just before the GPAI obligations of the AI Act took effect, on July 18, 2025, the European Commission (EC) released draft Guidelines to clarify the scope and definition of these obligations.

According to these guidelines, a model trained with a cumulative computational load of $10^{23}$ FLOPs (floating-point operations per second) or more, capable of performing a wide range of tasks such as text, audio, or image/video generation, is defined as a "GPAI model."

#### 1.2.2. Obligations for ALL GPAI Providers

Even providers of smaller GPAI models without "systemic risk" must comply with the following four key obligations if they exceed the $10^{23}$ FLOPs threshold:

1. **Technical Documentation:** Prepare and maintain detailed technical documentation describing the model's training, testing, and evaluation processes and results, and provide it to the AI Office upon request.
2. **Information for Downstream Providers:** Provide sufficient information about the model's capabilities, limitations, and usage to downstream developers (e.g., a startup building an HRAIS) so they can comply with their own AI Act obligations (e.g., HRAIS technical documentation).
3. **Copyright Policy:** Establish and implement a policy to respect and comply with EU copyright law (e.g., honoring "opt-out" requests from copyright holders during data collection).
4. **Training Data Summary:** Publicly publish a summary of the data used to train the model, following the official template released by the AI Office on July 24, 2025.

#### 1.2.3. Additional Obligations for GPAI with "Systemic Risk"

This is the special regulation targeting SOTA (state-of-the-art) large-scale models like GPT-4, Claude 3, and Gemini Ultra.

- **Definition:** The July 2025 guidelines presume that a GPAI model trained with a cumulative computational load of $10^{25}$ FLOPs or more has "Systemic Risk."
- **Additional Obligations (Art. 55):** Providers of these systemic risk models (e.g., OpenAI, Google, Anthropic) must adhere to the four basic obligations above, plus four much stronger additional obligations:
  1. **Perform Model Evaluations:** Conduct model evaluations according to state-of-the-art (SOTA) standards. This includes internal and external adversarial testing to identify biases, robustness, and potential misuse risks.
  2. **Assess & Mitigate Systemic Risks:** Identify, assess, and take appropriate mitigation measures for any EU-level systemic risks the model could cause (e.g., threats to democratic processes, public health, national security).
  3. **Track & Report Serious Incidents:** Track, document, and report serious incidents that occur after the model is deployed to the EU AI Office and relevant national authorities without delay. (On November 4, 2025, the EC released a template for this reporting.)
  4. **Ensure Adequate Cybersecurity:** Ensure an appropriate level of cybersecurity protection for the model itself as well as the physical infrastructure where the model weights are stored.

#### 1.2.4. Role of the July 2025 "Code of Practice"

How can a provider comply with these complex and ambiguous obligations (e.g., "adequate" cybersecurity, "state-of-the-art" model evaluation)? On July 10, 2025, the EC approved and published a "Code of Practice for GPAI," drafted by independent experts from industry, academia, and civil society.

- This code is legally "voluntary." A provider can demonstrate compliance through other means.
- However, if a provider adheres to and signs this code, they gain the powerful benefit of a "safe harbor" or "presumption of conformity"—they are considered to have fulfilled their obligations under the AI Act (Art. 53, 55).
- The code consists of three chapters: Transparency, Copyright, and (for systemic risk models) Safety and Security, specifying concrete measures to fulfill each obligation.

#### 1.2.5. [Critical] The 2025 "Compliance Crisis"

As of November 2025, the GPAI regulation is legally in effect but faces a severe compliance crisis in its execution. This is due to the Act's complex implementation schedule.

- **"Grandfathering":** Providers of GPAI models that were already on the market before August 2, 2025 (e.g., GPT-4, Llama 3, Claude 3) receive a 2-year grace period, until August 2, 2027.
- **"The Downstream Tragedy":** However, a downstream developer who builds a "High-Risk AI System (HRAIS)" (e.g., a hiring solution) after August 2, 2025, using one of these "grandfathered" models (like GPT-4), must comply with the HRAIS regulations (see Module 1.1.3) immediately.
- **"Lack of Vendor Transparency":** This is the problem you will face. To create the HRAIS technical documentation, the downstream developer needs information from the upstream model (GPT-4), such as its training data summary and bias testing results. But the upstream provider (OpenAI) has no legal obligation to provide that information until 2027.

In this regulatory gap and supply chain crisis, the "Code of Practice" published in July 2025 functions not as a mere "voluntary" code, but as a de facto "essential business certification." Downstream companies building HRAIS cannot wait until 2027, so they are forced to choose "safe" upstream models from providers (e.g., Google, Anthropic, Mistral) who have voluntarily signed this Code and provide the necessary documentation. As of 2025, signing this Code has become the only way to prove you are a "trustworthy partner" in the GPAI market.

### 1.3. The Great Divergence: Global Regulatory Comparison, 2025

2025 is the year the myth of the "Brussels Effect"—the expectation that the EU's strict standards would become the de facto global standard—was broken. We are now in an era of 3-4 clearly distinct regulatory blocs.

#### 1.3.1. United States: "Pro-Innovation" and Deregulation

- **Background:** The Trump administration, which took office in January 2025, views AI as key to economic and geopolitical leadership and has rescinded the previous Biden administration's "Safe, Secure, and Trustworthy AI" Executive Order (E.O. 14110).
- **Key Stance:** In July 2025, the White House announced "America's AI Action Plan." The plan's core is to avoid "excessive regulation" and promote "pro-growth AI policies."
- **Policy:** The NIST AI Risk Management Framework (NIST AI RMF 2.0) remains a voluntary guideline with no legal force. The administration even directed NIST to remove "ideological biases" such as "misinformation" and "Diversity, Equity, and Inclusion (DEI)" from the RMF.
- **Friction with EU:** The US administration has openly criticized the EU AI Act as "handwringing about safety," arguing it discriminates against US tech companies and stifles innovation.
- **November 2025 Status:** As a result of this intense pressure, it was reported on November 7, 2025, that the EU Commission has confirmed a "reflection... ongoing" about delaying some provisions of the AI Act (e.g., fines for HRAIS violations) until August 2027 or granting a one-year "grace period."

#### 1.3.2. South Korea: A "Third Way" of Innovation and Regulation

- **Legal Status:** South Korea enacted its "AI Basic Act" (Official name: Basic Act on Artificial Intelligence and Creation of a Trust Base) on January 21, 2025, set to take effect on January 22, 2026. This is the world's second comprehensive AI law after the EU's.
- **Key Stance:** It takes a "balanced" approach, different from the EU's "risk" focus. It prioritizes the "promotion" of the AI industry ("promote first, regulate later") and aims to impose "minimal regulation" only on "High-Impact AI" systems that significantly affect public life, safety, and fundamental rights.
- **Key Obligations:** "High-Impact AI" is defined similarly to the EU's "high-risk" (e.g., healthcare, hiring, loan screening). These systems have obligations to conduct impact assessments, build risk management systems, ensure human oversight, and "label" AI-generated content and notify users.
- **Difference:** It is an "innovation-friendly" regulatory model, with lower penalties than the EU and a greater emphasis on fostering the AI industry and supporting data infrastructure (e.g., training data).

#### 1.3.3. China: State-Centric Governance

- **Key Stance:** China exhibits a completely different, state-centric, and "social control" oriented approach compared to the EU and US. It views AI as a core tool for national competitiveness and maintaining social stability.
- **Legal Status:** Starting with the 2023 "Interim Measures for the Management of Generative AI Services," China is implementing strong regulations as of 2025.
- **Key Obligations:**
  1. **Content Control:** Content must reflect core socialist values and is prevented from generating illegal or harmful content (e.g., threats to national security, criticism of the Communist Party).
  2. **Data Sourcing:** Must ensure the legality of training data and respect others' intellectual property rights.
  3. **Explicit Labeling:** Under the "Labeling Measures" implemented in H2 2025, all AI-generated content must carry both "explicit" (e.g., watermarks, text notifications) and "implicit" (e.g., metadata) labels.

#### 1.3.4. Comparative Analysis of Global AI Regulations, 2025

| Feature                 | European Union (EU)                                                                                                                   | United States (US)                                                          | South Korea (ROK)                                                        | China (PRC)                                                               |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| **Core Philosophy**     | Fundamental rights protection, Trust, Human-centric                                                                                | Market innovation, Geopolitical leadership, Deregulation                 | Balance of innovation and trust ("Promote first, regulate later")     | Social stability, State control, Technological sovereignty             |
| **Legal Status**        | **Mandatory Law (Act)** (In force Aug 2024)                                                                                        | **Voluntary Guidelines** (NIST AI RMF)                                   | **Mandatory Law (Act)** (Enforced Jan 2026)                          | **Mandatory Administrative Measures** (Partially in effect)            |
| **Risk Classification** | 4-Tier Risk-Based (Unacceptable/High/Limited/Minimal)                                                                               | Unitary Risk Management Framework (RMF)                                  | 2-Tier (High-Impact / Other)                                          | No risk-based tiers (Content-based regulation)                         |
| **GenAI Regulation**    | Strong obligations for **GPAI (>$10^{23}$ FLOPs)** & **Systemic Risk (>$10^{25}$ FLOPs)** (documentation, evaluation, reporting) | **No regulation**. (Encourages open-source, removal of ideological bias) | "Labeling" and "Transparency" obligations for Generative AI           | "Labeling" and strong "Content Control" obligations for Generative AI  |
| **2025 Status**         | GPAI rules in effect (8/2), "Supply chain crisis" begins, Discussing "postponement" due to US pressure                          | "America's AI Action Plan" (7/25), Deregulation stance established       | "AI Basic Act" enacted (1/25), Preparing enforcement decrees for 2026 | "Labeling Measures" in effect, Global governance plan announced (7/25) |

This "Great Divergence" is forcing an architectural divergence upon global AI companies as of 2025. Companies can no longer build a "one-size-fits-all" responsible AI model. They must now consider at least three different versions: (1) **EU Version:** A "high-trust" model with strict documentation, bias audits, and human oversight built-in, (2) **US Version:** A "high-performance" model focused on capabilities and innovation, and (3) **China Version:** A "high-control" model with robust content filtering and labeling built-in. This is not just a problem for the legal team; it has become a core engineering challenge requiring different model architectures, data governance, and deployment strategies for each region.

### Checkpoint Questions

- What are the four risk tiers in the EU AI Act, and how do they differ in regulatory stringency?
- Why are most commercial NLP systems likely to be classified as "High-Risk AI Systems" (HRAIS)?
- What is the difference between a GPAI model and a GPAI model with "systemic risk"?
- Explain the "compliance crisis" facing downstream developers building HRAIS after August 2, 2025.
- How do the regulatory approaches of the EU, US, South Korea, and China differ in their core philosophy and legal status?

---

## Module 2: Technical Deep Dive: Privacy-Enhancing Technologies (PETs) for LLMs

Legal compliance doesn't end with paperwork from the legal team. The privacy and safety required by law must be implemented in code. We will now examine how Privacy-Enhancing Technologies (PETs) are evolving in the LLM era, based on the latest 2025 research.

### 2.1. Differential Privacy (DP): Learning "Patterns," Not Data

Differential Privacy (DP) is a powerful mathematical definition that ensures "no one can tell whether any specific individual's data was included in the training set by looking at the algorithm's output (e.g., model weights, predictions)." It prevents personal information exposure by injecting statistically calibrated noise into the results, all within a privacy budget ($\epsilon$ and $\delta$).

#### 2.1.1. The LLM-Era Threat: Embedding Inversion Attacks (EIAs)

As Retrieval-Augmented Generation (RAG) systems become commonplace, users' sensitive queries (e.g., "search for tax evasion laws") are converted into text embeddings and sent to cloud vector databases. In the past, these embeddings were considered safe.

However, recent research (2025) has shown that Embedding Inversion Attacks (EIAs) can substantially reconstruct the original text from the embedding vector alone. This implies that the embeddings stored in vector DBs can themselves be a sensitive information leak.

Research like "EntroGuard," published on arXiv in March 2025, offers a solution. It injects DP-based statistical perturbation into the embedding before it is sent to the cloud. This noise is designed to maintain vector search (RAG) accuracy as much as possible, while simultaneously forcing an EIA attacker to recover high-entropy (i.e., meaningless) text instead of the original sensitive text.

#### 2.1.2. The 2025 Key Trend 1: Differentially Private Synthetic Data Generation

In the past, DP was mainly about Private Training (e.g., DP-SGD), which injected noise into the gradients during the model training process. This was very complex, consumed a large privacy budget, and significantly degraded the resulting model's utility.

The new paradigm in 2025 is Private Data Generation. Instead of training a model directly on sensitive raw data, this approach applies DP to generate Synthetic Data that only contains the statistical properties of the original. This "safe" synthetic data is then used for model training.

- **Google's Approach (March 2025):** Google Research announced an "inference-only" DP synthetic data generation method. It takes multiple sensitive raw data points (e.g., user queries), formats them into a prompt, and feeds them into a pre-trained LLM. The LLM's next-token predictions (logits) are then "privately aggregated" via a DP mechanism (e.g., the exponential mechanism) to sample a synthetic next token. This process is repeated to generate synthetic data that follows the original patterns but protects individual privacy.
- **Microsoft/ICLR Research (2024):** This presented a "training-free" approach that treats a pre-trained foundation model (e.g., GPT-4) as a "black-box" API and generates synthetic data through DP queries.

#### 2.1.3. The 2025 Key Trend 2: Private Aggregate Trend Analysis

The pinnacle of DP's commercial application appeared in Apple Intelligence, announced in June 2025. Apple faces the contradictory challenge of upholding its strong privacy principle of "never collecting user data" while simultaneously needing to "improve the user experience."

- **Apple's Approach:**
  1. **On-Device Processing:** Apple does not collect users' on-device data (e.g., email content, notifications). Analysis needed to improve features like "email summarization" is performed locally on the user's device.
  2. **Private Updates:** "Trends" or "updates" useful for model improvement (e.g., gradients, specific patterns) are generated on-device.
  3. **DP Noise Injection:** This "update" is "anonymized" via a DP algorithm before it leaves the device.
  4. **Aggregation:** Apple's servers only receive these anonymized aggregate trends. They do not receive any individual user's raw data. This aggregated information is statistically significant, but it is mathematically impossible to identify any specific individual from it.
- **Result:** Through DP, Apple simultaneously achieves two contradictory goals: (1) the powerful privacy marketing claim that it "does not see user data" and (2) the engineering objective of "improving models with user data." DP is the legal and ethical shield that resolves this contradiction.

As of 2025, the mainstream application of DP has shifted away from the complexity of "model training" itself and moved toward "DP synthetic data generation" and "DP trend analysis." This has elevated DP from a technical challenge to a business process solution.

### 2.2. Federated Learning (FL): Training Without Moving Data

Federated Learning (FL) is a fundamentally different approach to privacy. Instead of sending data to a central server, the model (or model updates) is sent to each client (e.g., smartphone, hospital, bank) to be trained on local data. Only the trained model weights (or gradients) are sent back to the server to be aggregated. The data always stays local.

#### 2.2.1. The LLM Challenge: Communication & Computation Bottlenecks

Early FL (e.g., the FedAvg algorithm) assumed small models (e.g., mobile keyboard prediction). However, the emergence of LLMs with 70B or 175B parameters (e.g., Llama 3, GPT-3.5) made traditional FL nearly impossible.

- **Communication Cost:** Sending an entire LLM (hundreds of GBs) to each client (e.g., a hospital) and then sending the gradient updates (hundreds of GBs) back to the server requires enormous network bandwidth.
- **Computation Cost:** Each client (e.g., an individual hospital's server) must have the high-performance GPU infrastructure necessary to fine-tune a 70B model.
- **Data Heterogeneity (Non-IID):** Each client's data distribution is very heterogeneous (Non-IID, Not Independent and Identically Distributed), meaning a simple FedAvg approach may fail to converge or even degrade model performance.

#### 2.2.2. The 2025 Solution 1: "Federated PEFT"

As of 2025, the most promising solution for FL with LLMs is its combination with PEFT (Parameter-Efficient Fine-Tuning), especially LoRA (Low-Rank Adaptation).

- **Core Idea:** You don't federate the entire 70B model. The 70B-parameter pre-trained LLM is frozen. Only the lightweight LoRA adapters, which drastically reduce the number of trainable parameters, are federated.
- **How it Works:**
  1. **Distribution:** The server distributes the massive "frozen" pre-trained model (e.g., Llama 3 70B) to all clients (e.g., Hospital A, B, C) just once.
  2. **Local Training:** Each client (Hospital A) trains only the small LoRA adapter (e.g., 0.1% of the original model size, ~20-100MB) on its own local private data (patient records). The 70B model body is not touched.
  3. **Update:** The client sends only the 20MB LoRA adapter weights back to the server, not the 70B parameters (hundreds of GBs).
  4. **Aggregation:** The server "averages" (e.g., FedAvg) only these small LoRA adapters collected from hospitals worldwide to create a "Global LoRA adapter."
  5. This "Global adapter" is then sent back to the clients for the next round of training.
- This method reduces communication costs by thousands of times and shows strong performance even in Non-IID data environments.

#### 2.2.3. The 2025 Solution 2: "Layer-Skipping FL"

This is another PEFT-based approach, published on arXiv in April 2025.

- **Core Idea:** Instead of adding LoRA adapters, this method freezes (skips) some layers of the pre-trained LLM and fine-tunes only selected specific layers.
- **Performance:** When applied to a LLaMA 3.2-1B model, this approach reduced communication costs by ~70% while keeping performance degradation within 2% of centralized training. This proved to be a highly practical solution for multiple institutions collaboratively training on domain-specific data, such as in healthcare NLP (e.g., i2b2, MIMIC-III datasets), without sharing private data.

The combination of "PEFT (LoRA) + FL" is not just an optimization; it's a paradigm shift. It achieves "global generalization" and "local specialization" simultaneously. The server improves general domain knowledge (e.g., medicine) via the "Global LoRA," while each client (hospital) retains its own "Local LoRA," which is highly specialized for its own data. Federated Learning has thus evolved into a dual-purpose solution that improves the central model while also providing a "customized private model" to each client.

### 2.3. Homomorphic Encryption (HE): Practicalizing the "Holy Grail"

Homomorphic Encryption (HE) is the "dream" encryption technology that allows one to perform desired computations (like addition and multiplication) directly on encrypted data (ciphertext). Decrypting the encrypted result yields the same output as if the operation had been performed on the original plaintext data. Using this, a client can send their sensitive data encrypted to a server, the server can perform operations (e.g., LLM inference) without ever seeing the raw data, and then return the encrypted result to the client.

#### 2.3.1. The Practicality Barrier: 10,000x+ Overhead

HE has a fatal flaw when applied to massive neural networks like LLMs: enormous computational overhead.

- One 2025 paper notes that HE-based LLM inference is at least 10,000 times slower than plaintext inference.
- **Reason:** FHE (Fully Homomorphic Encryption) is relatively efficient for linear operations (e.g., nn.Linear, matrix multiplication). However, it is extremely inefficient for the non-linear activation functions (e.g., ReLU, GeLU, SiLU, Softmax) that are at the core of the Transformer architecture.
- Approximating these non-linear operations in an encrypted state causes the noise accumulated in the ciphertext to grow exponentially. To reset this, an ultra-high-cost operation called bootstrapping is required. An LLM has hundreds of layers, potentially requiring hundreds or thousands of bootstrapping operations for a single inference.

#### 2.3.2. The 2025 Solution 1: HE-Friendly Model Architectures

To solve this, research is underway to change the model architecture itself to be more friendly to HE operations, rather than just encrypting a standard Transformer.

- **Replacing Non-linear Functions:** ReLU or GeLU activation functions are replaced with low-degree polynomial approximations, which are easily computed under HE.
- **Changing the Attention Mechanism:** The complex attention mechanism, which includes Softmax, is replaced with a Gaussian kernel or a simple polynomial attention to optimize the computation.
- An October 2024 arXiv study showed that combining LoRA fine-tuning with a Gaussian kernel could improve the HE-based Transformer's inference speed by 2.3x and its fine-tuning speed by 6.94x.

#### 2.3.3. The 2025 Solution 2: "Safhire" Hybrid HE Inference

"Safhire," published on arXiv in September 2025, presents the most practical solution to date.

- **Core Idea:** It separates what HE does well (linear operations) from what it does poorly (non-linear operations) and has the server and client share the workload.
- **How it Works:**
  1. **Client:** Encrypts the input ($Enc(x)$) and sends it to the server.
  2. **Server (Encrypted):** Performs only the HE-friendly linear operations (e.g., nn.Linear) in the encrypted state. ($Enc(z) = W \cdot Enc(x) + b$)
  3. **Server:** When it's time for a non-linear activation (e.g., ReLU), it sends the encrypted result ($Enc(z)$) back to the client.
  4. **Client (Plaintext):** Decrypts $Enc(z)$ to get $z$, and quickly performs the HE-unfriendly non-linear operation $a = ReLU(z)$ in plaintext locally.
  5. **Client:** Re-encrypts the activated result $a$ ($Enc(a)$) and sends it back to the server to request the next layer's linear operation.
- This "client-server-client" round-trip completely eliminates the expensive bootstrapping, bringing HE inference down to a practical level.
- This hybrid approach has shifted the HE trade-off from "extreme computational overhead" to "manageable network latency overhead." This has opened the door to applying HE in real services like RAG.

### Checkpoint Questions

- What is differential privacy, and how does it protect individual data while allowing pattern learning?
- How do Embedding Inversion Attacks (EIAs) threaten RAG systems, and how can DP mitigate this threat?
- Explain the difference between "Private Training" (DP-SGD) and "Private Data Generation" approaches to differential privacy.
- What are the main challenges of applying Federated Learning to large language models, and how does "Federated PEFT" address them?
- Why is Homomorphic Encryption computationally expensive for LLMs, and how does the "Safhire" hybrid approach solve this problem?

---

## Module 3: Industry Case Studies: Designing Domain-Specific NLP Solutions

We will now combine the Regulations from Module 1 and the Technologies from Module 2 to examine specific blueprints for designing Responsible LLM Solutions in particular industry domains.

### 3.1. Healthcare: Designing a HIPAA-Compliant LLM Chatbot

#### 3.1.1. The Regulation & Problem

- **Law:** USA HIPAA (Health Insurance Portability and Accountability Act).
- **Key Concepts:**
  1. **PHI (Protected Health Information):** HIPAA defines 18 personal identifiers as PHI (e.g., name, all types of dates, phone numbers, addresses, medical record numbers, etc.).
  2. **BAA (Business Associate Agreement):** A legal contract required when a "Covered Entity" (like a hospital) entrusts PHI processing to a third-party "Business Associate" (e.g., cloud provider, EMR vendor, AI company). This contract legally ensures the third party also complies with HIPAA security rules.
- **The Problem:** Most public LLM API providers like OpenAI (ChatGPT) and Anthropic do not sign BAAs for their standard services. Therefore, if a doctor copies a patient's chart (containing PHI), pastes it into the ChatGPT web interface, and asks, "Summarize this patient's record," it is a severe HIPAA violation because PHI was transferred to a third party without a BAA.

#### 3.1.2. Technical Solution: The "De-ID + Self-Hosted RAG" Architecture

The way to solve this is to use a vendor that will sign a BAA (e.g., Google's Med-PaLM 2, or medical-specific vendors like BastionGPT) or to build a Self-Hosted architecture that ensures PHI never leaves the institution's firewall.

**Architecture Blueprint for a HIPAA-Compliant LLM Chatbot:**

1. **Infrastructure:** Build an isolated VPC (Virtual Private Cloud) within a HIPAA-compliant cloud (e.g., AWS, Azure, GCP). All services (LLM, DB, API) communicate only within this VPC internal network.
2. **Encryption:** HIPAA requires encryption for data in-transit and at-rest. Use TLS 1.3 or higher to protect data in-transit, and use AES-256 and FIPS 140-2 validated encryption modules to protect data at-rest in the DB.
3. **Database:** The patient's original PHI (e.g., EMR/EHR) is stored in an encrypted RDBMS or vector DB within this VPC.
4. **LLM:** A model like Llama 3 or (BAA-covered) Med-PaLM 2 is self-hosted within the VPC, guaranteeing that no data ever leaves the institution's firewall.
5. **Access Control:** Apply the "principle of least privilege" using RBAC (Role-Based Access Control) and MFA (Multi-Factor Authentication) to strictly control access, ensuring only authorized medical staff (e.g., the treating physician) can access that patient's PHI.
6. **Audit:** All access attempts to PHI and all AI queries and responses must be recorded in an immutable audit log.

**The Core NLP Pipeline: De-identification (De-ID) Gateway:**

This is the technical key to preventing PHI leaks.

1. **(Input Query):** A doctor asks the chatbot, "Summarize the cardiac exam results for patient John Snow (PHI) from October 1, 2025 (PHI)."
2. **(De-ID Filter):** Before this query goes to the LLM, it passes through a De-ID (De-identification) Engine. This engine uses a high-performance NER (Named Entity Recognition) model to detect the 18 PHI identifiers in real-time.
3. **(Masking / Obfuscation):** The detected PHI is masked (e.g., `[*******]`, `<NAME>`) or obfuscated (e.g., "John Snow" → "Michael Willian") according to policy.
4. **(Anonymized Query):** The anonymized query, "Summarize the cardiac exam results for patient <NAME> from <DATE>," is passed to the RAG system.
5. **(RAG + LLM):** The RAG system retrieves the patient's (encrypted) actual record to provide context to the LLM, and the self-hosted LLM within the VPC generates a summary based on this context.
6. **(Output):** The generated summary is returned to the doctor (if needed, the UI can re-identify `<NAME>` as "John Snow").

This "De-ID → RAG → Self-Hosted LLM" stack is the standard architecture for medical AI in 2025. It solves two key problems simultaneously: (1) It protects PHI to comply with HIPAA, and (2) It forces the LLM to base its answers on actual EMR data (RAG), thereby preventing LLM hallucinations.

### 3.2. Finance: GDPR and EU AI Act Compliant Credit Scoring

#### 3.2.1. The Regulation & Problem

- **Law 1: EU AI Act (HRAIS):** "Credit Scoring" is explicitly classified as "High-Risk" under EU AI Act Annex III. Therefore, all 7 key obligations discussed in Module 1.1.3 (Art. 10 Data Governance, Art. 13 Transparency, Art. 14 Human Oversight, etc.) apply.
- **Law 2: EU GDPR (Data Protection):**
  - **Art. 22:** Stipulates the right for individuals not to be subject to a decision "based solely on automated processing" that produces "legal or similarly significant effects" (e.g., an automatic loan denial by an AI). It also guarantees the right to "meaningful information" (i.e., an explanation) and "human intervention."
  - **Art. 9 (GDPR) vs. Art. 10 (AI Act) Conflict:** AI Act Art. 10 "strictly necessarily" allows a high-risk system to process "special categories data" (e.g., race, ethnicity, political orientation) to detect and correct bias. However, GDPR Art. 9 in principle prohibits the processing of this same sensitive data. (To resolve this conflict, one must satisfy the exceptions in GDPR Art. 9 (e.g., explicit consent, substantial public interest) and the conditions in AI Act Art. 10 simultaneously.)
- **The Problem:** Financial firms prefer complex black-box models (e.g., XGBoost, Deep Learning) to increase loan default prediction accuracy by even 1%. But the more complex the model, the more difficult it becomes to provide the "meaningful explanation" required by GDPR Art. 22.

#### 3.2.2. Technical Solution: XAI as a Compliance Tool

To solve this dilemma, XAI (Explainable AI) is used not just as a model debugging tool, but as an essential compliance engine to meet legal requirements.

- **Key Tools: SHAP and LIME**
  - **LIME (Local Interpretable Model-agnostic Explanations):** Creates a virtual sample (perturbation) of data around an individual prediction (e.g., "Why was this customer denied?") and builds a simple "surrogate model" (e.g., linear) that works only in that "local" area to provide an explanation.
  - **SHAP (SHapley Additive exPlanations):** Uses "Shapley values" from cooperative game theory to precisely calculate how much each feature (e.g., income, debt ratio, delinquency history) "contributed" (positively or negatively) to the final prediction (loan approval/denial).
- **Practical Application: Automating "Adverse Action Notice" Generation**
  - GDPR Art. 22 and the US ECOA (Equal Credit Opportunity Act) require that customers who are denied a loan be provided with "specific and accurate" reasons for the denial (in the US, this is the "Adverse Action Notice").
  - XAI is used to automate the generation of this notice:
    1. **(Prediction):** A black-box model (e.g., XGBoost) "denies" Customer A's loan.
    2. **(XAI Execution):** Immediately after this "denial," SHAP is run on Customer A's data. SHAP calculates the features that had the largest negative impact on the decision (e.g., `debt_to_income_ratio: +0.4`, `recent_inquiries: +0.2`, `age_of_oldest_account: +0.1`).
    3. **("Translation" Layer):** A business logic layer translates this mathematical SHAP value (e.g., $+0.4$) into legally compliant human language (Reason Codes).
    4. **(Final Notice):** "Your loan application has been denied. The principal reasons are: (1) High debt-to-income ratio (SHAP $+0.4$), (2) Too many recent credit inquiries (SHAP $+0.2$)."

In financial AI, XAI (SHAP/LIME) is no longer an optional "debugging" tool; it is an essential legal compliance layer for adhering to GDPR Art. 22. However, this has not solved the black-box problem, but rather "shifted the black-box problem." As of November 2025, a new legal risk is emerging. The legal battle is no longer about the model itself, but about the validity of the explanation. "Why did you use SHAP and not LIME?", "Why was the SHAP 'baseline' set to 0 instead of the average value?", "What is the basis for 'translating' a SHAP value of $+0.4$ into the 'principal reason'?" These are the new points of attack. In other words, the engineer now bears a "secondary burden of explanation"—they must be prepared to defend the accuracy and stability of the explanation itself.

### 3.3. Education: Designing a FERPA-Compliant AI Tutor

#### 3.3.1. The Regulation & Problem

- **Law:** USA FERPA (Family Educational Rights and Privacy Act).
- **Key Concept:** Protects PII (Personally Identifiable Information) contained in a student's "Education Records." This includes not only grades and attendance, but also the chat logs between a student and an AI tutor, the AI's analysis of the student's learning patterns, and performance analytics data, all of which can be considered "Education Records."
- **The Problem:** When a school provides student data to a third-party (AI vendor), that vendor must act as a "School Official" with a "legitimate educational interest" under a FERPA exception. However, if the vendor collects these student chat logs and uses them to train their own general-purpose model, this could be a serious FERPA violation, as it falls outside the contracted "educational purpose."

#### 3.3.2. Technical Solution: The "No-Training + RAG" Architecture

This architecture is well-demonstrated in the AI Study Companion case study built by Loyola Marymount University (LMU) in partnership with AWS in November 2025.

- **Core Principle:** "No training on your data."
- **FERPA-Compliant AI Tutor Blueprint:**
  1. **Infrastructure:** All infrastructure is built within the university's own cloud (e.g., AWS) account, ensuring the university retains control over the data.
  2. **Knowledge Base:** The LLM learns from the university-owned course materials (e.g., lecture transcripts, lecture notes, syllabi, textbooks, assignment guides), not the student's PII data. This material is transcribed (e.g., Amazon Transcribe), chunked, and stored in an S3 bucket.
  3. **RAG (Retrieval-Augmented Generation):** A RAG index is built only from these "course materials" (e.g., in Amazon OpenSearch).
  4. **LLM (No-Training):** A managed foundation model (e.g., Claude 3) is called via an API like Amazon Bedrock. The key is ensuring this LLM does not learn from the student's prompts or chat logs (the "no-training" principle). The LLM only generates answers based on the "course material" context provided by RAG (e.g., "Answer based on this course's Week 3 notes, not the internet").
  5. **Privacy:** The student's PII (login info, chat logs) is treated as an "Education Record," is not used for LLM training, and is stored encrypted in a separate, secure DB. The "Principle of Least Privilege" is applied, filtering the RAG search so that a student can only access and ask questions about the course materials for which they are enrolled.

The HIPAA-compliant architecture and the FERPA-compliant architecture are strikingly similar. This is because the core issue for both laws (HIPAA, FERPA) is "control over sensitive data" (PHI, PII). The problem is "AI learning from sensitive data," so the solution is "preventing the AI from learning from sensitive data."

In conclusion, the "Self-Hosted (or Private Cloud) RAG + De-ID/PII Filter" pattern is the standard LLM architecture for regulated industries (healthcare, finance, education) as of 2025. This architecture simultaneously solves three key problems: (1) Legal Compliance (isolating and controlling sensitive data), (2) Privacy (the "No-Training" principle), and (3) Reliability (preventing hallucination via RAG).

### Checkpoint Questions

- What are the key components of a HIPAA-compliant LLM chatbot architecture, and why is the De-ID gateway critical?
- How does the conflict between GDPR Art. 9 and EU AI Act Art. 10 create challenges for credit scoring systems?
- Explain how XAI (SHAP/LIME) serves as a compliance tool for GDPR Art. 22, and what new legal risks emerge from using XAI?
- What is the "no-training" principle in FERPA-compliant AI tutors, and how does RAG enable this?
- Why do HIPAA, FERPA, and GDPR-compliant architectures share similar patterns despite different regulations?

---

## Module 4: Workshop Guide (Core Practice/Assignment)

You have now learned Module 1 (Law), Module 2 (Technology), and Module 3 (Case Studies). Based on this, let's draft a "Design for an LLM Service Compliant with EU AI Act and other relevant regulations."

### 4.1. Assignment Scenario

- **You are:** The development team lead at an AI startup planning to enter the EU market.
- **Product:** Choose one of the "high-risk" scenarios below.
  1. **Finance:** A "Credit Scoring" solution for small and medium-sized enterprises (SMEs) across the EU, based on your own in-house GPAI model trained with $10^{24}$ FLOPs.
  2. **Education:** An "AI Writing Tutor" solution that evaluates and gives feedback on EU university students' writing, based on a commercial GPAI model API (e.g., GPT-4o, Claude 3.5).
  3. **Employment:** An "AI HR" solution that analyzes and ranks thousands of CVs for a large EU corporation's job postings, based on a fine-tuned commercial GPAI model (e.g., Llama 3).
- **Assignment:** For your chosen scenario, create an "EU AI Act Compliance Checklist" covering the steps from model development to deployment, and present why you took those technical/policy measures.

### 4.2. A Practical Compliance Checklist for the EU AI Act

Your design brief should, at a minimum, include answers to the following items.

#### Phase 1: System Classification

- [ ] **Risk Tier Identification:** What tier does our system fall under in the AI Act?
  - (Example Answer: Scenario 1 (Credit Scoring), 2 (Evaluating Learning Outcomes), and 3 (Recruitment) are all explicitly "High-Risk" per Annex III.)
- [ ] **GPAI Model Identification:** Is the model our system is based on a GPAI?
  - (Example Answer: Scenario 1 is a "GPAI" as $10^{24}$ FLOPs > $10^{23}$ FLOPs. Scenarios 2 & 3 use commercial models, so their providers are "GPAI Providers".)
- [ ] **Systemic Risk Identification:** Does the base model have "systemic risk"?
  - (Example Answer: Scenario 1 is $10^{24}$ FLOPs < $10^{25}$ FLOPs, so it is not presumed to have systemic risk. For Scenarios 2 & 3, if the base model (GPT-4o, etc.) exceeds $10^{25}$ FLOPs, it is a "systemic risk" model.)

#### Phase 2: GPAI Provider Obligations (If applicable) (Art. 53)

(If you developed the GPAI yourself like in Scenario 1, or if you "substantially modified" a base model like in Scenario 3, you may be a "GPAI Provider".)

- [ ] **Technical Documentation (Art. 53):** Have you documented the model's training/evaluation process?
- [ ] **Copyright Policy (Art. 53):** Have you established a policy to comply with EU copyright law (e.g., respecting opt-outs)?
- [ ] **Data Summary (Art. 53):** Are you ready to publish a training data summary using the AI Office template?
- [ ] **Code of Practice:** Will you sign the July 2025 GPAI "Code of Practice" to demonstrate compliance? (Advantageous for securing downstream partners)

#### Phase 3: Systemic Risk Obligations (If applicable) (Art. 55)

(If your system is based on a >$10^{25}$ FLOPs model, like Scenario 2. Note: This obligation is primarily on the "upstream" provider (OpenAI), but as a "downstream" user, you must verify its fulfillment.)

- [ ] **Model Evaluation (Art. 55):** Have you confirmed that the base model provider performed adversarial testing, etc.?
- [ ] **Risk Mitigation (Art. 55):** Are there measures to mitigate identified systemic risks (e.g., bias)?
- [ ] **Incident Reporting (Art. 55):** Is there a system for reporting serious incidents?
- [ ] **Cybersecurity (Art. 55):** Are cybersecurity measures in place for the model and infrastructure?

#### Phase 4: HRAIS Provider Obligations (Mandatory!) (Art. 8-15)

(Mandatory for Scenarios 1, 2, and 3. You are the provider of the "High-Risk AI System".)

- [ ] **Risk Management (Art. 9):** Have you established a risk management system for the AI lifecycle? (e.g., regular risk assessment and mitigation plans)
- [ ] **Data Governance (Art. 10):**
  - [ ] (Scenarios 1, 3) How do you prove your training data is not biased against a specific gender, race, or nationality? (e.g., dataset representativeness analysis)
  - [ ] (Scenarios 2, 3) Do you have lawful consent under GDPR (Art. 9) to process the PII and sensitive data of students/applicants?
  - [ ] (Scenarios 1, 3) If you must use sensitive data (e.g., race) for bias correction, how did you resolve the conflict between the AI Act (Art 10.5) and GDPR (Art. 9)? (e.g., explicit consent + strict purpose limitation)
- [ ] **Technical Documentation (Art. 11):** Have you prepared all technical documentation for the HRAIS? (Architecture, GPAI model used, dataset info, evaluation results)
  - (Warning: As of Nov 2025, your upstream GPAI provider may not give you this info (see 1.2.5). How will you solve this "supply chain crisis"?)
- [ ] **Logging (Art. 12):** Do you log all system decisions (e.g., loan denial, CV rejection) and their basis for traceability?
- [ ] **Transparency (Art. 13):**
  - [ ] (Scenarios 2, 3) Do you provide clear instructions for use (e.g., "This is for reference only; you make the final decision") and limitations (e.g., "This model is weak on certain writing types") to the system operators (teachers, HR team)?
  - [ ] (Scenario 1) Can you provide the "reason for denial" for a credit assessment using XAI? (Links to GDPR Art. 22)
- [ ] **Human Oversight (Art. 14):**
  - [ ] Is there a "Human Oversight" mechanism to stop, disregard, or reverse an automated decision (e.g., auto-rejection of a CV, auto-denial of a loan)?
  - [ ] (Scenario 3) "The AI recommends the top 10%, but a human makes the final decision." — Is this "meaningful" human oversight as required by the AI Act, or is it "rubber-stamping" where the human just blindly follows the AI? (You must defend this design choice.)
- [ ] **Robustness/Security (Art. 15):** Is the system robust against adversarial attacks (e.g., prompt injection, adding keywords in white text to a CV), and is its cybersecurity ensured?

### 4.3. Decision Framework for Integrating PETs

Your design brief should include a technical rationale for why you chose (or did not choose) specific PETs.

- **Option 1: Private Architecture (Default)**
  - **Design:** "Self-Hosted (or VPC) RAG + De-ID/PII Filter" (Architecture from Modules 3.1 & 3.3).
  - **Reason for Choice:** This is the simplest and most robust method for regulatory compliance in 2025. It avoids training on sensitive data (PII, PHI) by storing it in an isolated DB and using the LLM only as a "stateless" inference engine. This is the most reliable way to satisfy both AI Act Art. 10 (Data Governance) and GDPR. (Strongly recommended for Scenarios 2 & 3)
- **Option 2: Differential Privacy (DP)**
  - **Design:** (In addition to Option 1) Use when you need to retrain and improve your "general-purpose writing model" using the sensitive data collected during service (e.g., student essays).
  - **Reason for Choice:** Use when retraining on sensitive data is necessary to improve service quality. Either collect only DP-applied "trends" from local devices like Apple, or create "DP synthetic data" from the collected data to retrain the model, like Google.
- **Option 3: Federated Learning (FL)**
  - **Design:** (In addition to Option 1) Use when multiple institutions (e.g., multiple universities, multiple banks) want to build a "common" domain model (e.g., a joint credit scoring model) without sharing data.
  - **Reason for Choice:** Use when central data collection is legally/commercially impossible. Use the "PEFT(LoRA) + FL" architecture, where each institution trains only its local LoRA, and the server aggregates only those LoRAs. (May be suitable for Scenario 1)
- **Option 4: Homomorphic Encryption (HE)**
  - **Design:** Use when the user wants to perform inference without the server (credit scoring model) ever seeing their sensitive query/document (e.g., personal financial statements).
  - **Reason for Choice:** Use when the highest level of "query privacy" is needed. Use the "Hybrid HE" method, where the server handles linear operations (encrypted) and the client handles non-linear operations (decrypted), to achieve practical inference speeds. (Suitable for a B2C version of Scenario 1)

The most powerful and practical solution in your assignment may not be the "flashiest" PET from Module 2. As of 2025, most regulatory problems do not require complex cryptography (HE) or distributed learning (FL). The problems mostly stem from a failure of data governance and architectural separation. The "Self-Hosted RAG + De-ID Filter" architecture seen in Module 3 is the "default" and "best-practice" blueprint that solves 90% of regulatory issues. **The best privacy protection is to not collect the data in the first place (RAG), or to anonymize it (De-ID).** DP, FL, and HE are "Phase 2" solutions to be considered only when a special business need arises that this basic architecture cannot solve (e.g., "We must train on distributed data").

---

## References

1. High-level summary of the AI Act | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/high-level-summary/](https://artificialintelligenceact.eu/high-level-summary/)
2. EU AI Act timeline and risk tiers explained - Trilateral Research, [https://trilateralresearch.com/responsible-ai/eu-ai-act-implementation-timeline-mapping-your-models-to-the-new-risk-tiers](https://trilateralresearch.com/responsible-ai/eu-ai-act-implementation-timeline-mapping-your-models-to-the-new-risk-tiers)
3. Key Issue 3: Risk-Based Approach - EU AI Act, [https://www.euaiact.com/key-issue/3](https://www.euaiact.com/key-issue/3)
4. AI Act | Shaping Europe's digital future - European Union, [https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
5. EU AI Act Prohibited Use Cases | Harvard University Information Technology, [https://www.huit.harvard.edu/eu-ai-act](https://www.huit.harvard.edu/eu-ai-act)
6. New EU AI Act guidelines: what are the implications for businesses?, [https://www.twobirds.com/en/insights/2025/global/new-eu-ai-act-guidelines-what-are-the-implications-for-businesses](https://www.twobirds.com/en/insights/2025/global/new-eu-ai-act-guidelines-what-are-the-implications-for-businesses)
7. The EU AI Act: Where Do We Stand in 2025? | Blog - BSR, [https://www.bsr.org/en/blog/the-eu-ai-act-where-do-we-stand-in-2025](https://www.bsr.org/en/blog/the-eu-ai-act-where-do-we-stand-in-2025)
8. The AI Act Explorer | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/ai-act-explorer/](https://artificialintelligenceact.eu/ai-act-explorer/)
9. The EU AI Act: A Quick Guide, [https://www.simmons-simmons.com/en/publications/clyimpowh000ouxgkw1oidakk/the-eu-ai-act-a-quick-guide](https://www.simmons-simmons.com/en/publications/clyimpowh000ouxgkw1oidakk/the-eu-ai-act-a-quick-guide)
10. What you need to know about the EU AI Act and how Concentric AI can help, [https://concentric.ai/what-you-need-to-know-about-the-eu-ai-act-and-how-concentric-ai-can-help/](https://concentric.ai/what-you-need-to-know-about-the-eu-ai-act-and-how-concentric-ai-can-help/)
11. EU AI Act: different risk levels of AI systems - Forvis Mazars - Ireland, [https://www.forvismazars.com/ie/en/insights/news-opinions/eu-ai-act-different-risk-levels-of-ai-systems](https://www.forvismazars.com/ie/en/insights/news-opinions/eu-ai-act-different-risk-levels-of-ai-systems)
12. EU Artificial Intelligence Act | Up-to-date developments and analyses of the EU AI Act, [https://artificialintelligenceact.eu/](https://artificialintelligenceact.eu/)
13. Overview of Guidelines for GPAI Models | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/gpai-guidelines-overview/](https://artificialintelligenceact.eu/gpai-guidelines-overview/)
14. Guidelines for providers of general-purpose AI models | Shaping Europe's digital future, [https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers)
15. European Commission publishes guidelines on obligations for general-purpose AI models under the EU AI Act | DLA Piper, [https://www.dlapiper.com/insights/publications/ai-outlook/2025/european-commission-publishes-guidelines-for-general-purpose-ai-models-under-the-eu-ai-act](https://www.dlapiper.com/insights/publications/ai-outlook/2025/european-commission-publishes-guidelines-for-general-purpose-ai-models-under-the-eu-ai-act)
16. European Commission Issues Guidelines for Providers of General-Purpose AI Models, [https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20250724-european-commission-issues-guidelines-for-providers-of-general-purpose-ai-models](https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20250724-european-commission-issues-guidelines-for-providers-of-general-purpose-ai-models)
17. EU AI Act: first regulation on artificial intelligence | Topics - European Parliament, [https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence](https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence)
18. Generally Speaking: Does Your Company Have EU AI Act Compliance Obligations as a General-Purpose AI Model Provider? - Arnold & Porter, [https://www.arnoldporter.com/en/perspectives/advisories/2025/08/does-your-company-have-eu-ai-act-compliance-obligations](https://www.arnoldporter.com/en/perspectives/advisories/2025/08/does-your-company-have-eu-ai-act-compliance-obligations)
19. EU AI Act Compliance Checker | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/](https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/)
20. EU's General-Purpose AI Obligations Are Now in Force, With New Guidance - Skadden, [https://www.skadden.com/insights/publications/2025/08/eus-general-purpose-ai-obligations](https://www.skadden.com/insights/publications/2025/08/eus-general-purpose-ai-obligations)
21. An Introduction to the Code of Practice for General-Purpose AI | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/introduction-to-code-of-practice/](https://artificialintelligenceact.eu/introduction-to-code-of-practice/)
22. The EU Commission Publishes General-Purpose AI Code of Practice: Compliance Obligations Begin August 2025 - Nelson Mullins, [https://www.nelsonmullins.com/insights/blogs/ai-task-force/ai/ai-task-force-the-eu-commission-publishes-general-purpose-ai-code-of-practice-compliance-obligations-begin-august-2025](https://www.nelsonmullins.com/insights/blogs/ai-task-force/ai/ai-task-force-the-eu-commission-publishes-general-purpose-ai-code-of-practice-compliance-obligations-begin-august-2025)
23. Overview of the Code of Practice | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/code-of-practice-overview/](https://artificialintelligenceact.eu/code-of-practice-overview/)
24. General-purpose AI Obligations Under the EU AI Act Kick in From 2 August 2025 | Insight, [https://www.bakermckenzie.com/en/insight/publications/2025/08/general-purpose-ai-obligations](https://www.bakermckenzie.com/en/insight/publications/2025/08/general-purpose-ai-obligations)
25. The General-Purpose AI Code of Practice | Shaping Europe's digital future, [https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)
26. EU AI Act: General-Purpose AI Code of Practice · Final Version, [https://code-of-practice.ai/](https://code-of-practice.ai/)
27. Modifying AI Under the EU AI Act: Lessons from Practice on ..., [https://artificialintelligenceact.eu/modifying-ai-under-the-eu-ai-act/](https://artificialintelligenceact.eu/modifying-ai-under-the-eu-ai-act/)
28. Full article: Regulating AI from Europe: a joint analysis of the AI Act and the Framework Convention on AI - Taylor & Francis Online, [https://www.tandfonline.com/doi/full/10.1080/20508840.2025.2492524](https://www.tandfonline.com/doi/full/10.1080/20508840.2025.2492524)
29. Trust in the EU, U.S. and China to regulate use of AI - Pew Research Center, [https://www.pewresearch.org/2025/10/15/trust-in-the-eu-u-s-and-china-to-regulate-use-of-ai/](https://www.pewresearch.org/2025/10/15/trust-in-the-eu-u-s-and-china-to-regulate-use-of-ai/)
30. AI Watch: Global regulatory tracker - United States | White & Case LLP, [https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-united-states](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-united-states)
31. America's AI Action Plan - The White House, [https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf](https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf)
32. February 2025 AI Developments Under the Trump Administration, [https://www.insidegovernmentcontracts.com/2025/03/february-2025-ai-developments-under-the-trump-administration/](https://www.insidegovernmentcontracts.com/2025/03/february-2025-ai-developments-under-the-trump-administration/)
33. AI Risk Management Framework | NIST - National Institute of Standards and Technology, [https://www.nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
34. NIST AI Risk Management Framework: A simple guide to smarter AI governance - Diligent, [https://www.diligent.com/resources/blog/nist-ai-risk-management-framework](https://www.diligent.com/resources/blog/nist-ai-risk-management-framework)
35. European Industry Pushes Back on the EU AI Act – Key Takeaways for Employers, [https://www.fisherphillips.com/en/news-insights/european-industry-pushes-back-on-the-eu-ai-act.html](https://www.fisherphillips.com/en/news-insights/european-industry-pushes-back-on-the-eu-ai-act.html)
36. EU could water down AI Act amid pressure from Trump and big tech ..., [https://www.theguardian.com/world/2025/nov/07/european-commission-ai-artificial-intelligence-act-trump-administration-tech-business](https://www.theguardian.com/world/2025/nov/07/european-commission-ai-artificial-intelligence-act-trump-administration-tech-business)
37. South Korea's New AI law: What it Means for Organizations and How ..., [https://www.onetrust.com/blog/south-koreas-new-ai-law-what-it-means-for-organizations-and-how-to-prepare/](https://www.onetrust.com/blog/south-koreas-new-ai-law-what-it-means-for-organizations-and-how-to-prepare/)
38. South Korea Artificial Intelligence (AI) Basic Act - International Trade Administration, [https://www.trade.gov/market-intelligence/south-korea-artificial-intelligence-ai-basic-act](https://www.trade.gov/market-intelligence/south-korea-artificial-intelligence-ai-basic-act)
39. South Korea's New AI Framework Act: A Balancing Act Between Innovation and Regulation, [https://fpf.org/blog/south-koreas-new-ai-framework-act-a-balancing-act-between-innovation-and-regulation/](https://fpf.org/blog/south-koreas-new-ai-framework-act-a-balancing-act-between-innovation-and-regulation/)
40. Global Approaches to Artificial Intelligence Regulation, [https://jsis.washington.edu/news/global-approaches-to-artificial-intelligence-regulation/](https://jsis.washington.edu/news/global-approaches-to-artificial-intelligence-regulation/)
41. AI Watch: Global regulatory tracker - China | White & Case LLP, [https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-china](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-china)
42. China - AI Regulatory Horizon Tracker - Bird & Bird, [https://www.twobirds.com/en/capabilities/artificial-intelligence/ai-legal-services/ai-regulatory-horizon-tracker/china](https://www.twobirds.com/en/capabilities/artificial-intelligence/ai-legal-services/ai-regulatory-horizon-tracker/china)
43. accessed November 11, 2025, [https://www.anecdotes.ai/learn/ai-regulations-in-2025-us-eu-uk-japan-china-and-more\#:\~:text=China%3A%20Generative%20AI%20Regulation,-In%20brief%3A%20What\&text=In%20addition%2C%20providers%20are%20required,legal%20sourcing%20of%20training%20data.](https://www.anecdotes.ai/learn/ai-regulations-in-2025-us-eu-uk-japan-china-and-more#:~:text=China%3A%20Generative%20AI%20Regulation,-In%20brief%3A%20What&text=In%20addition%2C%20providers%20are%20required,legal%20sourcing%20of%20training%20data.)
44. AI Regulation Updates H2 2025 - FairNow, [https://fairnow.ai/ai-regulations-updates-h2-2025/](https://fairnow.ai/ai-regulations-updates-h2-2025/)
45. China Announces Action Plan for Global AI Governance, [https://www.ansi.org/standards-news/all-news/8-1-25-china-announces-action-plan-for-global-ai-governance](https://www.ansi.org/standards-news/all-news/8-1-25-china-announces-action-plan-for-global-ai-governance)
46. [2506.11687] Differential Privacy in Machine Learning: From Symbolic AI to LLMs - arXiv, [https://arxiv.org/abs/2506.11687](https://arxiv.org/abs/2506.11687)
47. Differential Privacy in Machine Learning: From Symbolic AI to LLMs - arXiv, [https://arxiv.org/html/2506.11687v1](https://arxiv.org/html/2506.11687v1)
48. [2503.12896] Safeguarding LLM Embeddings in End-Cloud Collaboration via Entropy-Driven Perturbation - arXiv, [https://arxiv.org/abs/2503.12896](https://arxiv.org/abs/2503.12896)
49. Generating synthetic data with differentially private LLM inference, [https://research.google/blog/generating-synthetic-data-with-differentially-private-llm-inference/](https://research.google/blog/generating-synthetic-data-with-differentially-private-llm-inference/)
50. The Crossroads of Innovation and Privacy: Private Synthetic Data for Generative AI, [https://www.microsoft.com/en-us/research/blog/the-crossroads-of-innovation-and-privacy-private-synthetic-data-for-generative-ai/](https://www.microsoft.com/en-us/research/blog/the-crossroads-of-innovation-and-privacy-private-synthetic-data-for-generative-ai/)
51. Apple Intelligence Foundation Language Models Tech Report 2025, [https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
52. Understanding Aggregate Trends for Apple Intelligence Using ..., [https://machinelearning.apple.com/research/differential-privacy-aggregate-trends](https://machinelearning.apple.com/research/differential-privacy-aggregate-trends)
53. Federated Learning With Differential Privacy for End-to-End Speech Recognition, [https://machinelearning.apple.com/research/fed-learning-diff-privacy](https://machinelearning.apple.com/research/fed-learning-diff-privacy)
54. TechDispatch #1/2025 - Federated Learning - European Data Protection Supervisor, [https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-06-10-techdispatch-12025-federated-learning_en](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-06-10-techdispatch-12025-federated-learning_en)
55. Revolutionizing healthcare data analytics with federated learning: A comprehensive survey of applications, systems, and future directions - PMC - PubMed Central, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/)
56. Federated Learning with Layer Skipping: Efficient Training of ... - arXiv, [https://arxiv.org/abs/2504.10536](https://arxiv.org/abs/2504.10536)
57. [2501.04436] Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions - arXiv, [https://arxiv.org/abs/2501.04436](https://arxiv.org/abs/2501.04436)
58. (PDF) Federated Large Language Model: Solutions, Challenges and Future Directions, [https://www.researchgate.net/publication/385183939_Federated_Large_Language_Model_Solutions_Challenges_and_Future_Directions](https://www.researchgate.net/publication/385183939_Federated_Large_Language_Model_Solutions_Challenges_and_Future_Directions)
59. FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning - arXiv, [https://arxiv.org/html/2510.04601v2](https://arxiv.org/html/2510.04601v2)
60. Implementing Federated Learning: A Privacy-Preserving AI Approach, [https://blog.4geeks.io/implementing-federated-learning-a-privacy-preserving-ai-approach/](https://blog.4geeks.io/implementing-federated-learning-a-privacy-preserving-ai-approach/)
61. encryption-friendly llm architecture - arXiv, [https://arxiv.org/pdf/2410.02486](https://arxiv.org/pdf/2410.02486)
62. HHEML: Hybrid Homomorphic Encryption for Privacy-Preserving Machine Learning on Edge, [https://arxiv.org/html/2510.20243v1](https://arxiv.org/html/2510.20243v1)
63. Agentic Privacy-Preserving Machine LearningA position paper. Under active development., [https://arxiv.org/html/2508.02836](https://arxiv.org/html/2508.02836)
64. Practical and Private Hybrid ML Inference with Fully ... - arXiv, [https://arxiv.org/abs/2509.01253](https://arxiv.org/abs/2509.01253)
65. Efficient Keyset Design for Neural Networks Using Homomorphic Encryption - MDPI, [https://www.mdpi.com/1424-8220/25/14/4320](https://www.mdpi.com/1424-8220/25/14/4320)
66. Development of Privacy-preserving Deep Learning Model with Homomorphic Encryption: A Technical Feasibility Study in Kidney CT Imaging | Radiology: Artificial Intelligence - RSNA Journals, [https://pubs.rsna.org/doi/10.1148/ryai.240798](https://pubs.rsna.org/doi/10.1148/ryai.240798)
67. ENCRYPTION-FRIENDLY LLM ARCHITECTURE - ICLR Proceedings, [https://proceedings.iclr.cc/paper_files/paper/2025/file/6715b4e97be055687c1ecaf33913d358-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/6715b4e97be055687c1ecaf33913d358-Paper-Conference.pdf)
68. [2410.02486] Encryption-Friendly LLM Architecture - arXiv, [https://arxiv.org/abs/2410.02486](https://arxiv.org/abs/2410.02486)
69. How to Encrypt Client Data Before Sending to an API-Based LLM? : r/LlamaIndex - Reddit, [https://www.reddit.com/r/LlamaIndex/comments/1iwzeph/how_to_encrypt_client_data_before_sending_to_an/](https://www.reddit.com/r/LlamaIndex/comments/1iwzeph/how_to_encrypt_client_data_before_sending_to_an/)
70. Natural Language Processing for Enterprise-scale De-identification of Protected Health Information in Clinical Notes - NIH, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9285160/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9285160/)
71. Healthcare Chatbot Development Guide for 2025 - MobiDev, [https://mobidev.biz/blog/healthcare-chatbot-development-guide](https://mobidev.biz/blog/healthcare-chatbot-development-guide)
72. HIPAA-Ready AI Chatbots: Secure Hosting for Healthcare Innovation, [https://www.atlantic.net/hipaa-compliant-hosting/hipaa-ready-ai-chatbots-secure-hosting-for-healthcare-innovation/](https://www.atlantic.net/hipaa-compliant-hosting/hipaa-ready-ai-chatbots-secure-hosting-for-healthcare-innovation/)
73. Is ChatGPT HIPAA Compliant? Updated for 2025 - The HIPAA Journal, [https://www.hipaajournal.com/is-chatgpt-hipaa-compliant/](https://www.hipaajournal.com/is-chatgpt-hipaa-compliant/)
74. ChatGPT for Healthcare | Medical GPT with HIPAA Compliance, [https://bastiongpt.com/](https://bastiongpt.com/)
75. HIPAA Compliance for Healthcare Chatbots: Essential Guide - Kommunicate, [https://www.kommunicate.io/blog/a-essential-guide-to-hipaa-compliance-in-healthcare-chatbots/](https://www.kommunicate.io/blog/a-essential-guide-to-hipaa-compliance-in-healthcare-chatbots/)
76. Effortless PHI De-Identification: De-identified Patient Data with ..., [https://www.johnsnowlabs.com/effortless-de-identification-running-obfuscation-and-deidentification-in-healthcare-nlp/](https://www.johnsnowlabs.com/effortless-de-identification-running-obfuscation-and-deidentification-in-healthcare-nlp/)
77. Large Language Models for Electronic Health Record De-Identification in English and German - MDPI, [https://www.mdpi.com/2078-2489/16/2/112](https://www.mdpi.com/2078-2489/16/2/112)
78. Software to Identify PHI: Complete 2025 Guide & Tools - Invene, [https://www.invene.com/blog/software-to-identify-phi-complete-guide](https://www.invene.com/blog/software-to-identify-phi-complete-guide)
79. Can Zero-Shot Commercial API's Deliver Regulatory-Grade Clinical Text De-Identification?, [https://arxiv.org/html/2503.20794v2](https://arxiv.org/html/2503.20794v2)
80. Case Studies of AI Applications Within HIPAA Guidelines - Accountable HQ, [https://www.accountablehq.com/post/case-studies-of-ai-applications-within-hipaa-guidelines](https://www.accountablehq.com/post/case-studies-of-ai-applications-within-hipaa-guidelines)
81. AI Fraud Detection Compliance in Financial Services: Balancing Security with Customer Rights - VerityAI, [https://verityai.co/blog/ai-fraud-detection-compliance-financial-services](https://verityai.co/blog/ai-fraud-detection-compliance-financial-services)
82. Privacy and responsible AI - IAPP, [https://iapp.org/news/a/privacy-and-responsible-ai](https://iapp.org/news/a/privacy-and-responsible-ai)
83. Law & Compliance in AI Security & Data Protection, [https://www.edpb.europa.eu/system/files/2025-06/spe-training-on-ai-and-data-protection-legal_en.pdf](https://www.edpb.europa.eu/system/files/2025-06/spe-training-on-ai-and-data-protection-legal_en.pdf)
84. GDPR Compliance for AI Developers - A Practical Guide - Essert Inc, [https://essert.io/gdpr-compliance-for-ai-developers-a-practical-guide/](https://essert.io/gdpr-compliance-for-ai-developers-a-practical-guide/)
85. The EU AI Act and the GDPR: collision or alignment? - Taylor Wessing, [https://www.taylorwessing.com/en/global-data-hub/2025/eu-digital-laws-and-gdpr/gdh---the-eu-ai-act-and-the-gdpr](https://www.taylorwessing.com/en/global-data-hub/2025/eu-digital-laws-and-gdpr/gdh---the-eu-ai-act-and-the-gdpr)
86. (PDF) Explainable AI in Credit Scoring: Balancing Accuracy and ..., [https://www.researchgate.net/publication/394998451_Explainable_AI_in_Credit_Scoring_Balancing_Accuracy_and_Transparency](https://www.researchgate.net/publication/394998451_Explainable_AI_in_Credit_Scoring_Balancing_Accuracy_and_Transparency)
87. AI-Driven Fraud Detection Under GDPR and Financial Regulations - ResearchGate, [https://www.researchgate.net/publication/393870899_AI-Driven_Fraud_Detection_Under_GDPR_and_Financial_Regulations](https://www.researchgate.net/publication/393870899_AI-Driven_Fraud_Detection_Under_GDPR_and_Financial_Regulations)
88. Explainable AI (XAI) for Credit Scoring and Loan Approvals - ResearchGate, [https://www.researchgate.net/publication/389847187_Explainable_AI_XAI_for_Credit_Scoring_and_Loan_Approvals](https://www.researchgate.net/publication/389847187_Explainable_AI_XAI_for_Credit_Scoring_and_Loan_Approvals)
89. Advance Journal of Econometrics and Finance Vol-3, Issue-1, 2025, [https://ajeaf.com/index.php/Journal/article/download/131/142](https://ajeaf.com/index.php/Journal/article/download/131/142)
90. TechDispatch #2/2023 - Explainable Artificial Intelligence, [https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2023-11-16-techdispatch-22023-explainable-artificial-intelligence_en](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2023-11-16-techdispatch-22023-explainable-artificial-intelligence_en)
91. Explainable AI in Finance: Addressing the Needs of Diverse Stakeholders, [https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance](https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance)
92. Explainable AI for Credit Assessment in Banks - MDPI, [https://www.mdpi.com/1911-8074/15/12/556](https://www.mdpi.com/1911-8074/15/12/556)
93. A novel framework for enhancing transparency in credit scoring: Leveraging Shapley values for interpretable credit scorecards - NIH, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/)
94. (PDF) Enhancing Model Interpretability and Regulatory Compliance in Credit Risk Assessment through Explainable Artificial Intelligence (XAI) Techniques (e.g., SHAP, LIME) applied to complex Black-Box models - ResearchGate, [https://www.researchgate.net/publication/397323127_Enhancing_Model_Interpretability_and_Regulatory_Compliance_in_Credit_Risk_Assessment_through_Explainable_Artificial_Intelligence_XAI_Techniques_eg_SHAP_LIME_applied_to_complex_Black-Box_models](https://www.researchgate.net/publication/397323127_Enhancing_Model_Interpretability_and_Regulatory_Compliance_in_Credit_Risk_Assessment_through_Explainable_Artificial_Intelligence_XAI_Techniques_eg_SHAP_LIME_applied_to_complex_Black-Box_models)
95. Explaining Deep Learning Models for Credit Scoring with SHAP: A Case Study Using Open Banking Data - MDPI, [https://www.mdpi.com/1911-8074/16/4/221](https://www.mdpi.com/1911-8074/16/4/221)
96. Using Explainable AI to Produce ECOA Adverse Action Reasons ..., [https://www.paceanalyticsllc.com/post/ecoa-adverse-actions-and-explainable-ai](https://www.paceanalyticsllc.com/post/ecoa-adverse-actions-and-explainable-ai)
97. The Accuracy-Interpretability Dilemma: A Strategic Framework for Navigating the Trade-off in Modern Machine Learning - Science Publishing Group, [https://www.sciencepublishinggroup.com/article/10.11648/j.ajist.20250903.15](https://www.sciencepublishinggroup.com/article/10.11648/j.ajist.20250903.15)
98. 2025 AI Guide to Ferpa Compliance | Concentric AI, [https://concentric.ai/maintain-ferpa-compliance-with-concentric-ai/](https://concentric.ai/maintain-ferpa-compliance-with-concentric-ai/)
99. Federal Regulations Related to Artificial Intelligence The United States does not have a comprehensive law that covers data priv, [https://www.nea.org/sites/default/files/2025-06/5.1-ai-policy-overview-of-federal-regulations-final.pdf](https://www.nea.org/sites/default/files/2025-06/5.1-ai-policy-overview-of-federal-regulations-final.pdf)
100. Case Studies in AI: - Helping staff and students use AI in fair and privacy-protective ways - MOREnet, [https://www.more.net/wp-content/uploads/2025/02/Case-Studies-in-AI.pdf](https://www.more.net/wp-content/uploads/2025/02/Case-Studies-in-AI.pdf)
101. How to build a FERPA-compliant AI study companion - Ki Ecke, [https://ki-ecke.com/insights/how-to-build-a-ferpa-compliant-ai-study-companion/](https://ki-ecke.com/insights/how-to-build-a-ferpa-compliant-ai-study-companion/)
