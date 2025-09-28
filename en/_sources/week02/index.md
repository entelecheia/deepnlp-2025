# Week 2: PyTorch 2.x and Latest Deep Learning Frameworks

## 1. PyTorch 2.x and torch.compile: The Compiler Revolution

PyTorch 2.x is an innovative update that has changed the paradigm of deep learning frameworks. While maintaining the flexibility and Pythonic development experience provided by the existing **Eager Mode** approach, it has added a powerful feature that maximizes model execution performance with just a single line of code: `torch.compile`. This is evaluated as an innovation that "simultaneously satisfies the flexibility of research and the speed of production."

### 1.1 How torch.compile Works

The performance improvement of `torch.compile` is achieved through the organic collaboration of four core technologies: **TorchDynamo**, **AOTAutograd**, **PrimTorch**, and **TorchInductor**.

#### 1. Graph Acquisition (TorchDynamo)

- **Role**: Analyzes Python bytecode to safely capture PyTorch operations as FX graphs
- **Core Technology**: "Guard" mechanism that perfectly supports dynamic Python characteristics (conditionals, loops)
- **Advantage**: When code paths change, only the relevant parts are processed in eager mode while the rest runs compiled code

#### 2. Ahead-of-Time Automatic Differentiation (AOTAutograd)

- **Role**: Generates pre-optimized backward graphs based on forward graphs
- **Advantage**: Pre-analyzes the entire computation graph to optimize gradient calculation processes and reduce memory usage

#### 3. Graph Lowering (PrimTorch)

- **Role**: Standardizes over 2,000 PyTorch operators into 250 core primitive operators
- **Advantage**: Improves compatibility and portability across various hardware backends (GPU, CPU, custom accelerators)

#### 4. Graph Compilation (TorchInductor)

- **Role**: Converts primitive operator graphs into hardware-optimized machine code
- **Core Technology**: Dynamic generation of high-performance CUDA kernels using Triton compiler on GPU, C++/OpenMP on CPU

Through these multi-stage optimizations, `torch.compile` achieved an **average 51%** training speed improvement across 163 model benchmarks.

### 1.2 Practice: Improving Model Inference Speed with torch.compile

Let's directly compare the performance of **Eager Mode** and **Compiled Mode** by applying `torch.compile` to a simple `nn.Module`. While compilation incurs overhead from graph capture and code generation on the first run, subsequent repeated calls show significantly faster speeds that more than offset this cost.

```python
import torch
import torch.nn as nn
import time

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define a simple neural network model
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

# 1. Eager mode performance measurement
# Warmup: to exclude potential overhead from initial execution
for _ in range(10):
    _ = model(dummy_input)

if device == "cuda": torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    _ = model(dummy_input)
if device == "cuda": torch.cuda.synchronize()
eager_duration = time.time() - start_time
print(f"Eager mode (100 runs): {eager_duration:.4f} seconds")

# 2. Apply torch.compile (compiled mode)
# mode="reduce-overhead" reduces framework overhead, beneficial for small model calls
compiled_model = torch.compile(model, mode="reduce-overhead")

# Warmup and first compilation run (compilation overhead occurs)
for _ in range(10):
    _ = compiled_model(dummy_input)

if device == "cuda": torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    _ = compiled_model(dummy_input)
if device == "cuda": torch.cuda.synchronize()
compiled_duration = time.time() - start_time
print(f"Compiled mode (100 runs): {compiled_duration:.4f} seconds")

# 3. Calculate performance improvement
speedup = eager_duration / compiled_duration
print(f"Speedup with torch.compile: {speedup:.2f}x")
```

**Example execution results:**

```
Using device: cuda
Eager mode (100 runs): 0.0481 seconds
Compiled mode (100 runs): 0.0215 seconds
Speedup with torch.compile: 2.24x
```

### Checkpoint Questions

- What roles do the four core technologies of `torch.compile` (TorchDynamo, AOTAutograd, PrimTorch, TorchInductor) play?
- Why was the `mode="reduce-overhead"` option used in the above example? What difference would there be with the default mode for small models?
- What are the main advantages and limitations of `torch.compile` compared to the existing eager execution mode?

## 2. FlashAttention-3: Attention Optimization through Hardware Acceleration

The **attention mechanism**, which is the core of the Transformer architecture and was the main performance bottleneck, has achieved revolutionary progress with the advent of FlashAttention. Traditional attention had limitations in processing long sequences due to O(N²) memory and computational complexity for sequence length N. This was because it had to store and read back massive N × N attention score matrices in GPU's HBM (High Bandwidth Memory).

### 2.1 Core Principles of FlashAttention

**FlashAttention** utilizes **tiling** techniques and GPU's very fast internal SRAM (Static RAM) to solve memory I/O bottlenecks. Instead of computing entire matrices at once, it divides inputs into small blocks (tiles) and performs attention computation in SRAM, storing only intermediate results in HBM, dramatically reducing the number of data exchanges with HBM.

### 2.2 Hardware Acceleration of FlashAttention-3

**FlashAttention-3** maximizes the utilization of hardware acceleration features built into NVIDIA's latest Hopper architecture (H100/H200 GPUs, etc.):

- **TMA (Tensor Memory Accelerator)**: Dedicated hardware that asynchronously accelerates tensor data movement between HBM and SRAM
- **WGMMA (Warpgroup Matrix Multiply-Accumulate)**: Hardware unit that processes matrix multiplication-accumulation operations more efficiently
- **FP8 Support**: Supports 8-bit floating-point (FP8) low-precision format, nearly doubling memory usage and throughput while minimizing precision loss

As a result, FlashAttention-3 achieved **1.5x to 2.0x** faster speed compared to FlashAttention-2 on H100 GPU.

### 2.3 Practice: Enabling FlashAttention in Hugging Face Transformers

Hugging Face's 🤗 Transformers library tightly integrates FlashAttention, allowing it to be easily activated with just the `attn_implementation` argument when loading models. This enables **significant inference speed and memory efficiency improvements** with minimal changes to existing code.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assuming GPU is Hopper architecture or above and flash-attn library is installed
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/gpt-oss-20b"  # Example model ID supporting FlashAttention-3

# 1. Load model with standard attention implementation (Eager)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_eager = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Model with standard attention loaded.")

# 2. Load model with FlashAttention-3 implementation
# This model can internally use vLLM's FlashAttention-3 kernels,
# which are automatically downloaded from the hub through the 'kernels' package.
try:
    model_flash = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="kernels-community/vllm-flash-attn3"  # Enable FlashAttention-3
    )
    print("Model with FlashAttention-3 loaded successfully.")
    print("Note: This requires a compatible GPU (e.g., NVIDIA Hopper series).")
except ImportError:
    print("FlashAttention is not installed or the environment does not support it.")
except Exception as e:
    print(f"An error occurred while loading with FlashAttention: {e}")
```

**Example execution results:**

```
Model with standard attention loaded.
Model with FlashAttention-3 loaded successfully.
Note: This requires a compatible GPU (e.g., NVIDIA Hopper series).
```

### Checkpoint Questions

- What problems does FlashAttention solve in the existing attention mechanism? How does the tiling technique work?
- What are the main hardware acceleration features of NVIDIA Hopper architecture utilized in FlashAttention-3?
- What conditions are required when using the `attn_implementation="kernels-community/vllm-flash-attn3"` option, and what happens when these conditions are not met?

### 2.4 Additional Practice: Direct Use of PyTorch scaled_dot_product_attention

From PyTorch 2.0, the core API includes the built-in `torch.nn.functional.scaled_dot_product_attention` (SDPA) function. This function automatically detects GPU architecture and input tensor properties (dtype, mask presence, etc.) in the backend to select the most efficient attention implementation. That is, under conditions compatible with Ampere architecture or above GPUs, it **automatically calls FlashAttention kernels or memory-efficient kernels**.

```python
import torch
import torch.nn.functional as F

# Use GPU and half-precision (FP16/BF16) to use FlashAttention path
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16  # or torch.bfloat16

# Generate random tensors with batch=1, heads=8, seq_len=1024, embed=64
q = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)
k = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)
v = torch.randn(1, 8, 1024, 64, device=device, dtype=dtype)

# Force use of specific kernel only to verify operation (for debugging/testing)
# enable_flash=True: Attempt to use FlashAttention kernel
# enable_math=False: Disable pure PyTorch math implementation
# enable_mem_efficient=False: Disable memory-efficient implementation
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    try:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        print("FlashAttention kernel was successfully used.")
        print("Output shape:", out.shape)
    except RuntimeError as e:
        print(f"Failed to use FlashAttention kernel exclusively: {e}")
```

The above code uses the `torch.backends.cuda.sdp_kernel` context manager to **force use of only the FlashAttention path**. If the GPU supports it and input conditions (half-precision, no mask, etc.) are met, the FlashAttention kernel is successfully called internally. If conditions are not met, PyTorch outputs useful warning messages like "Flash attention kernel not used because..." and safely falls back to other implementations outside the context manager. This allows developers to easily verify whether the intended optimization is being applied.

### Checkpoint Questions

- What happens when `scaled_dot_product_attention` function is executed in FP32? Why is FlashAttention not used?
- Will FlashAttention be used if the `attn_mask` argument is provided? What mask types does FlashAttention support?
- What hints can be obtained through the warning messages provided by PyTorch?

## 3. Hugging Face Transformers Ecosystem: Latest Trends and Practice

Hugging Face 🤗 Transformers library has evolved beyond a simple **model repository** into a massive integrated platform that supports making the latest AI technologies easily accessible and utilizable by anyone.

### 3.1 Latest Trends

- **Rapid Support for Latest Model Architectures**: Support for **multimodal models** has been significantly strengthened, including the latest LLMs like Vault-GEMMA and EmbeddingGemma, as well as Florence-2 (unified vision) and SAM-2 (advanced segmentation).

- **Integration of Advanced Quantization Technologies**: Natively supports **4-bit floating-point quantization** methods like **MXFP4** introduced with OpenAI's GPT-OSS model. This is more advantageous for dynamic range representation than existing 4-bit integer (INT4) quantization, dramatically reducing memory usage while minimizing accuracy loss, such as loading 120B parameter models on a single 80GB GPU.

- **Zero-Build Kernels**: Through a package called `kernels`, **pre-compiled high-performance kernels** like FlashAttention-3 and Megablocks MoE kernels can be downloaded directly from the hub and used. This eliminates the complex and error-prone process of users compiling source code directly in their own environments.

### 3.2 Practice: Korean Sentiment Analysis Using Pipeline API

Hugging Face's **`pipeline` API** is the most convenient and intuitive tool that abstracts the entire process from tokenization, model inference, to post-processing into just a few lines of code. Let's analyze the positive/negative sentiment of sentences using a model fine-tuned on Korean movie review data (NSMC).

```python
from transformers import pipeline

# Create a pipeline using a model fine-tuned for Korean sentiment analysis
# Model: WhitePeak/bert-base-cased-Korean-sentiment (fine-tuned on NSMC dataset)
classifier = pipeline(
    "sentiment-analysis",
    model="WhitePeak/bert-base-cased-Korean-sentiment"
)

# Sentences to analyze
reviews = [
    "이 영화는 제 인생 최고의 영화입니다. 배우들의 연기가 정말 인상 깊었어요.",
    "기대했던 것보다는 조금 아쉬웠어요. 스토리가 너무 평범했습니다.",
    "시간 가는 줄 모르고 봤네요. 강력 추천합니다!",
    "음악은 좋았지만 전체적으로 지루한 느낌을 지울 수 없었다."
]

# Execute sentiment analysis
results = classifier(reviews)

# Output results
for review, result in zip(reviews, results):
    label = "긍정" if result['label'] == 'LABEL_1' else "부정"
    score = result['score']
    print(f"리뷰: \"{review}\"")
    print(f"결과: {label} (신뢰도: {score:.4f})\n")
```

**Example execution results:**

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

### Checkpoint Questions

- What is **Zero-Build Kernels** among the latest trends in Hugging Face Transformers, and what advantages does it provide?
- What meanings do `LABEL_0` and `LABEL_1` output from the sentiment analysis pipeline above have?
- Why might results differ when running with different Korean sentiment analysis models?

## 4. AI Agent Frameworks: The Era of Automation and Collaboration

The development of LLMs has opened the **AI agent** paradigm that goes beyond a single model performing one task, to **autonomously using multiple tools** and **collaborating with other agents to achieve complex goals**. Various frameworks have emerged to systematically support this, each with distinct philosophies and strengths.

### 4.1 Comparison of Major AI Agent Frameworks

| Framework      | Core Philosophy                                           | Architecture Style                                       | Main Use Cases                                                                                          |
| :------------- | :-------------------------------------------------------- | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| **LangGraph**  | Explicit Control and State-based Orchestration            | Directed Acyclic Graph (DAG) with state, allowing cycles | Building reliable and auditable complex multi-stage agents where human supervision is important         |
| **CrewAI**     | Role-based Collaborative Intelligence                     | Hierarchical, role-based **multi-agent system**          | Automating complex business workflows with clear role division (e.g., market analysis team)             |
| **LlamaIndex** | Data-centric Agents and Advanced RAG                      | Data-centric, event-based workflow                       | Building **question-answering and reasoning** systems for large-scale private/unstructured databases    |
| **Haystack**   | Production-ready Modular Pipeline                         | Modular, branching/looping capable **pipeline**          | Building scalable and robust production-grade **AI applications**                                       |
| **DSPy**       | Declarative LM Programming ("Programming, not prompting") | Declarative, optimizable **pipeline**                    | Tasks requiring highest performance by replacing manual prompt tuning with **data-driven optimization** |

As shown in the table, **LangGraph** focuses on **long-term execution and reliability** by explicitly managing agent states and control flow, while **CrewAI** emphasizes **collaboration** among agents with different expertise. **LlamaIndex** aims for data-centric agents combined with vast knowledge bases, and **Haystack** is strong in practical application of **modular combination pipelines** like search-reasoning. Finally, **DSPy** abstracts prompt engineering itself to advance LLM utilization in a **declarative programming** style. Understanding the philosophy and structure of each framework allows selecting the most appropriate tool based on the nature of the problem to be solved.

### 4.2 DSPy: Declarative Prompt Programming

**DSPy** stands for _Declarative Self-Improving Python_ and is a **declarative prompt programming** framework released by Databricks. It reduces the complexity of managing **long prompt strings** that arise when directly handling LLMs, and allows you to create AI programs with modular composition as if **writing code**. In short, it's designed with the philosophy of "don't hardcode prompts, write them **like programming**."

DSPy's core concepts are divided into three: **LM**, **Signature**, **Module**:

- **LM**: Specifies the language model to use. For example, if you set desired models like OpenAI API's GPT-4, HuggingFace's Llama2, etc. with `dspy.LM(...)` and `dspy.configure(lm=...)`, then all subsequent modules generate results through this LM.

- **Signature**: Like specifying input and output types of functions, it declares the **input and output format** of prompt programs. For example, if you define signature like `"question -> answer: int"`, DSPy automatically generates prompts in a structure that takes `question`(str) and outputs `answer`(int). Signatures describe the structure of prompts given to models and expected output forms (e.g., JSON format, etc.).

- **Module**: Encapsulates **prompt techniques** themselves for solving problems as modules. For example, simple Q&A can be expressed as `dspy.Predict`, complex thinking cases as `dspy.ChainOfThought` (chain of thought), tool-using agents as `dspy.ReAct` modules. Modules have logic implemented internally for how to compose prompts according to the corresponding techniques.

Users combine these three to create **AI programs**, then can optimize by automatically improving module prompts or adding few-shot examples through **Optimizer** built into DSPy. For example, you can make simple combinations like below:

```python
!pip install dspy # Install required library (Databricks DSPy)
import dspy

# 1) LM setup (example: using local Llama2 model API)
llm = dspy.LM('ollama/llama2', api_base='http://localhost:11434') # local server example
dspy.configure(lm=llm)

# 2) Signature declaration: "question(str) -> answer(int)" format
simple_sig = "question -> answer: int"

# 3) Module selection: Predict (basic single-step Q&A)
simple_model = dspy.Predict(simple_sig)

# 4) Execute
result = simple_model(question="How many hours does it take from Seoul to Busan by KTX?")
print(result)
```

The above code creates a module called `simple_model` that defines the task of "output integer answers when receiving questions". Internally, DSPy generates optimal prompts matching these requirements and passes them to the LM. (For example, it constructs prompts like "Q: How many hours does it take from Seoul to Busan by KTX?\nA:" and expects numeric answers.) If the initially obtained answer is inaccurate, you can apply Optimizers like **BootstrapFewShot** to automatically add few-shot examples, or instruct continuous answer improvement with **Refine** modules. In this way, DSPy enables composition and optimization of complex LLM pipelines (e.g., RAG systems, multi-stage chains, agent loops, etc.) in module units.

DSPy's advantage is **improved productivity in prompt engineering**. Since LLM calls are designed within structured frameworks like code, it reduces the time people spend writing long prompt sentences manually and going through trial and error. Also, you can maintain the same module interface while switching various **models/techniques**, enabling **flexible experiments** like testing the same Chain-of-Thought module on both GPT-4 and Llama2 to compare performance. Thanks to the declarative approach, even **changing only part of the program** easily reflects in the entire LLM pipeline, making maintenance easy. Although it's still an early-stage framework, it's gaining attention for presenting the paradigm of **"handling LLMs like programming"**.

### 4.3 Haystack: Document-based Search and Reasoning

**Haystack** is an **open-source NLP framework** developed by Deepset in Germany, mainly used for building **knowledge-based question answering** systems. Haystack's strength lies in **flexible pipeline composition**. Users can easily create **end-to-end NLP systems** that return answers when questions are input by linking a series of stages from databases (document stores) to search engines, reader (Reader) or generator (Generator) models into one Pipeline. For example, **Retrieval QA** like "find answers to questions from given document sets" or Wikipedia-based chatbots can be implemented with Haystack.

Haystack's main components are as follows:

- **DocumentStore**: Literally a database for storing documents. It supports backends like In-Memory, Elasticsearch, FAISS, etc., and stores document text, metadata, embeddings, etc.

- **Retriever**: Plays the role of **searching** for relevant documents regarding user questions (Query). It's diversely implemented from traditional keyword-based methods like BM25 to **Dense Passage Retrieval** models like SBERT, DPR, etc. Retriever finds **top k** relevant documents from DocumentStore.

- **Reader** or **Generator**: Takes searched documents as input to generate final **answers**. **Reader** usually uses Extractive QA models (BERT-based, etc.) to extract correct answer spans from the documents, and **Generator** can generate answers using generative models like GPT. Both can be plugged in as nodes (Node) in Haystack.

- **Pipeline**: Structure that defines **query->response flow** by combining the above elements. There are simple ExtractiveQAPipeline that puts Retriever results into Reader, and GenerativeQAPipeline that creates answers generatively. You can also connect **Retriever + Large LM** like Retrieval-Augmented Generation, or implement multi-stage conditional flows.

Let's look at a **simple practice example** using Haystack. For example, if you want to create a QA system that answers questions using FAQ document collections:

```python
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack import Pipeline

# 1) Create document store and write documents
document_store = InMemoryDocumentStore()
docs = [{"content": "Drama **Squid Game** is a Korean survival drama...", "meta": {"source": "Wikipedia"}}]
document_store.write_documents(docs)

# 2) Configure Retriever and Reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="monologg/koelectra-base-v3-finetuned-korquad", use_gpu=False)

# 3) Build pipeline
pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

# 4) Execute QA
query = "Who is the director of Squid Game?"
result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
print(result['answers'][0].answer)
```

In the above code, we simply put one document in an in-memory document store and built a pipeline combining BM25-based Retriever and Electra Reader trained on Korean KorQuAD data. When you put a query in pipeline.run(), Retriever finds the top 5 documents, and Reader extracts and returns the answer from among them. As a result, you can get an answer like "Hwang Dong-hyuk".

Haystack's powerful point is that **components can be easily replaced or extended** like this. You can switch to Dense Retriever, or attach generative models like GPT-3 as Generator instead of Reader. It also supports complex reasoning scenarios by sequentially/parallelly configuring multiple nodes in the middle like multi-hop QA.

In industrial settings, there are many cases of using Haystack to configure **domain document search** + **QA** services or RAG pipelines that inject external knowledge into **chatbots**. In summary, Haystack is a **framework that ties search engines and NLP models together**, a tool that enables building powerful document-based QA systems with relatively little code.

### 4.4 CrewAI: Role-based Multi-Agent Framework

**CrewAI** is one of the recently spotlighted **AI agent** frameworks, a platform that organizes multiple LLM agents in **team (crew)** form to perform **collaborative work**. While existing frameworks like LangChain were centered on single agents or chains, CrewAI specializes in **role-based multi-agents**. For example, to solve one problem, you can divide roles like **Researcher, Analyst, Writer**, etc., and configure each agent to act autonomously with their own tools and goals while collaborating overall to produce final results.

CrewAI's concepts can be organized by main components as follows:

- **Crew (Team)**: Organization or environment of all agents. Crew objects contain multiple agents and oversee their **collaboration process**. One Crew corresponds to one agent team for achieving specific goals.

- **Agent**: Independent **autonomous AI**, each with defined **role**, **tools**, and **goals**. For example, a "literature researcher" agent uses web search tools to collect information, and a "report writer" agent writes final reports with writing tools and appropriate style. Agents can delegate work to other agents or request results when needed (like people collaborating in teams).

- **Process**: Defines **interaction rules** or **workflows** of agents within Crew. For example, you can set up flows like "Step 1: Researcher collects materials -> Step 2: Analyst summarizes -> Step 3: Writer organizes" as processes. In CrewAI, such processes are also extended with the concept of **Flow**, and agent execution can be controlled according to events or conditions.

Using CrewAI, developers can define each agent's role and tools, create and execute Crews to **automate complex tasks**. Let's look at a simple usage example. For instance, an agent team that finds materials and writes summary reports on given topics:

```python
from crewai import Crew, Agent, tool

# Agent definition: searcher and writer
searcher = Agent(name="Researcher", role="Information Collection", tools=[tool("wiki_browser")])
writer = Agent(name="Writer", role="Report Writing", tools=[tool("text_editor")])

# Create Crew and add agents
crew = Crew(agents=[searcher, writer], goal="Write a 1-page summary report on the given topic")
crew.run(task="Investigate and summarize traditional Korean food.")
```

The above example is conceptual code, but it describes the flow of assigning roles and tools (e.g., wiki browser, text editor functions) to Agents, registering them in Crew, and then executing. During execution, the Researcher agent first searches Wikipedia to gather information, then passes the results to the Writer agent. The Writer organizes the received information and writes a summary report to produce the final answer. All these processes occur automatically without human intervention, and the CrewAI framework manages **execution of each step and message exchange between agents**.

CrewAI's characteristics are **high flexibility and control**. Rather than simply running multiple agents independently, developers can design **collaboration patterns** as desired. Additionally, by finely configuring prompt rules, response formats, etc. for individual agents, **specialized AIs** can be built within teams. In practice, it can be applied to **automated customer support** (e.g., one agent understanding user intent, another agent searching FAQs, another agent generating responses) or **research assistants** (dividing roles to organize literature).

CrewAI is designed to be **compatible with LangChain and others** rather than being a completely new framework, allowing reuse of existing tool chains. However, due to the nature of multi-agent systems, **safety mechanism design** to prevent unexpected interactions or infinite loops is also important. CrewAI recommends setting **restrictions and policies** by role so agents only act within defined boundaries.

In summary, CrewAI is a framework that **systematizes collaboration of role-based autonomous agents**, helping multiple specialized LLMs perform more complex tasks through **division of labor and cooperation** instead of one giant LLM doing everything. This enables approaching multi-agent AI system development in an easy and standardized way.

### Checkpoint Questions

- What roles do the core components of CrewAI - **Crew**, **Agent**, and **Process** - play?
- In what types of AI tasks can CrewAI demonstrate advantages? Consider utilization in specialized document writing or customer support scenarios.
- What design is needed to prevent problems that may occur when multiple agents work together (e.g., infinite loops, conflicts)?

### 4.5 LangGraph: State-based Multi-Agent Orchestration

**LangGraph** is a **low-level orchestration framework** developed by the LangChain team, specialized for building **multi-agent systems with persistent state**. LangGraph manages agent execution as **graph data structures**, where each node represents an agent's state and behavior, and edges express interaction paths. This allows explicit handling of inter-agent message flow, state changes, and **recovery points (checkpoints)** when errors occur, making it suitable for scenarios requiring **reliability** and **durability**.

The core of LangGraph usage is defining a graph object called _StateGraph_ and, when necessary, linking it with **checkpoint storage** to continuously save/recover agent states. For example, even if one agent fails during **long conversations** or **plan execution**, you can design **fault-tolerant** systems that rollback to the last saved state and retry. Additionally, **Human-in-the-loop** intervention is easy, allowing people to review or modify at intermediate states and then continue execution.

Let's examine the concept through a simple LangGraph example. The code below creates a React-style agent with one tool and processes user questions on the graph (assuming Anthropic Claude model):

```python
!pip install langgraph # Install LangGraph library
from langgraph.prebuilt import create_react_agent

# Define a simple tool function
def get_weather(city: str) -> str:
    # Return fixed answer instead of actual external API (example)
    return f"The weather in {city} is always sunny!"

# Create React agent (assuming Anthropic Claude API usage)
agent = create_react_agent(
    model="anthropic:claude-2",    # Anthropic Claude model (API key required)
    tools=[get_weather],
    prompt="You are a helpful assistant."  # Basic prompt
)

# Execute agent (pass user message as graph input)
response = agent.invoke({"messages": [{"role": "user", "content": "What's the weather like in Seoul?"}]})
print(response)
```

In the above code, a **ReAct pattern** agent is created through the `create_react_agent` function. The `tools` list provides the `get_weather` function, allowing the agent to use the tool when needed. When `agent.invoke(...)` is called with the user's message as graph input, LangGraph internally constructs a **state graph (StateGraph)** to track the agent's reasoning process. The `response` contains the final answer produced by the agent (e.g., "The weather in Seoul is always sunny!").

Using LangGraph, such **agent workflows** can be expanded more complexly. You can add multiple agents as nodes, define **message passing paths** between nodes, and set up processes where different agents perform tasks like "question analysis -> information search -> answer organization" according to graph order. LangGraph's **checkpoint** feature allows periodic saving of intermediate states of long-running agents, enabling resumption from the middle rather than the beginning when unexpected errors occur. Additionally, it can integrate with monitoring tools like **LangSmith** to visualize and debug graph execution.

In summary, LangGraph is a framework for ensuring **reliability and persistence** in multi-agent system development. It's useful for building **agent teams that must operate continuously without interruption for long periods** in web services or business automation. Since it integrates with the LangChain ecosystem, existing LangChain users can easily introduce state-based approaches.

### Checkpoint Questions

- What is LangGraph's core feature of **state-based orchestration**, and what advantages does it provide?
- How can LangGraph's **checkpoint** and **Human-in-the-loop** features be utilized in long processes or long-running agents?
- What design is needed to prevent infinite conversations or conflicts between agents in LangGraph-built agent systems?

## 5. Practice: BERT vs Mamba Model Comparison Experiment

Due to the absence of publicly available Korean (NSMC) Mamba classification models, we configured a comparison experiment using the **IMDB English dataset**. We compare **accuracy, inference speed, and GPU memory** using the publicly available checkpoint `trinhxuankhai/mamba_text_classification` for Mamba and a public IMDB classification baseline for BERT.

### 5.1 Environment Setup

```bash
# GPU recommended. Colab/CUDA environment recommended
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install transformers datasets accelerate
```

### 5.2 Dataset Loading (IMDB)

```python
from datasets import load_dataset

imdb = load_dataset("imdb")
imdb_test = imdb["test"]  # 25k samples

# For speed comparison, evaluating on a small sample (e.g., 1000) is acceptable
imdb_test_small = imdb_test.select(range(1000))
```

### 5.3 Model and Tokenizer Loading

- **Mamba**: `trinhxuankhai/mamba_text_classification` (user-provided card)
- **BERT**: Public IMDB classification model (e.g., `textattack/bert-base-uncased-imdb`)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

# 1) Mamba classification model (IMDB trained)
mamba_id = "trinhxuankhai/mamba_text_classification"
tok_mamba = AutoTokenizer.from_pretrained(mamba_id, use_fast=True)
model_mamba = AutoModelForSequenceClassification.from_pretrained(mamba_id).to(device)
model_mamba.eval()

# 2) BERT classification model (public IMDB baseline)
bert_id = "textattack/bert-base-uncased-imdb"
tok_bert = AutoTokenizer.from_pretrained(bert_id, use_fast=True)
model_bert = AutoModelForSequenceClassification.from_pretrained(bert_id).to(device)
model_bert.eval()

print("Loaded:", mamba_id, "|", bert_id)
```

### 5.4 Evaluation Function (Accuracy, Speed, Memory)

```python
import time, numpy as np

@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size=16, max_length=256, warmup=2):
    # Preprocessing
    def enc(batch):
        encodings = tokenizer(
            batch["text"], truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        encodings["labels"] = torch.tensor(batch["label"])
        return encodings

    # Pre-tensorize
    texts = dataset["text"]
    labels = dataset["label"]

    # Batch encoding (on-the-fly also possible for memory saving)
    encoded = [enc({"text": [t], "label": [l]}) for t, l in zip(texts, labels)]

    # Warmup
    for _ in range(warmup):
        for i in range(0, len(encoded), batch_size):
            batch = {k: torch.cat([encoded[j][k] for j in range(i, min(i+batch_size, len(encoded)))], dim=0).to(device)
                     for k in encoded[0].keys()}
            _ = model(**batch)

    # Memory/time measurement
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

# Execute evaluation (1k samples)
res_mamba = evaluate(model_mamba, tok_mamba, imdb_test_small, batch_size=32, max_length=256)
res_bert  = evaluate(model_bert,  tok_bert,  imdb_test_small, batch_size=32, max_length=256)

print("Mamba results:", res_mamba)
print("BERT results:", res_bert)
```

### 5.5 Example Results and Interpretation

**Example (may vary by environment):**

- **Mamba(IMDB)** → `{'accuracy': 0.94, 'throughput': X samples/sec, 'peak_mem_mb': Y MB}`
- **BERT(IMDB)** → `{'accuracy': 0.94, 'throughput': Z samples/sec, 'peak_mem_mb': W MB}`

- **Accuracy**: Based on the provided Mamba model card, Val/Test Acc ≈ 0.94, showing **similar high performance** to BERT baseline.

- **Inference Speed (Throughput)**: Under conditions of input length 256, batch 32, there are variations depending on quantization application or hardware specs. As input length increases, BERT's speed drops sharply due to attention's O(N²) complexity, while **Mamba's linear time O(N) advantage** becomes prominent.

- **Memory Usage (Peak Memory)**: Mamba is theoretically more memory efficient as it doesn't generate massive attention matrices due to state space model (SSM) characteristics. This difference becomes clearer with longer sequences.

**Note**

- The above code compares speed/memory with **1000 samples** (full 25k also possible, but considering practice time).
- For experimental fairness, maintain **same `max_length`, `batch_size`, `dtype`**.
- Results may vary significantly depending on **GPU specs** like Colab T4/V100, A100, RTX 30/40 series.

### Checkpoint Questions

1. Even though **accuracy was similar** in this comparison, predict how **inference speed/memory** of the two models will differ with **longer input lengths**.
2. List 2 advantages and risks each of Mamba for real service deployment (e.g., library maturity, ecosystem, debugging tools).
3. Change **`max_length=512/1024`** in the same code and re-measure, then summarize Throughput/PeakMem changes in a report.

## 6. Experiment Summary and Implications

Through the **BERT vs Mamba comparison experiment**, we examined the characteristics and pros/cons of both models. In summary, **existing BERT (Transformer)** models still show high accuracy and stable speed for medium-length inputs and are **still efficient in short input environments**. On the other hand, **Mamba (SSM)** models show potential for ultra-long context processing and **efficiency without performance degradation** as input length increases. However, at the current point, Transformer series are validated in terms of model completeness and optimization, while Mamba is in the research stage, so **Transformers have some advantage in general tasks** (e.g., accuracy comparison in this experiment).

**Which model is suitable for which situation?** First, **input sequence length** is the determining factor. For **short sentence-level tasks** (e.g., sentence classification, short-answer QA, etc.), using Transformers like BERT is advantageous in terms of implementation ease and performance. Rich pre-training and tuning techniques are accumulated, making it easy to achieve high accuracy with short inference latency. For **long context or document-level tasks** (e.g., summarizing documents of thousands of words, sentiment analysis of long texts, etc.), linear architectures like Mamba may be advantageous. This is because Mamba can efficiently process input lengths that are impossible or would consume many resources with Transformers. In fact, Mamba shows the ability to process up to **1 million tokens**, suggesting the possibility of opening the era of ultra-long context LLMs.

From an **inference speed** perspective, judgment should also be based on context length. With short inputs, the two models may have similar speeds or Transformers may be faster, but as input length increases, Transformers **slow down dramatically** as O(n²), so reports suggest that Mamba will show **up to 5x faster inference** in sufficiently long contexts. Additionally, Mamba has strengths in time series data and continuous stream processing due to the nature of state space models, and also has generality that can be widely applied to **speech and sequence data processing beyond language**.

**Service/Production Application Implications:** Currently in production environments, Transformer series (e.g., BERT, GPT) models are mature and widely used in terms of performance and tooling. Mamba is a very promising technology but **library support, community, and pre-trained model pools** are not as rich as Transformers yet. Therefore, more stability validation may be needed to immediately introduce Mamba as a replacement in industry. However, for services that have had difficulty with ultra-long context processing due to **memory capacity limits or latency issues**, introducing models like Mamba in the future could provide a breakthrough. For example, in **long legal document analysis services** or **chatbots that need to maintain long-term conversation history**, Mamba architecture has the potential to be a game changer.

Additionally, attention should be paid to future hybrid models (e.g., **Jamba: Transformer+Mamba mixed experts**) and competition with other linear sequence models. Currently, it can be viewed as **Transformer's universality vs. Mamba's specificity**, and in actual production, approaches to **mutually complementary utilization** of both methods are also considered. For example, a system that processes general conversations with Transformers and switches to Mamba mode when ultra-long context processing is needed for specific requests would be possible.

In summary, **BERT** and **Mamba** each have their strengths and different use cases. **Mature BERT series** are suitable for **short inputs/existing tasks**, while **Mamba** shows potential for **long inputs/new expansion tasks**. If research and technological development continue, it is expected that cases where SSMs like Mamba complement or replace Transformer limitations will gradually increase. When applying to actual services, current model stability, support tools, licenses, etc. should be comprehensively considered, but from a **future-oriented perspective, architectural innovation for ultra-long context and high-efficiency inference is being realized**, and this comparison experiment of the two models provides meaningful insights.

### Checkpoint Questions

- What are the differences in **time complexity** between BERT and Mamba models, and how does this affect long sequence processing?
- Predict how **inference speed** and **memory usage** results will change if the input sentence length is increased to 512 or 1024 tokens in this experiment.
- Why is it difficult to immediately deploy Mamba models in current industry systems? Conversely, what utilization scenarios could make Mamba popular in the future?

---

## References

### Major Papers and Research Materials

- PyTorch Official Blog – _"Better Performance with torch.compile"_ (2023)
- Tri Dao Blog – _"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision"_
- Databricks DSPy Introduction – _Programming, not prompting_

### Technical Documentation and Implementations

- Hugging Face Transformers Documentation & Tutorials
- Deepset Haystack Documentation – _Flexible Open Source QA Framework_
- CrewAI Official Documentation – _Role-based Autonomous Agent Teams_
- LangGraph Official Documentation – _State-based Multi-Agent Orchestration_

### Online Resources and Blogs

- "torch.compile: A Deep Dive into PyTorch's Compiler" - PyTorch Blog
- "FlashAttention-3: The Next Generation of Attention Optimization" - Technical Blog
- "AI Agent Frameworks: A Comprehensive Comparison" - Medium
- "DSPy: The Future of Prompt Engineering" - Databricks Blog
