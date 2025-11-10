# Week 11: Production Agent Systems

## 1. Introduction: 2025, The Dawn of Production Agent Systems

### 1.1. Paradigm Shift: From Single QA Bots to Multi-Agent Systems (LLM-MAS)

If 2023 and 2024 were the "information retrieval" era, focusing on "what LLMs know" through RAG (Retrieval-Augmented Generation), then 2025 is defined as the "task execution" era, concentrating on "what LLMs can do". Companies are no longer satisfied with using LLMs as simple Q&A chatbots. They are moving towards LLM-based Multi-Agent Systems (LLM-MAS), where multiple LLM agents collaborate to automate complex business processes.

This change is not just a trend; it is being practically applied in production environments. According to LangChain's "State of AI Agents" report, 51% of industry experts are already using AI agents in production as of 2025. In particular, 63% of mid-sized companies (between 100 and 2000 employees) are aggressively adopting agents, proving that agent systems are no longer just laboratory toys.

### 1.2. 2025 Market Trend: The Rise of "Agent-first" LLMs

These industrial demands are driving the evolution of LLM models themselves. In early 2025, the release of Anthropic's Claude 3.7 Sonnet and Opus 4 models introduced the new concept of "agent-first" to the market. This signifies that LLMs are moving beyond simply generating text and are now being developed from the initial design phase with the premise of using external tools, performing autonomous actions, and generating complex code.

Notably, the Claude Code model has captured 42% of the market share for code generation tasks among developers, more than double that of OpenAI (21%). This clearly shows that the "first killer app" for agent systems is code generation and automation tasks.

### 1.3. The True Meaning of "Production": The Battle with "Trust"

However, behind the push for production adoption lies a serious problem of trust. While about 70% of organizations are actively exploring or implementing LLM use cases, deploying an actual agent requires a level of risk management similar to putting a self-driving car on the road.

IBM expert Ashoori emphasizes, "Using an agent today is basically grabbing an LLM and allowing it to take actions on your behalf," stressing that systems must be built to be "trustworthy and auditable" from day one. Deloitte's 2025 Technology Trends report also warns that the proliferation of agents with system access rights demands a fundamental paradigm shift in "cybersecurity" and "risk".

Therefore, the core challenge for production agent systems in 2025 is not "performance" but "control" and "trust". The fact that 51% of companies are using agents does not signify success, but rather "the beginning of the challenge". Many of them are hitting real-world problems like the "prototyping trap" or the "MAST failure taxonomy" that will be discussed later in this lecture. The ultimate goal of this course is not simply to build an agent that "works", but to learn the engineering methodology for building a "trustworthy and controllable" production system.

### Checkpoint Questions

- What distinguishes the 2025 "task execution" era from the previous "information retrieval" era?
- Why is trust and control more critical than performance for production agent systems?
- What does "agent-first" mean in the context of LLM development, and why is it significant?

## 2. Theoretical Foundations of Multi-Agent Collaboration Architectures

Before learning frameworks like CrewAI or LangGraph, it is crucial to understand the fundamental collaboration architecture patterns they aim to implement. The choice of architecture determines the fundamental trade-off between "Flexibility" and "Control/Debuggability".

In 2025, production environments prioritize predictability and auditability above all else. Therefore, "Supervisor" or "Hierarchical" patterns are overwhelmingly preferred over "Network" patterns, which offer high flexibility but are difficult to control.

### 2.1. Detailed Analysis of Key Architectural Patterns

**1. Supervisor/Manager → Workers Pattern**

- **Structure**: A central "Supervisor" agent coordinates all other agents, managing task distribution and routing decisions.
- **Core**: Shows excellent performance in "Task Decomposition". It breaks down a complex task into parallelizable chunks, assigns them to specialized "Worker" agents, and then aggregates the results.
- **Application**: Optimal for scenarios where task decomposition is clear, such as large-scale document processing pipelines, web scraping, and OCR workflows. Frameworks like MetaGPT have refined this approach.

**2. Hierarchical Pattern**

- **Structure**: Goes beyond a manager managing workers; it has a tree structure where managers manage other sub-managers. Recent 2025 research like "HALO" has demonstrated the efficiency of this structure.
- **Core**: Can scale complex, multi-layered tasks to a large group of agents and allows for clear delegation of responsibility.
- **Application**: Suitable for complex software development projects or automating multi-departmental business processes.

**3. Network and Custom Workflow Patterns**

- **Network**: All agents can communicate freely with each other. This maximizes flexibility and can be useful for creative brainstorming, but as the number of agents increases, coordination complexity explodes, making it unsuitable for production control.
- **Custom Workflow**: Agents communicate only with a specific subset of other agents according to predefined rules. Used in specific, performance-critical domain workflows where predictability must be guaranteed, such as financial trading systems.

### Checkpoint Questions

- What is the fundamental trade-off between different multi-agent architecture patterns?
- Why are Supervisor and Hierarchical patterns preferred over Network patterns in production environments?
- When would a Custom Workflow pattern be more appropriate than a Supervisor pattern?

## 3. CrewAI: Role-Based Collaborative Orchestration

### 3.1. Core Philosophy: Role-Based Autonomy

CrewAI uses the metaphor of organizing a multi-agent system as a "team". It is designed to have agents collaborate autonomously by assigning each a clear role. Every agent has three core attributes:

1. **role**: "What do you do?" (e.g., Senior Researcher)
2. **goal**: "What must you achieve?" (e.g., Collect the latest information on a specific topic)
3. **backstory**: "What experience do you have?" (e.g., A veteran analyst with 20 years of experience)

This backstory is not just flavor text; it is a powerful prompt engineering technique that guides the LLM to immerse itself more deeply into that persona, generating consistent, high-quality responses. The system is composed of four core components: "Crew" (team), "Agents" (team members), "Tasks" (assignments), and "Process" (work procedure).

### 3.2. 2025 Core Architecture: "Crews" vs "Flows"

As of 2025, CrewAI has evolved beyond a simple "Crew" framework to offer a dual architecture that addresses the diverse needs of production environments.

**CrewAI Crews:**

- **Concept**: A "team" optimized for autonomy and collaborative intelligence.
- **Features**: Agents decide for themselves what to do next and interact dynamically (e.g., hierarchical process).
- **Suited for**: _Fluid and exploratory_ tasks like creative content generation, open-ended research, and strategy development.

**CrewAI Flows:**

- **Concept**: A "workflow" that provides granular, event-driven control.
- **Features**: The order of tasks and state transitions are explicitly defined and executed deterministically. "Crews" can be called as one step within these "Flows" for complex, hybrid use.
- **Suited for**: _Clearly defined and predictable_ tasks that require auditing, such as API orchestration and data processing pipelines.

The emergence of "Flows" is a critical indicator of CrewAI's maturation from a "prototype" level to a "production" level. Early agent systems focused on autonomy, but this led to fatal flaws of "unpredictability" and "difficulty in debugging". As frameworks like LangGraph gained popularity by offering a controllable alternative with "stateful graphs", CrewAI responded to market demands for deterministic workflows by introducing "Flows". This means developers can now flexibly choose the level of "autonomy" vs. "control" based on the nature of the task.

### 3.3. The Importance of State Management

In a multi-agent system, "State" is the core "shared memory" that transfers information and maintains context between agents. "CrewAI Flows" explicitly manage this state through Pydantic models (structured state) or dictionaries (unstructured state).

For example, if the result of one agent's task (e.g., an FAQ bot) changes a specific field in the state object to `status="escalated"`, this state change acts as a trigger, allowing the next agent (e.g., a ticket-issuing bot) to begin its task. This enables production-level orchestration.

### 3.4. Enterprise Trend: CrewAI AMP (Agent Management Platform)

CrewAI is expanding beyond open-source (OSS) to offer an enterprise commercial platform called "AMP". AMP provides a GUI environment where users can build agent crews without writing code (No-Code) using a "visual editor" and "AI copilot". This aims to accelerate AI adoption within companies by allowing not only skilled developers but also subject-matter experts (SMEs) to participate directly in building agent workflows.

### Checkpoint Questions

- What are the three core attributes of a CrewAI agent, and how does the backstory attribute function as a prompt engineering technique?
- What is the key difference between CrewAI Crews and Flows, and when should each be used?
- How does state management enable production-level orchestration in multi-agent systems?

## 4. Mirascope: Type-Safety through Pydantic

### 4.1. The Production Bottleneck: Reliability Issues with Unstructured LLM Outputs

The fundamental output of an LLM is a "string". In a production system, parsing this unpredictable string to execute the next step of logic is extremely unstable and a core cause of system failure.

This is directly linked to the "Inter-Agent Misalignment" problem. For example, if one agent returns a response in YAML format and the next agent expects JSON format, this small format discrepancy will cause the entire workflow to fail immediately.

### 4.2. Mirascope's Solution: Structured I/O with Pydantic

Mirascope directly tackles this chronic problem through "Type-Safety".

The core mechanism involves the `@llm.call` decorator and the `response_model` argument. The developer first defines the desired output format from the LLM as a Pydantic BaseModel.

```python
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str

@llm.call(provider="openai", model="gpt-4o-mini", response_model=Book)
def extract_book(text: str) -> str:
    return f"Extract {text}"

# This function returns a 'Book' object instance, not a string.
book_object: Book = extract_book("The Name of the Wind by Patrick Rothfuss")
```

When this code executes, Mirascope automatically performs two key tasks in the background:

1. **Automatic Schema Injection**: It converts the Book Pydantic model into a JSON schema that the LLM understands and automatically injects it into the prompt, much like OpenAI's "tools" parameter.
2. **Automatic Validation and Parsing**: It _validates and parses_ the JSON-formatted response returned by the LLM back into a Book Pydantic object instance, returning it as a type-guaranteed object.

### 4.3. Overwhelming Simplicity Compared to Native SDKs

Using Mirascope allows you to treat LLM calls just like "type-safe Python functions". _All the boilerplate code_ required when using native SDKs—such as complex "tools" definitions, "tool_choice" settings, `client.chat.completions.create` calls, and response `message.content` parsing—is abstracted away. This allows the developer to focus solely on the business logic (defining the Pydantic model).

Mirascope elevates the LLM from an "unreliable text generator" to a "reliable Object Instantiator". A multi-agent system is essentially software, and software operates on predictable interfaces (APIs). An LLM, returning unpredictable text, cannot be a stable API. However, Mirascope applies a "mandatory schema" (Pydantic) to the LLM's output, ensuring the LLM call returns a validated Book object rather than a `str`. This serves as a critical bridge, incorporating LLMs into the realm of traditional software engineering and providing the most robust method for preventing "Inter-Agent Misalignment" failures at their source.

### Checkpoint Questions

- Why are unstructured LLM outputs a production bottleneck in multi-agent systems?
- How does Mirascope's `response_model` parameter solve the Inter-Agent Misalignment problem?
- What are the two automatic processes that Mirascope performs when using Pydantic models with LLM calls?

## 5. Haystack Agents: Domain-Specific "Agentic RAG"

### 5.1. The Evolution of RAG: From Passive RAG to Active "Agentic RAG"

Haystack is a leading open-source framework specialized in building RAG (Retrieval-Augmented Generation) pipelines. Haystack's definition of an agent's components follows the standard definition: LLM (brain), Tools (interaction), Memory (context), and Reasoning (planning).

Haystack's core concept in 2025 is "Agentic RAG". This means moving beyond a simple linear "Retrieve-Augment-Generate" pipeline to imbuing the RAG pipeline itself with _active decision-making capabilities_.

### 5.2. Core Component: ConditionalRouter

The key technology for implementing "Agentic RAG" is "Conditional Routing" within the pipeline. Haystack's ConditionalRouter component performs this role.

**Case Study: "Fallback to Websearch" Architecture**

The most representative implementation pattern of Agentic RAG is the "Fallback to Websearch".

1. **Initial RAG**: Takes the user's query ($Query$) and retrieves relevant documents from the internal corporate document store.
2. **Decision Prompt**: The LLM is explicitly instructed, "If you cannot answer with the provided documents, do not say anything else and return _only_ the keyword 'no_answer'."
3. **Router (Core)**: The ConditionalRouter component intercepts the LLM's response.
4. **Branching**:
   - **IF (Answer Successful)**: If the LLM generates a valid answer (not "no_answer"), the ConditionalRouter passes this answer to the user and terminates the pipeline normally.
   - **IF (Answer Failed, "no_answer" returned)**: The ConditionalRouter detects this keyword and routes the user's _original query_ ($Query$) to a _different tool (web search)_, such as SerperDevWebSearch.
5. **Secondary RAG**: A new answer is generated based on the web search results and delivered to the user.

This Agentic RAG architecture is the production standard for solving RAG's chronic problem of "hallucination upon retrieval failure". Traditional RAG lacks a "task verification" step, so even if the retrieved documents are low-quality, the LLM will generate a plausible-sounding lie based on them. This is a classic example of a "shallow check failure" that is difficult for users to detect.

Haystack's "Agentic RAG" embeds a _verification step_ into the pipeline using the explicit "no_answer" signal. The ConditionalRouter acts as a "Judge" agent that behaves according to this verification result. This elevates the RAG system's reliability from a passive attitude of "it's good if it's found, oh well if not" to an _active problem-solving_ approach: "If it's not found, execute Plan B (web search)".

### Checkpoint Questions

- What is the key difference between traditional RAG and Agentic RAG?
- How does the ConditionalRouter component enable active decision-making in RAG pipelines?
- Why is the "no_answer" verification step critical for preventing hallucination in RAG systems?

## 6. Low-Code Integration Platforms and the "Prototyping Trap"

### 6.1. Visual Workflow Builders (Flowise, LangFlow, n8n)

Platforms like Flowise AI, LangFlow, and n8n allow users to visually design agent workflows using a "drag-and-drop" GUI (Graphical User Interface).

This has a significantly lower barrier to entry compared to code-centric frameworks like CrewAI or LangGraph, offering a powerful advantage for "rapid prototyping" to quickly validate ideas.

- **n8n**: Its unique strength lies in combining AI agent features with over 1000 traditional business automation integrations (CRM, email, Slack, etc.).
- **Flowise/LangFlow**: Focus on visually implementing the LangChain ecosystem.

### 6.2. The Harsh Reality of 2025: The "Prototyping Trap"

However, field data from 2025 warns that these tools hit severe limitations when applied to production environments. A report from ZenML and feedback from the developer community call this the "Prototyping Trap".

**Case Study: LangFlow's Production Limitations**

1. **Memory Leak**: LangFlow's caching mechanism has a known memory leak issue. When files are repeatedly uploaded or components are rebuilt, memory usage doubles, causing frequent application crashes in RAG pipelines that handle large documents.
2. **Scalability and Concurrency Issues**: When handling multiple concurrent LLM queries, it exhibits severe latency and 100% CPU usage, revealing its inability to handle real service traffic.
3. **File Upload Limits**: The file upload capacity, essential for RAG systems, is limited to 100MB by default, making it difficult to build large-scale knowledge bases.

Due to these limitations, experienced developers are using platforms like LangFlow only as a "visual demonstrator for prototyping and showing logic to other teams," and then following a workflow of "rewriting everything in Python code" for production deployment.

Ultimately, the value of low-code platforms like LangFlow lies in "design" and "communication", not "deployment". One must recognize that the "convenience" of a GUI is in a trade-off relationship with production "reliability", "scalability", and "maintainability".

### Checkpoint Questions

- What are the main advantages of low-code visual workflow builders for prototyping?
- What is the "Prototyping Trap" and why does it occur with platforms like LangFlow?
- Why do experienced developers typically rewrite low-code prototypes in Python for production?

## 7. The Evolution of LLM-Intrinsic Capabilities: From Toolformer to Next-Generation Function Calling

### 7.1. Two Approaches: Extrinsic Frameworks vs. Intrinsic Capabilities

The frameworks discussed so far (CrewAI, Haystack) are "extrinsic" frameworks that control agent logic from "outside" the LLM. In contrast, research to "internalize" (intrinsic) tool-using capabilities _within the LLM itself_ has been actively underway, starting with Toolformer.

### 7.2. Foundational Research: Toolformer and Self-Supervised Learning

Toolformer's goal was to have the LLM learn _by itself_ when to call an API (tool) without human intervention. Its core idea was an ingenious self-supervised learning method called "Loss-Based Filtering".

1. Start with a huge plain text corpus ($C$).
2. The model itself samples random [API Call] candidates, wondering, "What if I called an API (e.g., a calculator) at this position?"
3. It _actually executes_ the API to get the $Result$.
4. **Filtering (The Core)**: It checks if inserting the `` text snippet into the original text _significantly reduces the prediction loss_ for the _original text_ ($C$) that comes _after_ it.
5. **Meaning**: This is a process where the model _verifies for itself_ whether the API call and its result are _useful_ hints for predicting future text.
6. Only the API calls deemed useful (loss was reduced) are kept, creating a new dataset ($C^*$). The model is then fine-tuned on this dataset.

### 7.3. 2025 Status: Gorilla LLM and the Berkeley Function Calling Leaderboard (BFCL)

Toolformer's idea has been commercialized and advanced in 2025 under the name "Function Calling".

- **Gorilla LLM**: A LLaMA-based model fine-tuned _exclusively_ for writing API calls. In benchmarks, it demonstrated the ability to generate API calls more accurately than GPT-4.
- **BFCL (Berkeley Function Calling Leaderboard)**: The industry-standard benchmark for evaluating which LLM generates function calls most accurately and reliably.
- **Latest Technology (OpenFunctions-v2)**: State-of-the-art function calling in 2025 goes beyond calling a single function. Gorilla's OpenFunctions-v2 supports complex scenarios like "calling multiple functions in parallel for a single prompt" or "selecting multiple appropriate functions from a provided list." It also natively supports various languages, including Python, Java, and REST APIs.

The development of these "intrinsic" capabilities suggests that in the future (like Anthropic's "agent-first" models), "extrinsic" orchestration frameworks like CrewAI will become thinner, and we will rely more on the LLM's own intrinsic function-calling abilities. However, as of 2025, intrinsic function calling still lacks "explainability" as to "why it chose that tool", leaving debugging and reliability in complex production environments as an ongoing challenge.

### Checkpoint Questions

- What is the fundamental difference between extrinsic and intrinsic approaches to agent capabilities?
- How does Toolformer's "Loss-Based Filtering" method enable self-supervised learning of tool usage?
- What are the current limitations of intrinsic function calling in production environments?

## 8. Why Do Production Agents Fail? - The MAST Failure Taxonomy (2025)

When agent systems are deployed to production, they frequently don't just "perform poorly" — they _collapse_. An arXiv paper (arXiv:2503.13657), with versions published in March (v1) and October (v3) 2025, presents a systematic analysis of these failure causes: the MAST (Multi-Agent System Failure Taxonomy).

The most important lesson from this research is that agent systems do not fail because "the LLM (e.g., GPT-4) isn't smart enough." The fundamental cause of failure is a "flawed organization structure"—that is, the _system design_.

### 8.1. MAST: The 3 Major Failure Categories and Real-World Examples

MAST classifies 14 unique failure modes into 3 high-level categories:

**1. Specification Issues (41.8% of failures)**

- **Cause**: Flawed initial setup.
- **Detailed Modes**: Disobeying task specification, missing role constraints, lack of termination criteria, or poor task decomposition.
- **Production Solution**: CrewAI's explicit role, goal, and backstory definitions and the deterministic workflows of "Flows" mitigate this problem.

**2. Inter-Agent Misalignment (36.9%)**

- **Cause**: Miscommunication that occurs during execution.
- **Detailed Modes**: Ignoring other agents' input, failure to propagate context.
- **Critical Example**: "The planner agent assigned the task in YAML format, but the executor agent expected JSON format." This small discrepancy halts the entire workflow.
- **Production Solution**: Mirascope's `response_model=PydanticModel` forces all inter-agent I/O to be _type-safe objects_, preventing such format mismatches.

**3. Task Verification Failures (21.3%)**

- **Cause**: Inadequate quality control.
- **Detailed Modes**: Absence of a "Judge" agent, premature termination, or missing verification steps.
- **Critical Example**: A verification agent only performs _shallow checks_, such as "does the code compile?" (e.g., a chess program compiles but plays with incorrect game rules).
- **Production Solution**: Haystack's "Agentic RAG" using a ConditionalRouter to add an explicit verification step (e.g., checking for "no_answer") is a direct solution to this problem.

### 8.2. Core Table: MAST - Multi-Agent System Failure Taxonomy (2025)

| Failure Category              | Occurrence Rate | Description                                                                                  | Production Solution from This Lecture                                                                                          |
| :---------------------------- | :-------------- | :------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| 1. Specification Issues       | 41.8%           | Flawed prompts, missing roles/constraints, task decomposition failure.                       | CrewAI: Clear role, goal definition. CrewAI Flows: Forcing "specification" as deterministic code.                              |
| 2. Inter-Agent Misalignment   | 36.9%           | Communication failure, context loss, data format mismatch (e.g., YAML vs. JSON).             | Mirascope: Forcing `response_model` to Pydantic objects, ensuring type-safe I/O between agents.                                |
| 3. Task Verification Failures | 21.3%           | Absence of a "Judge" agent, shallow checks, premature termination without error recognition. | Haystack (Agentic RAG): Using ConditionalRouter as a "verification" step to check context quality and execute a fallback plan. |

This table shows a direct 1:1 mapping between the 3 major production challenges of 2025 (MAST) and the 3 core frameworks we learned (CrewAI, Mirascope, Haystack). This proves that these frameworks are not "trends" but were born from real production "needs". Therefore, these three frameworks should not be seen as individual competitors, but as a "Solution Stack" to be used together to build a production system.

### Checkpoint Questions

- What is the MAST failure taxonomy, and why is it significant for understanding production agent failures?
- Which failure category has the highest occurrence rate, and what are its main causes?
- How do the three frameworks (CrewAI, Mirascope, Haystack) map to the three MAST failure categories?

## 9. [Lab] Designing an Automated Customer Support System Prototype

### 9.1. Objective

To integrate the strengths of the three core frameworks (CrewAI, Haystack, Mirascope) and the lessons from the MAST failure taxonomy to design a production-grade hybrid architecture for the "Automated Customer Support System" required by the syllabus.

### 9.2. Architecture Blueprint: A "Flow-calls-RAG-calls-Crew" Hybrid

We will go beyond a simple CrewAI script to design a robust system that utilizes the strengths of each framework and avoids the weaknesses identified by MAST.

### 9.3. Step 1: Defining Data Integrity (Mirascope + Pydantic)

First, we define the Pydantic model that will serve as the "State" for the entire system. This is the "contract" that prevents "Inter-Agent Misalignment" failures at the source.

```python
from pydantic import BaseModel, Literal
from typing import Optional

class CustomerTicketState(BaseModel):
    """
    The shared state object for the entire system.
    Applies Mirascope's type-safety philosophy.
    """
    original_query: str
    customer_id: Optional[str] = None
    status: Literal["new", "faq_answered", "escalated", "ticket_created"]
    category: Literal["billing", "technical", "general", "unknown"]
    priority: Literal["low", "medium", "high"]
    faq_response: Optional[str] = None
    final_summary: Optional[str] = None
    ticket_id: Optional[str] = None
```

### 9.4. Step 2: Overall Orchestration (CrewAI Flows)

This acts as the "brain" of the system, managing the CustomerTicketState object. To solve "Specification Issues", we use the deterministic Flows architecture as the main controller, not an autonomous Crew.

```python
from crewai.flow import Flow

@Flow
class CustomerSupportFlow:
    """
    The main orchestrator.
    Manages the 'CustomerTicketState' object and advances the steps.
    """
    def __init__(self, state_model=CustomerTicketState):
        self.state = state_model

    @step
    def start(self, query: str):
        # Initialize state
        self.state.original_query = query
        self.state.status = "new"
        self.state.category = "unknown"
        # Step 1: Run FAQ Agent
        self.run(start_at="run_faq_agent")

    @step
    def run_faq_agent(self):
        # Step 3.1: Call Haystack Agentic RAG
        response, status = faq_rag_agent.run(self.state.original_query)

        if status == "answered":
            # Solved by FAQ -> End flow
            self.state.status = "faq_answered"
            self.state.faq_response = response
            self.run(next="end_flow")
        else:
            # Cannot be solved by FAQ -> Escalate to Step 2 (Ticketing)
            self.state.status = "escalated"
            self.run(next="run_triage_crew")

    @step
    def run_triage_crew(self):
        # Step 3.2: Call CrewAI Crew
        # 'Flows' calls 'Crews' and passes the state object
        updated_state = triage_crew.kickoff(self.state.model_dump())  # Pass as dict
        self.state = CustomerTicketState(**updated_state)  # Re-validate with Pydantic
        self.run(next="end_flow")

    @step
    def end_flow(self):
        # Return final state
        return self.state
```

### 9.5. Step 3.1: First-Level Response - FAQ Bot (Haystack Agentic RAG)

This is the agent called in the `run_faq_agent` step. To perform "Task Verification", it uses Haystack's Agentic RAG to search the internal FAQ DB.

```python
# (Pseudo-code - Haystack pipeline configuration)

# 1. LLM modified with a prompt to return 'no_answer'
qa_llm = OpenAIChatGenerator(model="gpt-4o-mini",
    prompt_template="...If context is not enough, return 'no_answer'.")

# 2. Conditional Router (Core of Agentic RAG)
router = ConditionalRouter(routes=[
    {"condition": "'no_answer' in replies", "output": "fallback",...},
    {"condition": "'no_answer' not in replies", "output": "answer",...}
])

# 3. Pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=...))
rag_pipeline.add_component("prompt_builder",...)
rag_pipeline.add_component("qa_llm", qa_llm)
rag_pipeline.add_component("router", router)  # Verification step

class FAQAgent:
    def run(self, query):
        result = rag_pipeline.run({"query": query, "prompt_builder": {"query": query}})
        if "answer" in result["router"]:
            return result["router"]["answer"].content, "answered"
        else:
            # 'fallback' was triggered (answer not found)
            return None, "escalated"

faq_rag_agent = FAQAgent()
```

### 9.6. Step 3.2: Second-Level Response - Ticketing Crew (CrewAI Crew)

This is the "expert team (Crew)" called in the `run_triage_crew` step. When the FAQ bot fails, it receives the escalated CustomerTicketState and performs the complex task of issuing a ticket.

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool

# Define Tools (e.g., DB lookup, Ticket creation)
class CustomerDBTool(BaseTool):
    name: str = "Customer Database Lookup"
    description: str = "Looks up customer details by query text."
    def _run(self, query: str) -> dict:
        #... DB lookup logic...
        return {"customer_id": "C-123", "priority": "high"}

class JiraTicketTool(BaseTool):
    name: str = "Jira Ticket Creator"
    description: str = "Creates a new support ticket in Jira."
    def _run(self, summary: str, category: str, priority: str) -> str:
        #... JIRA API call logic...
        return "JIRA-TICKET-567"

# Agent 1: Classifier (Query + State Context)
classifier_agent = Agent(
    role="Triage Specialist",
    goal="Analyze the query and current state to categorize the issue.",
    backstory="Expert in routing complex support issues.",
    ...
)

# Agent 2: DB Look-up (Uses Tool)
db_agent = Agent(
    role="Database Analyst",
    goal="Find customer ID in query and lookup DB for priority.",
    backstory="Connects to internal systems to fetch customer data.",
    tools=[CustomerDBTool()]
)

# Agent 3: Ticket Creator (Uses Tool + Final Summary)
ticketing_agent = Agent(
    role="Ticketing Agent",
    goal="Summarize all info and create a JIRA ticket.",
    backstory="Formats information for engineers.",
    tools=[JiraTicketTool()]
)

# Tasks
classify_task = Task(
    description="Classify query: '{original_query}'. Current state: '{status}'.",
    agent=classifier_agent,
    expected_output="JSON with 'category' (billing, technical, general)."
)
db_task = Task(
    description="Lookup customer details for query: '{original_query}'.",
    agent=db_agent,
    context=[classify_task],  # Executes after classification
    expected_output="JSON with 'customer_id' and 'priority'."
)
ticket_task = Task(
    description="Create a final ticket using all collected information.",
    agent=ticketing_agent,
    context=[db_task],  # Executes after DB lookup
    expected_output="Final JSON state update with 'ticket_id'."
)

triage_crew = Crew(
    agents=[classifier_agent, db_agent, ticketing_agent],
    tasks=[classify_task, db_task, ticket_task],
    process=Process.sequential  # Execute sequentially
)
```

### 9.7. Lab Architecture Summary

This hybrid architecture meets all the requirements of a 2025 production system:

1. **Mirascope (Pydantic)**: Defines a "data contract", CustomerTicketState, to prevent "Inter-Agent Misalignment" at the source.
2. **CrewAI Flows**: Acts as the main orchestrator, managing the CustomerTicketState object and ensuring a deterministic workflow (solves "Specification Issues").
3. **Haystack Agentic RAG**: Serves as the first line of defense (FAQ), performs "Task Verification" via the ConditionalRouter, and safely passes the flow to the next step upon failure.
4. **CrewAI Crew**: Serves as the second line of defense (Ticketing), where specialized agents (classify, DB, create) collaborate to perform complex "Task Decomposition".

This is a robust architecture that solves all three of the major production challenges (MAST) of 2025.

### Checkpoint Questions

- How does the hybrid architecture combine the strengths of CrewAI, Mirascope, and Haystack?
- Why is the Pydantic state model critical for preventing Inter-Agent Misalignment?
- How does the ConditionalRouter in the FAQ agent implement Task Verification?

## 10. Appendix: Core Framework and Platform Comparison

### 10.1. Core Table 2: 2025 Multi-Agent Framework Comparison

| Framework | Core Philosophy              | Primary Architecture Model                                                  | Production Suitability & 2025 Status                                                                                                                                              |
| :-------- | :--------------------------- | :-------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CrewAI    | Role-Based Collaboration     | Crews: Autonomous Team (Autonomy↑) Flows: Deterministic Workflow (Control↑) | High. Production-ready with the "Flows" architecture and AMP platform. Balances autonomy and control.                                                                             |
| LangGraph | State-Based Control          | Cyclical Graph                                                              | Very High. Explicitly defines and passes "State" between nodes. Optimal for complex, conditional logic and tasks requiring high debuggability.                                    |
| Haystack  | Data-Centric                 | Pipelines + Routers                                                         | High (RAG-Specific). Most powerful for building domain-specific, knowledge-based agents through the "Agentic RAG" concept.                                                        |
| AutoGen   | Conversational Collaboration | Group Chat                                                                  | Medium (Research). Simulates natural language conversations between agents. Flexible but unpredictable and hard to control. Moved to _maintenance mode_ in Oct 2025 by Microsoft. |

### 10.2. Core Table 3: Low-Code Platform Production Readiness Assessment

| Platform   | Core Function                          | Prototyping                         | Production Readiness (As of 2025)                                                                                  |
| :--------- | :------------------------------------- | :---------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| LangFlow   | LangChain Visualization (Python-based) | Excellent. Very fast and intuitive. | Very Low (Risky). The "Prototyping Trap". Severe memory leaks, caching, and concurrency issues. Assumes a rewrite. |
| Flowise AI | LangChain Visualization (JS-based)     | Excellent. Fast and intuitive.      | Low. Similar scalability limits to LangFlow. Lacks a traditional automation layer.                                 |
| n8n        | AI + Traditional Automation            | Good.                               | High. Strength in 1000+ business app integrations. Optimal for linking AI agents to legacy systems.                |
| CrewAI AMP | Enterprise Agent Management            | Excellent (No-Code)                 | Very High. Provides a visual builder, monitoring, deployment, and governance based on proven OSS logic.            |

## References

1. What's next for AI? - Deloitte, accessed November 10, 2025, https://www.deloitte.com/us/en/insights/topics/technology-management/tech-trends/2025/tech-trends-ai-agents-and-autonomous-ai.html
2. LLMs for Multi-Agent Cooperation | Xueguang Lyu, accessed November 10, 2025, https://xue-guang.com/post/llm-marl/
3. LangChain State of AI Agents Report, accessed November 10, 2025, https://www.langchain.com/stateofaiagents
4. 2025 Mid-Year LLM Market Update: Foundation Model Landscape + Economics, accessed November 10, 2025, https://menlovc.com/perspective/2025-mid-year-llm-market-update/
5. AI Agents in 2025: Expectations vs. Reality - IBM, accessed November 10, 2025, https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality
6. Multi-Agent and Multi-LLM Architecture: Complete Guide for 2025 ..., accessed November 10, 2025, https://collabnix.com/multi-agent-and-multi-llm-architecture-complete-guide-for-2025/
7. Designing Cooperative Agent Architectures in 2025 | Samira ..., accessed November 10, 2025, https://samiranama.com/posts/Designing-Cooperative-Agent-Architectures-in-2025/
8. Hierarchical Multi-Agent Systems: Concepts and Operational Considerations - Over Coffee, accessed November 10, 2025, https://overcoffee.medium.com/hierarchical-multi-agent-systems-concepts-and-operational-considerations-e06fff0bea8c
9. #11: AIAgents -CrewAI: How Role-Based Agents Work Together | by Jayakrishnan M | Oct, 2025 | Medium, accessed November 10, 2025, https://medium.com/@jmelethil/11-aiagents-crewai-how-role-based-agents-work-together-87662ad25f33
10. CrewAI Guide: Build Multi-Agent AI Teams in October 2025, accessed November 10, 2025, https://mem0.ai/blog/crewai-guide-multi-agent-ai-teams
11. Building Your First AI Customer Support System with CrewAI: A ..., accessed November 10, 2025, https://medium.com/@tahaML/building-your-first-ai-customer-support-system-with-crewai-a-beginners-guide-f6a41f04fd2e
12. Introduction - CrewAI, accessed November 10, 2025, https://docs.crewai.com/en/introduction
13. Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. - GitHub, accessed November 10, 2025, https://github.com/crewAIInc/crewAI
14. Crewai vs LangGraph: Know The Differences - TrueFoundry, accessed November 10, 2025, https://www.truefoundry.com/blog/crewai-vs-langgraph
15. Mastering Flow State Management - CrewAI, accessed November 10, 2025, https://docs.crewai.com/en/guides/flows/mastering-flow-state
16. Crew AI, accessed November 10, 2025, https://www.crewai.com/
17. Why Do Multi-Agent LLM Systems Fail? | by Anna Alexandra ..., accessed November 10, 2025, https://thegrigorian.medium.com/why-do-multi-agent-llm-systems-fail-14dc34e0f3cb
18. Why Multi-Agent LLM Systems Fail: Key Issues Explained ... - Orq.ai, accessed November 10, 2025, https://orq.ai/blog/why-do-multi-agent-llm-systems-fail
19. Mirascope - LLMs Text Viewer | Mirascope, accessed November 10, 2025, https://mirascope.com/docs/mirascope/llms-full
20. Response Models | Mirascope, accessed November 10, 2025, https://mirascope.com/docs/mirascope/learn/response_models
21. Welcome - Mirascope, accessed November 10, 2025, https://mirascope.com/docs/mirascope
22. Structured Outputs - Mirascope, accessed November 10, 2025, https://mirascope.com/docs/mirascope/guides/getting-started/structured-outputs
23. Haystack Documentation, accessed November 10, 2025, https://docs.haystack.deepset.ai/docs/intro
24. Haystack vs. FlowiseAI: Comparing AI-Powered Development Platforms - SmythOS, accessed November 10, 2025, https://smythos.com/developers/agent-comparisons/haystack-vs-flowiseai/
25. Agents - Haystack Documentation, accessed November 10, 2025, https://docs.haystack.deepset.ai/docs/agents
26. Build an Agentic RAG Pipeline in deepset Studio - Haystack, accessed November 10, 2025, https://haystack.deepset.ai/blog/agentic-rag-in-deepset-studio
27. Building an Agentic RAG with Fallback to Websearch | Haystack, accessed November 10, 2025, https://haystack.deepset.ai/tutorials/36_building_fallbacks_with_conditional_routing
28. Why Do Multi-Agent LLM Systems Fail? - arXiv, accessed November 10, 2025, https://arxiv.org/pdf/2503.13657
29. Langflow | Low-code AI builder for agentic and RAG applications, accessed November 10, 2025, https://www.langflow.org/
30. Dify: Leading Agentic Workflow Builder, accessed November 10, 2025, https://dify.ai/
31. FlowiseAI vs. Langflow: Compare top AI agent builders - SmythOS, accessed November 10, 2025, https://smythos.com/developers/agent-comparisons/flowiseai-vs-langflow/
32. AI Agent Orchestration Frameworks: Which One Works Best for You ..., accessed November 10, 2025, https://blog.n8n.io/ai-agent-orchestration-frameworks/
33. CrewAI vs LangGraph vs AutoGen: Choosing the Right Multi-Agent AI Framework, accessed November 10, 2025, https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen
34. 9 AI Agent Frameworks Battle: Why Developers Prefer n8n, accessed November 10, 2025, https://blog.n8n.io/ai-agent-frameworks/
35. We Tried and Tested 8 Langflow Alternatives for Production-Ready AI Workflows - ZenML, accessed November 10, 2025, https://www.zenml.io/blog/langflow-alternatives
36. LangFlow vs Flowise vs n8n vs Make - Reddit, accessed November 10, 2025, https://www.reddit.com/r/langflow/comments/1ij66dl/langflow_vs_flowise_vs_n8n_vs_make/
37. We Tried and Tested 8 Langflow Alternatives for Production-Ready ..., accessed November 10, 2025, https://zenml.io/blog/langflow-alternatives
38. The Best Langflow vs Flowise Comparison to Guide Your AI Tool Decision - Lamatic.ai Labs, accessed November 10, 2025, https://blog.lamatic.ai/guides/langflow-vs-flowise/
39. Toolformer: Language Models Can Teach Themselves to Use Tools | Research - AI at Meta, accessed November 10, 2025, https://ai.meta.com/research/publications/toolformer-language-models-can-teach-themselves-to-use-tools/
40. Toolformer: How Language Models Learn to Use Tools by ... - Medium, accessed November 10, 2025, https://medium.com/@darshantank_55417/toolformer-how-language-models-learn-to-use-tools-by-themselves-9724fb64ed0e
41. Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates - arXiv, accessed November 10, 2025, https://arxiv.org/html/2509.18076v1
42. FunReason: Enhancing Large Language Models' Function Calling via Self-Refinement Multiscale Loss and Automated Data Refinement - arXiv, accessed November 10, 2025, https://arxiv.org/html/2505.20192v1
43. [2305.15334] Gorilla: Large Language Model Connected with Massive APIs - arXiv, accessed November 10, 2025, https://arxiv.org/abs/2305.15334
44. Gorilla, accessed November 10, 2025, https://gorilla.cs.berkeley.edu/
45. Why Do Multi-Agent LLM Systems Fail? - arXiv, accessed November 10, 2025, https://arxiv.org/html/2503.13657v2
46. [2503.13657] Why Do Multi-Agent LLM Systems Fail? - arXiv, accessed November 10, 2025, https://arxiv.org/abs/2503.13657
47. WHY DO MULTI-AGENT LLM SYSTEMS FAIL? - OpenReview, accessed November 10, 2025, https://openreview.net/pdf?id=wM521FqPvI
48. AutoGen vs CrewAI vs LangGraph: AI Framework Comparison 2025 - JetThoughts, accessed November 10, 2025, https://jetthoughts.com/blog/autogen-crewai-langgraph-ai-agent-frameworks-2025/
