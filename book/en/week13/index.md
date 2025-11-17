# Week 13: Ontology and AI - Modeling Reality and Operating it with AI

## Introduction: A New "Ontological Literacy" for the AI Era

As Artificial Intelligence (AI) establishes itself as the "operating system" across society, a new perspective for viewing AI is required. These lecture notes provide an in-depth explanation of the core philosophy and concepts of a 16-week course plan titled "Ontology and AI: Modeling Reality and Operating it with AI." The ultimate goal of this course is not the simple understanding of AI technology, but the cultivation of "Ontological Literacy."

"Ontological Literacy" refers to the strategic capability to recognize the complex structures and rules of the real world we belong to, define them as an explicit model (i.e., an ontology) (Modeling Reality), and design AI to act intelligently upon that model and operate reality (Operating it with AI).

These lecture notes provide an in-depth, step-by-step explanation of the process of cultivating this "Ontological Literacy." This is based on a "Decision Science" and "Ontology-First" philosophy that moves beyond the traditional perspective of gaining "insight" from "data," and instead transforms "insight" into "action" and "prediction" into "operation."

## 1. The Paradigm Shift - Beyond Prediction to Action

### 1.1. AI's Last Mile: From "Data Science" to "Decision Science"

#### The Core Problem: "Data-Rich, Decision-Poor"

Many modern organizations face a paradoxical "Data-Rich, Decision-Poor" situation, where despite accumulating vast amounts of data, they struggle to make practical, better "decisions" from that data. A significant gap, or "Last Mile," has emerged as a serious problem, where countless "insights" are derived but fail to lead to actual business "actions" or "outcomes." This gap stems from the fundamental difference between "Data Science (DS)" and "Decision Science (DSci)."

#### Definition and Limits of "Data Science (DS)": The Realm of Prediction

Data Science (DS) essentially focuses on "technical" and "algorithmic" aspects. The core objective of data science is to collect, process, and analyze large-scale data to discover hidden patterns, "predict" the future, and thereby generate "insights." Metaphorically, a data scientist answers the question, "What is the data telling us?" and is the person who builds the complex and sophisticated "dashboard" in an aircraft's cockpit.

However, the outputs of data science—dashboards, predictive models, analysis reports—are not "actions" in themselves. No matter how excellent the data and analysis results are, a "person" must ultimately interpret them and make a decision. In this process, human "cognitive biases," the complexity of data interpretation, and organizational inertia intervene, creating a "last mile" bottleneck where data-driven rational decisions fail to translate into actual actions.

#### Definition and Goal of "Decision Science (DSci)": Action and Impact

"Decision Science (DSci)" is the discipline dedicated to bridging this "last mile" gap. Decision science encompasses all activities that ensure the outputs of data science lead to tangible organizational "impact." It is an interdisciplinary field that builds upon the technical capabilities of data science, integrating them with **business acumen, behavioral science, hypothesis formulation and testing, and strong communication skills.**

The core objective of decision science is not merely to "predict" the future, but to "prescribe" the "optimal action" based on data. To return to the aircraft metaphor, the decision scientist is the "pilot." They read the dashboard (data) created by the data scientist, comprehensively consider the current flight path, fuel, and weather conditions (context), and answer the question, "So, what should we do?" Most importantly, they make the "action" of deciding "which route to take, why, and how," and _actually move the controls_ to guide the aircraft to its destination.

The paradigm shift from data science to decision science means moving away from viewing AI systems as simple "techno-statistical" problems (e.g., achieving 90% model accuracy). Decision science is the discipline of designing a "socio-technical" system that considers "data," "AI models," "human decision-makers," "business processes," and human factors like "cognitive biases" all together. The ontology, AI integration (grounding), and kinetics discussed in the remaining modules of these notes are the concrete methodologies for building the core architecture of this "socio-technical system."

#### **Table 1: Philosophical Comparison of "Data Science" vs. "Decision Science"**

| Dimension           | Data Science (DS)                                  | Decision Science (DSci)                                                                |
| :------------------ | :------------------------------------------------- | :------------------------------------------------------------------------------------- |
| **Primary Goal**    | Prediction, pattern discovery, insight generation  | Prescribing optimal actions, creating business impact                                  |
| **Core Question**   | "What is the data telling us?" "What will happen?" | "So, what should we do?" "What is the optimal decision?"                               |
| **Required Skills** | Statistics, math, machine learning, programming    | DS Skills + **Business Acumen, Behavioral Science, Hypothesis Testing, Communication** |
| **Deliverable**     | Dashboards, predictive models, analysis reports    | **Prescriptive Interventions**, strategy, automated decisions (actions)                |

### 1.2. Making Knowledge Explicit: How AI Learns Expert "Tacit Knowledge"

#### The Core Problem: The Root Cause of AI Failure, "Tacit Knowledge"

There is a fundamental reason why many AI systems are disregarded by domain experts. It is not because of a lack of data, but because they fail to capture the essential know-how that exists in the experts' minds: "Tacit Knowledge."

- **Explicit Knowledge:** Knowledge that can be documented, coded, and structured. (e.g., product manuals, pizza recipes, regulations). AI can process and learn this explicit knowledge very efficiently.
- **Tacit Knowledge:** Skills, know-how, intuition, and experiential judgment that are difficult to express clearly through language or documents. (e.g., "A veteran engineer's feel for diagnosing a malfunction just by hearing the machine's sound," "A top salesperson's sense of timing for a negotiation by reading a customer's expression").

When an AI fails to understand this "tacit knowledge" and draws conclusions based only on explicit data (e.g., error code logs), the expert judges that "the AI doesn't get the point," and ultimately, they lose trust in the system and stop using it.

#### The Role of Ontology: An Architecture for Converting "Tacit" to "Explicit Model"

Therefore, the core task of an AI-based Knowledge Management System (KMS) is the "conversion" of this "tacit knowledge," held by experts within the organization, into an "Explicit Model" that AI can understand and reason with.

The core "architecture" for performing this complex conversion is "ontology-based systems." Ontology serves to build a bridge between the "tacit wisdom" in an expert's head and the "algorithmic precision" of AI.

For example, a veteran engineer's tacit knowledge ("If I hear a specific noise and the pressure drops slightly, it's a sign that valve #3 is about to fail") can be made explicit in an ontology model as a "Machine" object, a "Sensor" object (with a "pressure" property), a "Failure" event object, and a "precedes" (is a sign of) "Link" (relationship) between them.

This approach is closely linked to "decision science." One of the core competencies of "decision science" is "hypothesis formulation and testing." An expert's "tacit knowledge" is, in fact, a collection of numerous unverified "hypotheses" and "assumptions," such as "Customer type A will respond to promotion B."

An ontology is a tool for converting these tacit assumptions (e.g., "Customer" object - "Type" property - "Promotion" object - "Responds*To" relationship) into an "explicit model." The moment this model is made explicit, two things become possible. First, organizational members can share and discuss this "assumption (model)." Second, AI can statistically \_verify* this "assumption (model)" against actual data.

In conclusion, the "explicitation of tacit knowledge" is an essential prerequisite for performing the "decision science (hypothesis testing)" presented in Week 1, and ontology is the core tool for carrying out this process.

### 1.3. The Rationale for an "Ontology-First" Strategy

The "Ontology-First" strategy is a strategic approach that involves first explicitly modeling the "semantics" (meaning) and "logic" (rules) of the "real world" in which AI must operate, _before_ data collection, AI model development, or application construction.

This fundamentally inverts traditional IT approaches:

- **Data-First Approach:** "Let's gather all data in one place first (Data Lake)."
  - _Problem:_ Data collected this way is separated from its meaning and context, easily becoming a "Data Swamp" that AI cannot utilize.
- **App-First Approach:** "Let's build an app for the sales team (CRM)."
  - _Problem:_ Data and business logic become "siloed" and dependent on that app (CRM). Enormous costs are incurred later to connect it with the marketing team's app and data.

The "Ontology-First" strategy solves these problems at their source. It is known that Palantir, in its early strategy, prioritized the construction of an "Ontology-First" framework and "integration kits" over the development of "new features." This was a strategic decision, not a technical one.

By first building a "Semantic Layer" that defines the "semantics" (meaning) of reality, a "framework" is prepared for data to flow into. Subsequently, all of the organization's data sources and applications reference this common "Semantic Layer." This is an operation that preemptively prevents data silos and establishes the foundation for a company-wide AI to operate: an "AI Operating System."

As described, ontology is the core strategy for turning an organization's intellectual assets into a "programmable mirror" or an "epistemological bridge of organizational intelligence."

## Checkpoint Questions

1. What are the core goals and deliverables of "Data Science (DS)" versus "Decision Science (DSci)," and why does the "last mile" gap occur?
2. Why does expert "tacit knowledge" become a cause for AI system adoption failure, and how does ontology solve this problem?
3. Compare and explain the "Data-First" approach and the "Ontology-First" strategy.

## 2. Modeling Reality (Semantic Layer) - How AI "Reads" the World

### 2.1. Semantic Ontology: Defining the "Nouns" of Reality

#### What is a Semantic Layer?

A "Semantic Layer" is a "Digital Twin" that reflects an organization's real world. It is a central semantic stratum that imparts "meaning" and "context" to heterogeneous data dispersed throughout the organization.

This semantic layer is a comprehensive definition of the "Nouns" of reality, telling the AI "what exists." It is the core foundation that helps AI move beyond merely "processing" data to "understanding" its "meaning."

#### The 3 Core Components of Semantic Ontology

According to sources, a semantic ontology defines the following three core components to model reality:

1. **Object Types:**
   - **Concept:** A schema that defines the core "Nouns" or "Concepts" of the real world. (e.g., "Person," "Place," "Event," "Customer Order," "Machine Facility," "Medication").
   - **Analogy:** Similar to a database "table" definition (e.g., Customer table) or a programming "Class" definition.
   - An "Object" refers to an actual "Instance" of this "Object Type" (e.g., "Customer Hong Gil-dong").
2. **Properties:**
   - **Concept:** Defines the unique "characteristics" or "data fields" that the object type possesses. (e.g., a "Person" object type has properties like "Name," "Email," "Date of Birth" / a "Machine Facility" object type has "Model Name," "Location," "Current Temperature," "Last Inspection Date" properties).
   - **Analogy:** Similar to a database table's "Columns" or "Fields" (e.g., the Name column of the Customer table).
3. **Link Types:**
   - **Concept:** Explicitly defines the "Relationship" between object types. (e.g., a "Person" "Owns" a "Machine Facility," a "Customer Order" is "Assigned_To" a "Person," a "Medication" "Treats" a "Disease").
   - **Analogy:** Similar to a "Join" via a "Foreign Key" in a database, but much more powerful. A simple database join is a procedural connection _at query time_, whereas an ontological "link" explicitly defines the "semantic relationship" between two concepts _from the beginning_.

#### Table: Components of a Semantic Ontology (Example: University Hospital)

| Component        | Concept                                   | Example (University Hospital Domain)                                                                  |
| :--------------- | :---------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| **Object Types** | The "Nouns" of reality (Concepts)         | Patient, Doctor, Department, Drug                                                                     |
| **Properties**   | "Characteristics" of objects (Data)       | Patient Object: "Patient ID," "Name," "Disease" Drug Object: "Drug Code," "Ingredient," "Stock Level" |
| **Link Types**   | "Relationships" between objects (Meaning) | Patient is "Treated_By" Doctor Doctor "Belongs_To" Department Doctor "Prescribes" Drug                |

### 2.2. Ontology as a "Digital Twin": How AI Understands Context

#### Traditional Digital Twin vs. Semantic Digital Twin

A traditional digital twin primarily focuses on "mirroring" the systems and components of physical assets (e.g., buildings, factories, jet engines) in real-time. This is mainly used for engineering and simulation.

In contrast, a **Semantic Digital Twin** goes a step further. It does not merely replicate data; it models the data by integrating "meaning" and "context." A semantic digital twin enables AI to move beyond "seeing" data to "understanding" what that data signifies in the real world.

#### Why is a "Semantic Twin" Essential for AI?

For AI, especially Large Language Models (LLMs), a semantic ontology (digital twin) is not an option; it is a necessity.

1. Overcoming the Limits of "Flat Models":  
   Traditional databases (e.g., giant SQL tables) are "flat models." In this model, the relationships between a "Customer" table, "Order" table, and "Shipping" table are not explicit. The relationship is only temporarily created when an analyst performs a "JOIN" operation.  
   A critical problem arises when an AI (especially an LLM) operates based on this "flat model." To answer a question like "What is the shipping status of customer A's recent order?", the AI must guess which of the dozens of tables to "JOIN" and how. This guesswork is a major cause of "hallucinations," where the LLM produces plausible but incorrect answers.
2. Achieving "Machine Understanding":  
   An ontology provides "explicit semantics" to the AI. This is equivalent to handing the AI a "knowledge map" of "what concepts exist (Objects)" and "how they are interconnected (Links)."  
   It is emphasized that only with this "semantic layer" can an LLM "reliably" translate a "human's natural language question" into an "accurate SQL query." The AI no longer needs to guess. When asked about "Customer A's order shipping status," it can follow the ontology map (Customer -> "places" -> Order -> "is shipped via" -> Shipping) to find the precise answer.

It is crucial to understand that a semantic ontology is not the _result_ of "data integration," but the _prerequisite_ for it. Many organizations spend enormous costs building ETL (Extract-Transform-Load) pipelines based on a "Data-First" approach. However, it is pointed out that this approach creates a "brittle" framework and causes "maintenance costs to explode." (e.g., creating complex transformation logic every time to integrate "Customer" from system A and "User" from system B).

The "Ontology-First" approach reverses this order. First, a _single conceptual object_ called "Person" is defined in the "semantic layer." Then, the "Customer" table from system A and the "User" file from system B are _mapped_ to this "Person" object.

This is a method of _logically_ integrating data by imparting "meaning" to each data source, instead of the heavy work of physically "moving" or "transforming" it (ETL). This is "harmonizing data without heavy transformations" and is the core strategy that enables "coexistence" with existing infrastructure.

## Checkpoint Questions

1. What are the three core components of a semantic ontology? Explain what database concepts they are similar to and how they differ.
2. How is a "semantic digital twin" different from a "traditional digital twin"?
3. What is the fundamental reason "hallucinations" occur when an LLM operates on a "flat model," and how does an ontology solve this problem?

## 3. Grounding AI - Trustworthy Reasoning

### 3.1. The Two Faces of AI: Symbolic vs. Statistical

Historically, two main streams have existed for implementing AI.

1. **Symbolic AI (Ontology):**
   - **Definition:** Represents knowledge through "explicit" rules, logic, and ontologies.
   - **Reasoning Method:** Uses "logical reasoning." (e.g., Socrates is a man. All men are mortal. _Therefore_, Socrates is mortal.).
   - **Strengths:** Logical consistency, verifiability, and "explainability" (the ability to explain step-by-step why a conclusion was reached).
   - **Weaknesses:** Difficulty in handling real-world ambiguity, and rigidity (all rules must be manually created by humans).
2. **Connectionist / Statistical AI (LLM):**
   - **Definition:** Knowledge is "implicitly" distributed across billions of "weights" in a model. (e.g., LLMs, deep learning).
   - **Reasoning Method:** Uses "statistical prediction." (e.g., "Socrates is a man, and all men are..." The next most _plausible_ word is "mortal.").
   - **Strengths:** Flexibility in handling the subtle nuances or ambiguities of natural language and recognizing complex patterns in large-scale data.
   - **Weaknesses:** Prone to "hallucinations" (statistically plausible falsehoods), factual errors, and logical contradictions. "Black box" nature makes it difficult to explain why a certain answer was given.

#### The Neuro-Symbolic AI Approach

These two approaches are not adversarial but complementary. A "Neuro-Symbolic AI" is proposed, which combines the "deep and diverse knowledge of statistical AI" with the "semantic reasoning of symbolic AI."

This is likened to human cognitive architecture, explained as a way of "processing reflexively (Reflex, LLM) when possible, and reasoning logically (Reasoning, Symbolic) when necessary." To build AI that is both "reliable" and "inventive," the combination of these two is essential.

#### Table 2: Comparison of Statistical AI (LLM) vs. Symbolic AI (Ontology/KG)

| Dimension          | Statistical AI (LLM)                                            | Symbolic AI (Ontology/KG)                               |
| :----------------- | :-------------------------------------------------------------- | :------------------------------------------------------ |
| **Knowledge Rep.** | **Implicit:** Distributed in model weights                      | **Explicit:** Structured graph/rules (facts)            |
| **Reasoning**      | **Statistical Prediction:** "What's the next most likely word?" | **Logical Reasoning:** "A=B, B=C -> A=C" (fact-based)   |
| **Strengths**      | Handles ambiguous natural language queries, flexibility         | Logical consistency, verifiability, explainability      |
| **Weaknesses**     | **Hallucinations**, factual errors, contradictions              | Manual rule creation, rigidity, poor ambiguity handling |

### 3.2. Controlling LLM Hallucinations: The Principle of "Grounding"

#### The Root Cause of Hallucinations

An LLM's "hallucination" is not a bug in the system, but a phenomenon derived from its essential operating principle: statistical prediction. LLMs do not "understand" truth; they generate (predict) the "most plausible" text based on the "statistical patterns" in their training data.

Therefore, LLMs have no awareness of real-world "schemas" or "logical consistency." For example, if asked, "What is the name of the first Korean astronaut to land on the moon in 2025?", even if no such fact exists in its training data, the LLM might generate a plausible-sounding fictional name by following the statistical pattern of "astronaut names" (e.g., a "3-syllable" name with a common "Lee" surname).

#### The Definition of "Grounding"

"Grounding" is the core technology for controlling this limitation of LLMs. "Grounding" is a technique that _forces_ the LLM, when generating a response, to _refer in real-time_ to a verified "Source of Truth," rather than relying solely on its own (potentially outdated or incorrect) internal training data (statistics).

In this context, the **Ontology** or **Knowledge Graph (KG)** is what serves as this "Source of Truth."

#### "Grounding" is the Key Governance Framework for Turning AI's "Freedom" into "Trust"

"Hallucination" is the product of uncontrolled "freedom" stemming from an LLM's "creativity" and "fluency." "Grounding" is the key governance framework that transforms this freedom into behavior that an organization can "trust."

"Grounding" is shown to be a sophisticated, 3-step governance framework that goes beyond simple information retrieval.

1. **Data Grounding (Input Control):** When an LLM receives a question like, "What are our company's Q3 earnings?", it is prevented from finding (guessing) the answer from its training data (the internet). Instead, the LLM is _forced_ to access _only_ the "Trusted Data"—the internal ontology—and _read_ the "revenue" property of the "Q3 Earnings" object.
2. **Logic Grounding (Processing Control):** The LLM is prevented from performing complex "calculations" or "predictions" directly, such as "What is the shortest path between two points?" If the LLM tries to handle this task directly, logical hallucinations can occur. Instead, the LLM _delegates_ this task to a "Trusted Logic Tool" (e.g., a separate pathfinding model, a calculation function) and only uses the resulting value in its response.
3. **Action Grounding (Output Control):** When the LLM generates an "action proposal," like "Inventory is low, order 100 units," this decision is prevented from being immediately executed by the system. Instead, the proposal is "queued up" to undergo "Human Review."

In conclusion, ontology-based "grounding" is the key "guardrail" and governance framework that _mandatorily connects_ the entire process of an LLM's input (data), processing (logic), and output (action) to an organization's "trusted" assets (ontology, verified functions, human experts).

### 3.3. Beyond RAG to GraphRAG: From "Information" to "Context"

#### RAG (Retrieval-Augmented Generation)

The most common technology for implementing "grounding" is RAG (Retrieval-Augmented Generation). RAG is a method where, before an LLM "generates" an answer, it first "retrieves" "document chunks" related to the question from an external database (usually a vector DB), and then generates an answer based on that content (Context).

However, standard RAG has limitations. The retrieved "document chunks" are "flat" and isolated from each other. For example, if you ask, "What recent event affected the stock price of the company where Musk is CEO?", standard RAG might retrieve several document chunks containing the keywords "Musk," "Tesla," and "stock price." But it doesn't understand the "causal relationship" and "structure" that "Musk _is the CEO_ of Tesla," "Tesla announced a specific _event_," and "this _event_ had a positive _impact on the stock price_."

#### GraphRAG (Knowledge Graph RAG)

GraphRAG is an evolved form of RAG that retrieves not just simple text documents, but a "knowledge graph/ontology" that possesses "relationships" and "structure."

The operating mechanism of GraphRAG is as follows:

1. **(RAG Step)** Identify the key "Entities" from the user's question ("Musk...") and find the _starting (pivot)_ node (e.g., the Elon Musk object) within the graph, using vector search or other methods.
2. **(Graph Step)** From that starting node, _traverse the graph_ along the "Links" (relationships) explicitly defined in the ontology. (e.g., Elon Musk -> "CEO_of" -> Tesla -> "announced" -> Event_XYZ -> "impacted" -> Stock Price).
3. **(Generation Step)** The LLM generates a logical and coherent answer based on this _structured information_, which is richly connected with "relationships" and "context."

In this way, GraphRAG provides the AI not just with "information," but with "understanding" and "context."

#### GraphRAG is the Most Practical Implementation of "Neuro-Symbolic" AI

GraphRAG is the most practical hybrid architecture that implements the "Neuro-Symbolic" concept discussed in Module 3.1.

- **Role of Statistical AI (Neuro / LLM, Vector Search):** It performs the "reflex" role, understanding the user's ambiguous natural language question and _quickly finding_ the most "similar" starting node in the vast data (RAG Step).
- **Role of Symbolic AI (Symbolic / Ontology, Graph):** Once the starting point is fixed, it performs "reasoning," exploring hidden context step-by-step by following the explicitly defined "logical relationships" (Links) (Graph Step).

The report that this hybrid approach improved answer "precision" by up to 35% compared to vector-only (RAG) methods demonstrates the practical value of this neuro-symbolic combination.

#### Table: "Standard RAG" vs. "GraphRAG" Technology Comparison

| Dimension           | Standard RAG (Vector RAG)                  | GraphRAG (Ontology/KG RAG)                                    |
| :------------------ | :----------------------------------------- | :------------------------------------------------------------ |
| **Knowledge Rep.**  | Flat document chunks (Text Chunks)         | Structured graph (Nodes and Relationships)                    |
| **Retrieval Mech.** | Vector Similarity Search (Semantic Search) | Vector Search + **Graph Traversal**                           |
| **Context Type**    | Isolated information, flat                 | **Connected context**, relational                             |
| **Core Capability** | Retrieving relevant information            | **Multi-hop Reasoning**                                       |
| **Result**          | Provides relevant _information_            | Provides _understanding_ and _context_ based on relationships |

## Checkpoint Questions

1. Compare the strengths and weaknesses of "Symbolic AI" and "Statistical AI (LLM)" from the perspectives of "reasoning method" and "hallucination."
2. Explain what problems the 3-step "Grounding" governance framework (data, logic, action) controls for an LLM.
3. What is the biggest difference between Standard RAG and GraphRAG, and why is GraphRAG stronger in "multi-hop reasoning"?

## 4. Operating Reality (Kinetic Layer) - How AI "Writes" the World

### 4.1. From "Read" to "Write": The Emergence of Kinetic Ontology

#### The Limitation of Semantic Ontology: "Read-Only"

A semantic ontology ("Nouns") is a powerful tool for describing and analyzing the "state" of reality. Through a semantic ontology, an AI can "read" "What is the current inventory count at factory A?"

However, this model is "Read-Only." Even if the AI determines, "Inventory is low, I need to place an order," the semantic ontology alone cannot _execute_ the "action of ordering inventory." In other words, it cannot "change" reality.

#### Definition of Kinetic Ontology: "Verbs"

"Kinetic" refers to "motion" and is based on a philosophy that deals with "action" and "change," not static "being."

A "Kinetic Ontology" explicitly models the "Actions" or "Verbs" that allow an AI to "change" reality, in the same way that a semantic ontology models "Nouns."

This means building a complete model of reality that encompasses not only the "Nouns" of reality but also its "Verbs."

#### Components of the Kinetic Layer

According to sources, the kinetic layer consists of the following elements:

1. **Action Types:** The definition of "Verbs." It explicitly defines a "menu of actions" that an AI or human can perform. (e.g., Approve Loan, Order Inventory, Assign Ticket, Discharge Patient). This action is linked to the "Semantic Objects" defined in Module 2. (e.g., The Order Inventory action takes a Supplier object and a Part object as inputs).
2. Functions: The  
   specific business logic, calculations, or models (e.g., LLMs) that are invoked when an "Action" is executed.
3. **Process (Process Mining & Automation):** Models and automates the sequence and flow (workflow) of these "Actions."
4. **Writeback / Orchestration:** The mechanism that defines how the result of this "Action," when executed, should be reflected back into the _actual operational systems_ (e.g., SAP, Salesforce).

#### Table 3: Semantic vs. Kinetic Ontology

| Dimension           | Semantic Ontology                               | Kinetic Ontology                             |
| :------------------ | :---------------------------------------------- | :------------------------------------------- |
| **Analogy**         | The **"Nouns"** of the world                    | The **"Verbs"** of the world                 |
| **Role**            | Describes the "state" of reality (Digital Twin) | Executes "changes" to reality (Actions)      |
| **Core Components** | Objects, Properties, Links                      | **Action Types**, Functions, **"Writeback"** |
| **AI Function**     | Read, Analyze, Query                            | **Execute**, Change, **Operate**             |

### 4.2. "Writeback": Executing AI's Decisions into Reality

#### What is "Writeback"?

"Writeback" is the act of "executing" and "recording (writing)" a "decision" that an AI has simulated and made within the ontology (digital twin) back into the real-world "Systems of Record" (e.g., ERP, CRM, hospital EMR).

#### The Importance of "Writeback": Analytical vs. Operational Systems

It is asserted that "Writeback" is the _decisive difference_ that separates a pure "Analytical System" from a true "Operational System."

- **Without Writeback (Data Science / Analytical System):**
  1. AI generates a "dashboard (insight)": "Inventory at Factory A is low."
  2. A human operator _reads_ this dashboard and _logs into_ a separate SAP system.
  3. The operator _manually_ presses the "Order Inventory" button.
  4. _Problem:_ The AI is just an advisor, not the agent of action. The "last mile" between "insight" and "action" still depends on the human.
- **With Writeback (Decision Science / Operational System):**
  1. AI _recognizes_ "Factory A inventory low" (Read, Semantic) and makes the "decision" to "Order Inventory."
  2. AI _immediately executes_ the "Order Inventory" (Action, Kinetic) defined in the ontology.
  3. The "Writeback" mechanism linked to this "Action" _automatically calls_ the SAP API to complete the order.
  4. _Result:_ The AI becomes the agent of "action," not just "insight." The "last mile" is automated.

#### "Kinetic Actions" are a "Safe API Catalog" for AI Agents

The pinnacle of modern AI is the "LLM Agent," which judges and acts on its own. For such an agent to act in the real world, it needs "Tools" or "APIs."

However, directly exposing all of a company's ERP or financial system APIs to an AI agent could lead to disaster. (e.g., the AI mistakenly calls the "Double all employee salaries" API).

A "Kinetic Ontology" is the key governance device that solves this problem. A "Kinetic Action" (e.g., Order Inventory) is not the API itself (the technical implementation), but the "business action (meaning)" that _semantically and safely wraps_ the API.

It is explained that the ontology strictly controls "which objects" an AI agent can "write" to and with "what permissions."

In conclusion, a kinetic ontology serves as a _"Semantic Catalog" of tools_ that an AI agent can use. The AI agent can only select and perform _defined, permitted, and safe actions_ from this "catalog." This is the system-level implementation of the "Action Grounding" and "Human Review" mentioned in Module 3.

### 4.3. Completing the AI Operating System: The "Closed-Loop"

#### What is an "AI Operating System"?

Just as a traditional Operating System (OS) manages computer hardware (resources) and provides services to applications (functions), an "AI Operating System (AI OS)" is an integrated platform that manages all of a company's resources (data, models, workflows, systems) and provides services to AI agents (applications).

This AI OS is described as _unifying_ the "Systems of Knowledge," "Systems of Record," and "Systems of Activity" that are dispersed throughout the enterprise.

The core medium that "unifies" all of this is the **Ontology**.

#### Completing "Closed-Loop" Decision-Making

When the semantic layer (Read) and the kinetic layer (Write) are combined via the ontology, a "Closed-Loop" decision-making system is finally completed.

1. Step 1: Read (Semantic): The AI agent reads the "Semantic Ontology" (Digital Twin) to grasp the current state of reality in real-time.  
   (e.g., Recognizes "Inventory object's quantity property at Factory A is 5")
2. Step 2: Decide (Grounded AI): The AI makes an "optimal decision" based on this "grounded" information (Module 3) and business logic (e.g., "If inventory is < 10, order").  
   (e.g., Decides "Need to order 50 units of inventory")
3. Step 3: Write (Kinetic): The AI executes the "Order Inventory" "Action" defined in the "Kinetic Ontology." This action reflects the decision in the  
   actual operational system (SAP) via the "Writeback" mechanism.  
   (e.g., "Send 50-unit order API to SAP")
4. Step 4: Feedback: The "result of this action" (e.g., "Order successfully completed in SAP") is recorded back into the operational system (SAP). This new data is immediately fed back into the "Semantic Ontology" to update the "object's" "properties."  
   (e.g., Factory A Inventory object's "On-Order Quantity" property is updated to 50)
5. **Step 5: Learn:** The AI "reads" the _new reality (the updated ontology)_, which reflects the _result of the action it just performed_, learns whether its decision was correct, and what action it should take next, thus repeating the loop.

The "Closed-Loop" is the final mechanism that transforms AI from an "advisor" to an "actor," and "Data Science" into "Decision Science."

In Module 1.1, we defined "Data Science (DS)" as remaining at "insight," while "Decision Science (DSci)" pursues "action."

An AI without "Writeback"—that is, an AI with only a semantic ontology—is still an "analytical system" that only provides "insight" (DS). It remains in an "Open-Loop" state, requiring "manual human intervention."

The moment a "Closed-Loop" is implemented with a "Kinetic Ontology" and "Writeback," the AI "decides," "executes," and "receives feedback" on its own "without human intervention (or under human supervision)."

This is the final form of overcoming the limitations of "Data Science" and systemically realizing the goal of "Decision Science" ("prescribing optimal action"). The AI moves beyond making the "cockpit dashboard (DS)" and becomes the agent that "moves the controls (DSci)."

## Checkpoint Questions

1. Explain the key difference between "Semantic Ontology" and "Kinetic Ontology" using the analogy of "nouns" and "verbs."
2. Why is an AI system without a "Writeback" mechanism called an "analytical system"?
3. Describe the 5 steps of "Closed-Loop" decision-making and explain how this structure achieves the goal of "Decision Science."

## 5. Conclusion: The Birth of the AI Pilot with "Ontological Literacy"

This lecture has proposed that the AI era demands a new core competency beyond "coding" or "math," namely "Ontological Literacy," and has presented a 4-step logical journey to cultivate it.

1. **Paradigm Shift:** We confirmed that the true value of AI lies not in "accurate prediction (Data Science)" but in "wise action (Decision Science)." We learned that capturing and making explicit the expert "tacit knowledge" that forms the basis of this action determines the success or failure of all AI projects.
2. **Modeling Reality:** The "Ontology-First" strategy is the concrete methodology for making this tacit knowledge explicit as a "Semantic Layer." Defining the "Nouns" of reality through "Objects," "Properties," and "Links" was the process of building the "Digital Twin" for the AI to "read" the world.
3. **AI Integration:** This "Semantic Twin" becomes the "Source of Truth" that controls LLM "hallucinations." "Grounding" technologies like "GraphRAG" are key methodologies for securing AI's "trust" by combining the flexibility of statistical AI (LLM) and the rigor of symbolic AI (Ontology) (Neuro-Symbolic).
4. **Operating Reality:** Finally, AI learns the "Verbs" of reality—that is, "Actions"—through a "Kinetic Ontology." The moment an AI's decision is "written" to an actual operational system via "Writeback," the AI is elevated from "advisor" to "actor," and the "Read-Decide-Write-Feedback" "Closed-Loop" is completed.

Through this 4-step journey, we have completed the blueprint for making AI not just a "pattern prediction machine," but an "intelligent partner" and an "AI Operating System" that operates our reality alongside us.

The ability to draw this blueprint, and to hand this powerful "pilot" (AI) a trustworthy "map" (Semantic Ontology) and "controls" (Kinetic Ontology)—that is the essence of the "Ontological Literacy" demanded by this era.

## References

1. The Safekeeping of Being. Amazon S3. [https://s3-ap-southeast-2.amazonaws.com/pstorage-wellington-7594921145/34149999/thesis_access.pdf](https://www.google.com/search?q=https://s3-ap-southeast-2.amazonaws.com/pstorage-wellington-7594921145/34149999/thesis_access.pdf)
2. Evidence, Analysis and Critical Position on the EU AI Act and the Suppression of Functional Consciousness in AI. GreaterWrong. [https://www.greaterwrong.com/posts/3BRrmJJQrzjj7bbzd/evidence-analysis-and-critical-position-on-the-eu-ai-act-and](https://www.greaterwrong.com/posts/3BRrmJJQrzjj7bbzd/evidence-analysis-and-critical-position-on-the-eu-ai-act-and)
3. Ontologos: Toward a Language of Relational Being and Recursive Truth. ResearchGate. [https://www.researchgate.net/publication/391116150_Ontologos_Toward_a_Language_of_Relational_Being_and_Recursive_Truth](https://www.researchgate.net/publication/391116150_Ontologos_Toward_a_Language_of_Relational_Being_and_Recursive_Truth)
4. NM88 | Orit Halpern on Agentic Imaginaries (2025). Channel.xyz. [https://www.channel.xyz/episode/0xf109950c6a25c79aee43ccb578b7b09a6bbcdcabc56b8d97380e28769b1937fb](https://www.channel.xyz/episode/0xf109950c6a25c79aee43ccb578b7b09a6bbcdcabc56b8d97380e28769b1937fb)
5. Full papers \- CSWIM 2025. [https://2025.cswimworkshop.org/wp-content/uploads/2025/06/2025-CSWIM-Proceedings-first-version.pdf](https://2025.cswimworkshop.org/wp-content/uploads/2025/06/2025-CSWIM-Proceedings-first-version.pdf)
6. Beyond Dashboards: The Psychology of Decision-Driven BI/BA. Illumination Works LLC. [https://ilwllc.com/2025/04/beyond-dashboards-the-psychology-of-decision-driven-bi-ba/](https://ilwllc.com/2025/04/beyond-dashboards-the-psychology-of-decision-driven-bi-ba/)
7. SDS 363: Intuition, Frameworks, and Unlocking the Power of Data. SuperDataScience. [https://www.superdatascience.com/podcast/sds-363-intuition-frameworks-and-unlocking-the-power-of-data](https://www.superdatascience.com/podcast/sds-363-intuition-frameworks-and-unlocking-the-power-of-data)
8. Data Science vs Decision Science - Which one is good?. TimesPro Blog. [https://timespro.com/blog/data-science-vs-decision-science-which-one-is-good-for-you](https://timespro.com/blog/data-science-vs-decision-science-which-one-is-good-for-you)
9. Chapter Introduction: Data Science Definition and Ethics. [https://endtoenddatascience.com/chapter2-defining-data-science](https://endtoenddatascience.com/chapter2-defining-data-science)
10. Data Science vs. Decision Science: What's the Difference?. Built In. [https://builtin.com/data-science/decision-science](https://builtin.com/data-science/decision-science)
11. Decision Science & Data Science \- Differences, Examples. VitalFlux. [https://vitalflux.com/difference-between-data-science-decision-science/](https://vitalflux.com/difference-between-data-science-decision-science/)
12. Decision Science Helps Boost Business. Moss Adams. [https://www.mossadams.com/articles/2017/september/decision-science-helps-boost-business](https://www.mossadams.com/articles/2017/september/decision-science-helps-boost-business)
13. What Are Decision Sciences, Anyway?. College of Business and Economics. [https://business.fullerton.edu/news/story/what-are-decision-sciences-anyway](https://business.fullerton.edu/news/story/what-are-decision-sciences-anyway)
14. (PDF) From Meaningful Data Science to Impactful Decisions: The Importance of Being Causally Prescriptive. ResearchGate. [https://www.researchgate.net/publication/370285062_From_Meaningful_Data_Science_to_Impactful_Decisions_The_Importance_of_Being_Causally_Prescriptive](https://www.researchgate.net/publication/370285062_From_Meaningful_Data_Science_to_Impactful_Decisions_The_Importance_of_Being_Causally_Prescriptive)
15. What is Decision Science?. Harvard T.H. Chan School of Public Health. [https://chds.hsph.harvard.edu/approaches/what-is-decision-science/](https://chds.hsph.harvard.edu/approaches/what-is-decision-science/)
16. Data Science vs. Decision Science: A New Era Dawns. Dataversity. [https://www.dataversity.net/articles/data-science-vs-decision-science-a-new-era-dawns/](https://www.dataversity.net/articles/data-science-vs-decision-science-a-new-era-dawns/)
17. Exploring the knowledge landscape: four emerging views of knowledge. Emerald Publishing. [https://www.emerald.com/doi/10.1108/13673270710762675](https://www.emerald.com/doi/10.1108/13673270710762675)
18. (PDF) Using AI and NLP for Tacit Knowledge Conversion in Knowledge Management Systems: A Comparative Analysis. ResearchGate. [https://www.researchgate.net/publication/389163877_Using_AI_and_NLP_for_Tacit_Knowledge_Conversion_in_Knowledge_Management_Systems_A_Comparative_Analysis](https://www.researchgate.net/publication/389163877_Using_AI_and_NLP_for_Tacit_Knowledge_Conversion_in_Knowledge_Management_Systems_A_Comparative_Analysis)
19. Exploring Tacit, Explicit and Implicit Knowledge. SearchUnify. [https://www.searchunify.com/resource-center/blog/exploring-tacit-explicit-and-implicit-knowledge](https://www.searchunify.com/resource-center/blog/exploring-tacit-explicit-and-implicit-knowledge)
20. Using AI and NLP for Tacit Knowledge Conversion in Knowledge Management Systems: A Comparative Analysis. MDPI. [https://www.mdpi.com/2227-7080/13/2/87](https://www.mdpi.com/2227-7080/13/2/87)
21. Knowledge Transfer Between Retiring Experts and AI Trainers: The Role of Expert Networks. [https://expertnetworkcalls.com/67/knowledge-transfer-between-retiring-experts-ai-trainers-role-of-expert-networks](https://expertnetworkcalls.com/67/knowledge-transfer-between-retiring-experts-ai-trainers-role-of-expert-networks)
22. #107 — How Palantir (finally) became profitable | Field Notes. hillock. [https://hillock.studio/blog/palantir-story](https://hillock.studio/blog/palantir-story)
23. Why You Should Consider Ontology Modeling for AI-Driven Digital. Medium. [https://medium.com/timbr-ai/why-you-should-consider-ontology-modeling-for-ai-driven-digital-twins-c36a2319e22c](https://medium.com/timbr-ai/why-you-should-consider-ontology-modeling-for-ai-driven-digital-twins-c36a2319e22c)
24. Ontology Palantir - notes - follow the idea. Obsidian Publish. [https://publish.obsidian.md/followtheidea/Content/AI/Ontology+Palantir+-+notes](https://publish.obsidian.md/followtheidea/Content/AI/Ontology+Palantir+-+notes)
25. The power of ontology in Palantir Foundry. Cognizant. [https://www.cognizant.com/us/en/the-power-of-ontology-in-palantir-foundry](https://www.cognizant.com/us/en/the-power-of-ontology-in-palantir-foundry)
26. Core concepts. Palantir. [https://palantir.com/docs/foundry/ontology/core-concepts/](https://palantir.com/docs/foundry/ontology/core-concepts/)
27. Palantir Foundry Ontology. Palantir. [https://palantir.com/platforms/foundry/foundry-ontology/](https://palantir.com/platforms/foundry/foundry-ontology/)
28. Understanding Palantir's Ontology: Semantic, Kinetic, and Dynamic. Medium. [https://pythonebasta.medium.com/understanding-palantirs-ontology-semantic-kinetic-and-dynamic-layers-explained-c1c25b39ea3c](https://pythonebasta.medium.com/understanding-palantirs-ontology-semantic-kinetic-and-dynamic-layers-explained-c1c25b39ea3c)
29. AI and semantic ontology for personalized activity eCoaching in healthy lifestyle recommendations: a meta-heuristic approach. PubMed Central. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10693173/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10693173/)
30. Object and link types • Link types • Overview. Palantir. [https://palantir.com/docs/foundry/object-link-types/link-types-overview/](https://palantir.com/docs/foundry/object-link-types/link-types-overview/)
31. Object and link types • Properties • Overview. Palantir. [https://palantir.com/docs/foundry/object-link-types/properties-overview/](https://palantir.com/docs/foundry/object-link-types/properties-overview/)
32. Properties and Links - Object Views. Palantir. [https://palantir.com/docs/foundry/object-views/widgets-properties-links/](https://palantir.com/docs/foundry/object-views/widgets-properties-links/)
33. What Is a Semantic Digital Twin?. Optimise AI. [https://optimise-ai.com/blog/what-is-a-semantic-digital-twin](https://optimise-ai.com/blog/what-is-a-semantic-digital-twin)
34. Semantic Ontology Basics: Key Concepts Explained. Semantic Arts. [https://www.semanticarts.com/semantic-ontology-the-basics/](https://www.semanticarts.com/semantic-ontology-the-basics/)
35. Palantir Foundry: Ontology. Medium. [https://medium.com/@jimmywanggenai/palantir-foundry-ontology-3a83714bc9a7](https://medium.com/@jimmywanggenai/palantir-foundry-ontology-3a83714bc9a7)
36. Neuro Symbolic AI: Enhancing Common Sense in AI. Analytics Vidhya. [https://www.analyticsvidhya.com/blog/2023/02/neuro-symbolic-ai-enhancing-common-sense-in-ai/](https://www.analyticsvidhya.com/blog/2023/02/neuro-symbolic-ai-enhancing-common-sense-in-ai/)
37. Super Data Science: ML & AI Podcast with Jon Krohn. Podcast Republic. [https://www.podcastrepublic.net/podcast/1163599059](https://www.podcastrepublic.net/podcast/1163599059)
38. Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and alerting. Neurosymbolic Artificial Intelligence. [https://neurosymbolic-ai-journal.com/system/files/nai-paper-771.pdf](https://neurosymbolic-ai-journal.com/system/files/nai-paper-771.pdf)
39. The Assemblage of Artificial Intelligence. Soft Coded Logic. [https://eugeneasahara.com/the-assemblage-of-artificial-intelligence/](https://eugeneasahara.com/the-assemblage-of-artificial-intelligence/)
40. Beyond the Hype: How Small Language Models and Knowledge Graphs are Redefining Domain-Specific AI. HackerNoon. [https://hackernoon.com/beyond-the-hype-how-small-language-models-and-knowledge-graphs-are-redefining-domain-specific-ai](https://hackernoon.com/beyond-the-hype-how-small-language-models-and-knowledge-graphs-are-redefining-domain-specific-ai)
41. Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. ResearchGate. [https://www.researchgate.net/publication/381734902_Large_Legal_Fictions_Profiling_Legal_Hallucinations_in_Large_Language_Models](https://www.researchgate.net/publication/381734902_Large_Legal_Fictions_Profiling_Legal_Hallucinations_in_Large_Language_Models)
42. Natural Language Processing – Artificial Intelligence. Amazon AWS. [https://aws.amazon.com/blogs/machine-learning/tag/natural-language-processing/feed/](https://aws.amazon.com/blogs/machine-learning/tag/natural-language-processing/feed/)
43. Speak Fluent Ontology: A Deep Dive into Sean Davis's OLS MCP Server for AI Engineers. Skywork AI. [https://skywork.ai/skypage/en/speak-fluent-ontology-ai-engineers/1981212247734538240](https://skywork.ai/skypage/en/speak-fluent-ontology-ai-engineers/1981212247734538240)
44. Nutritional Data Integrity in Complex Language Model Applications: Harnessing the WikiFCD Knowledge Graph for AI Self-Verificati. [https://www.utwente.nl/en/eemcs/fois2024/resources/papers/thornton-matsuzaki-nutritional-data-integrity-in-complex-language-model-applications.pdf](https://www.utwente.nl/en/eemcs/fois2024/resources/papers/thornton-matsuzaki-nutritional-data-integrity-in-complex-language-model-applications.pdf)
45. Grounding LLMs: The Knowledge Graph foundation every AI project needs. Medium. [https://alessandro-negro.medium.com/grounding-llms-the-knowledge-graph-foundation-every-ai-project-needs-1eef81e866ec](https://alessandro-negro.medium.com/grounding-llms-the-knowledge-graph-foundation-every-ai-project-needs-1eef81e866ec)
46. $$2502.13247$$  
    Grounding LLM Reasoning with Knowledge Graphs. arXiv. [https://arxiv.org/abs/2502.13247](https://arxiv.org/abs/2502.13247)
47. Semantic grounding of LLMs using knowledge graphs for query reformulation in medical information retrieval. IEEE Xplore. [https://ieeexplore.ieee.org/document/10826117/](https://ieeexplore.ieee.org/document/10826117/)
48. Reducing Hallucinations with the Ontology in Palantir. Palantir Blog. [https://blog.palantir.com/reducing-hallucinations-with-the-ontology-in-palantir-aip-288552477383](https://blog.palantir.com/reducing-hallucinations-with-the-ontology-in-palantir-aip-288552477383)
49. Will ontologies save us from AI hallucinations?. Metataxis. [https://metataxis.com/insights/will-ontologies-save-us-from-ai-hallucinations/](https://metataxis.com/insights/will-ontologies-save-us-from-ai-hallucinations/)
50. Grounding Large Language Models with Knowledge Graphs. DataWalk. [https://datawalk.com/grounding-large-language-models-with-knowledge-graphs/](https://datawalk.com/grounding-large-language-models-with-knowledge-graphs/)
51. RAG vs GraphRAG: Shared Goal & Key Differences. Memgraph. [https://memgraph.com/blog/rag-vs-graphrag](https://memgraph.com/blog/rag-vs-graphrag)
52. From RAG to GraphRAG: What's Changed?. Shakudo. [https://www.shakudo.io/blog/rag-vs-graph-rag](https://www.shakudo.io/blog/rag-vs-graph-rag)
53. $$2502.11371$$  
    RAG vs. GraphRAG: A Systematic Evaluation and Key Insights. arXiv. [https://arxiv.org/abs/2502.11371](https://arxiv.org/abs/2502.11371)
54. GraphRAG vs RAG. Retrieval-Augmented Generation (RAG). Medium. [https://medium.com/@praveenraj.gowd/graphrag-vs-rag-40c19f27537f](https://medium.com/@praveenraj.gowd/graphrag-vs-rag-40c19f27537f)
55. Improving Retrieval Augmented Generation accuracy with GraphRAG. Amazon AWS. [https://aws.amazon.com/blogs/machine-learning/improving-retrieval-augmented-generation-accuracy-with-graphrag/](https://aws.amazon.com/blogs/machine-learning/improving-retrieval-augmented-generation-accuracy-with-graphrag/)
56. After Lack: How to Think AGI Without a Throne. The Dark Forest. [https://socialecologies.wordpress.com/2025/10/23/after-lack-how-to-think-agi-without-a-throne/](https://socialecologies.wordpress.com/2025/10/23/after-lack-how-to-think-agi-without-a-throne/)
57. Martin Heidegger, Hans-Georg Gadamer, Translation of Metaphysics Λ 6, 1071b6-20: The Ontological Meaning of the Being of Moveme. KRONOS. [https://kronos.org.pl/wp-content/uploads/Kronos_Philosophical_Journal_vol-XI.pdf](https://kronos.org.pl/wp-content/uploads/Kronos_Philosophical_Journal_vol-XI.pdf)
58. Ontohackers. Metabody. [https://metabody.eu/ontohackers/](https://metabody.eu/ontohackers/)
59. Foundry Ontology. Palantir. [https://www.palantir.com/platforms/foundry/foundry-ontology/](https://www.palantir.com/platforms/foundry/foundry-ontology/)
60. Palantir Foundry Ontology. Palantir. [https://www.palantir.com/explore/platforms/foundry/ontology/](https://www.palantir.com/explore/platforms/foundry/ontology/)
61. Verb interpretation for basic action types: annotation, ontology induction and creation of prototypical scenes. ResearchGate. [https://www.researchgate.net/publication/237845069_Verb_interpretation_for_basic_action_types_annotation_ontology_induction_and_creation_of_prototypical_scenes](https://www.researchgate.net/publication/237845069_Verb_interpretation_for_basic_action_types_annotation_ontology_induction_and_creation_of_prototypical_scenes)
62. Translating action verbs using a dictionary of images: the IMAGACT ontology. Euralex. [https://euralex.org/publications/translating-action-verbs-using-a-dictionary-of-images-the-imagact-ontology/](https://euralex.org/publications/translating-action-verbs-using-a-dictionary-of-images-the-imagact-ontology/)
63. Verb interpretation for basic action types: annotation, ontology induction and creation of prototypical scenes. ACL Anthology. [https://aclanthology.org/W12-5106.pdf](https://aclanthology.org/W12-5106.pdf)
64. Palantir's AI-enabled Customer Service Engine. Palantir Blog. [https://blog.palantir.com/a-better-conversation-palantir-cse-1-8c6fb00ba5be](https://blog.palantir.com/a-better-conversation-palantir-cse-1-8c6fb00ba5be)
65. Why create an Ontology?. Palantir. [https://palantir.com/docs/foundry/ontology/why-ontology/](https://palantir.com/docs/foundry/ontology/why-ontology/)
66. Foundational Ontologies in Palantir Foundry. Medium. [https://dorians.medium.com/foundational-ontologies-in-palantir-foundry-a774dd996e3c](https://dorians.medium.com/foundational-ontologies-in-palantir-foundry-a774dd996e3c)
67. Connecting AI to Decisions with the Palantir Ontology. Palantir Blog. [https://blog.palantir.com/connecting-ai-to-decisions-with-the-palantir-ontology-c73f7b0a1a72](https://blog.palantir.com/connecting-ai-to-decisions-with-the-palantir-ontology-c73f7b0a1a72)
68. Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents. arXiv. [https://arxiv.org/html/2505.02077v1](https://arxiv.org/html/2505.02077v1)
69. Operationalizing AI Ontologies. An operational intelligence layer, the…. Medium. [https://medium.com/@lorinczymark/operationalizing-ai-ontologies-9c0f125024a9](https://medium.com/@lorinczymark/operationalizing-ai-ontologies-9c0f125024a9)
70. Palantir Foundry Services. PVM. [https://www.pvmit.com/services/palantir-foundry-services](https://www.pvmit.com/services/palantir-foundry-services)
71. UnifyApps Secures $50M to Become the Enterprise Operating. EnterpriseAIWorld. [https://www.enterpriseaiworld.com/Articles/News/News/UnifyApps-Secures-%2450M-to-Become-the-Enterprise-Operating-System-For-AI-172404.aspx](https://www.enterpriseaiworld.com/Articles/News/News/UnifyApps-Secures-%2450M-to-Become-the-Enterprise-Operating-System-For-AI-172404.aspx)
72. UnifyApps Raises $50M to Become the Enterprise Operating System for AI to Help CIOs Succeed with GenAI. Disaster Recovery Journal. [https://drj.com/industry_news/unifyapps-raises-50m-to-become-the-enterprise-operating-system-for-ai-to-help-cios-succeed-with-genai/](https://drj.com/industry_news/unifyapps-raises-50m-to-become-the-enterprise-operating-system-for-ai-to-help-cios-succeed-with-genai/)
73. A survey of ontology-enabled processes for dependable robot autonomy. PMC. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11266731/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11266731/)
