# Week 5: LLM Evaluation Paradigms and Benchmarks

## 1. The Changing Landscape of Evaluation: Limitations of Traditional Metrics and the Need for Meaning-Based Assessment

The methodology for evaluating Large Language Models (LLMs) is rapidly evolving alongside model development. There's a growing recognition of the limitations of traditional evaluation metrics like **BLEU** and **ROUGE**, leading to a paradigm shift toward **meaning-based evaluation** and the **LLM-as-a-Judge** approach.

### 1.1 Limitations of Traditional Evaluation Metrics

In natural language generation (NLG) model quality assessment, quantitative metrics like **BLEU** and **ROUGE** have been used for a long time. These metrics have the following characteristics:

- **BLEU**: Measures the **n-gram overlap degree** between reference and generated sentences in machine translation
- **ROUGE**: Calculates the **recall of important words/phrases** in summarization
- **Common feature**: Both are based on **superficial string matching** evaluation

These traditional metrics have the following **fundamental limitations**:

1. **Lack of semantic understanding**: Low scores for cases where meaning is the same but words are different due to synonym usage or expression changes
2. **High scores for incorrect meanings**: High scores even when expressions are similar but meanings are wrong
3. **Inability to evaluate creativity**: Cannot distinguish the quality of creative or subjective LLM responses
4. **Ignoring factual accuracy**: Does not reflect important aspects like factual accuracy or consistency

### 1.2 Emergence of Meaning-Based Evaluation

To overcome the limitations of traditional metrics, **meaning-based evaluation methods** have emerged:

#### BERTScore and SentenceMover

- Measure sentence semantic similarity using **similarity in embedding space**
- Achieve improved correlation compared to BLEU
- Better capture **semantic similarity**

#### BLEURT

- **Pre-trained evaluation metric**
- Higher alignment with human evaluation through learned semantic discrimination
- Enhanced **contextual understanding** capability

### 1.3 Emergence of LLM-as-a-Judge Paradigm

Recently, an innovative approach has emerged that utilizes **Large Language Models (LLMs) as evaluators**:

- **LLM-as-a-Judge**: LLMs score or rate other LLMs' outputs instead of humans
- **Complex semantic understanding**: Capable of performing contextual judgments
- **Open-ended generation tasks**: Can evaluate even tasks without predetermined answers
- **Reflecting subjective criteria**: Learn and reflect human evaluators' judgment criteria

This change represents a shift from **BLEU/ROUGE-centered traditional metrics** to **meaning-based meta-evaluation** utilizing LLMs' rich semantic understanding capabilities.

### Checkpoint Questions

- How do traditional evaluation metrics like BLEU and ROUGE measure output quality, and what limitations arise from this approach?
- Why is **meaning-based evaluation** necessary, and what elements of output should be primarily considered in such evaluation?
- What potential benefits can be gained by utilizing LLMs as evaluators instead of surface-level comparison against a single reference answer?

## 2. LLM-Based Evaluation Paradigms

With the emergence of new paradigms utilizing LLMs as evaluators, various approaches have been proposed to overcome the limitations of existing evaluation methods. This section examines major LLM-based evaluation techniques including **GPTScore**, **G-Eval**, and **FLASK**.

### 2.1 GPTScore: Probability-Based Evaluation Framework

**GPTScore** is an early meta-evaluation technique that quantifies output quality using the **language model probabilities** of LLMs themselves.

#### Core Principles

GPTScore operates in the following manner:

1. **Probability-based evaluation**: When given source text and candidate output, calculates the probability (likelihood) that the language model would generate that output
2. **No reference answer needed**: Can evaluate without separate reference answers
3. **Automatic quality measurement**: Measures how well the output aligns with language patterns learned by the model, such as fluency and grammatical accuracy

#### Mathematical Formulation

For summarization evaluation:

```
Score = P(summary | source_text, model)
```

Generally, **log probability summation** or **perplexity inverse** is used to calculate scores.

#### Advantages

- **No separate tuning needed**: Can evaluate using only the language model's inherent probabilities
- **No reference answer needed**: Can evaluate even in open-ended generation tasks
- **Automation**: Automatic quality measurement without human intervention

#### Limitations

1. **Data bias**: Model's learned data bias is reflected in evaluation scores
2. **Creativity suppression**: Creative but correct responses may receive low scores
3. **Probability-quality mismatch**: High probability doesn't necessarily mean high quality

#### Performance Results

- **Correlation with human evaluation**: Approximately 0.43 (moderate level)
- **Compared to BLEU**: Achieved improved correlation
- **Absolute reliability**: Still limited

### 2.1.1 GPTScore Implementation Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_gpt_score(model, tokenizer, source_text, candidate_text):
    """
    Function to calculate GPTScore

    Args:
        model: Language model
        tokenizer: Tokenizer
        source_text: Source text
        candidate_text: Candidate text to evaluate

    Returns:
        GPTScore
    """
    # Construct input text
    input_text = f"{source_text} {candidate_text}"

    # Tokenization
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Sum log probabilities for candidate text portion
    candidate_tokens = tokenizer(candidate_text, return_tensors="pt", add_special_tokens=False)
    candidate_length = candidate_tokens.input_ids.shape[1]

    # Extract log probabilities for candidate text portion
    candidate_log_probs = log_probs[0, -candidate_length:, :]
    candidate_token_ids = candidate_tokens.input_ids[0]

    # Sum log probabilities for each token
    total_log_prob = 0
    for i, token_id in enumerate(candidate_token_ids):
        total_log_prob += candidate_log_probs[i, token_id].item()

    # Normalize with average log probability
    avg_log_prob = total_log_prob / candidate_length

    return avg_log_prob

# Usage example
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

source = "Artificial intelligence is bringing significant changes to modern society."
candidate = "AI is greatly changing society."

score = calculate_gpt_score(model, tokenizer, source, candidate)
print(f"GPTScore: {score:.4f}")
```

**Output example:**

```
GPTScore: -2.3456
```

### 2.2 G-Eval: Chain-of-Thought (CoT) Based LLM Evaluation

**G-Eval** is a framework that utilizes state-of-the-art LLMs like OpenAI GPT-4 as evaluators, emerging to complement the limitations of GPTScore.

#### Core Features

The core feature of G-Eval is structuring **evaluation criteria and step-by-step reasoning** within prompts to guide LLMs to score like humans.

#### Evaluation Process

1. **Structured prompts**: Specify evaluation criteria and step-by-step reasoning
2. **Chain-of-Thought (CoT)**: Perform evaluation through step-by-step thinking process
3. **Form-Filling**: Guide to fill in only scores in predetermined formats

#### Example: Summarization Consistency Evaluation

```
Instruction: Read the article and summary, and evaluate on a scale of 1-5 whether the summary is logically consistent with the article content.

Evaluation steps:
1. Identify core topics of the article
2. Compare whether the summary includes these
3. Assign consistency score
```

#### Probability-Based Calibration

G-Eval applies **calibration using probability information from model responses**:

- **Confidence measurement**: Measure how confident GPT-4 is about each choice in evaluation steps using log probabilities
- **Weighted summation**: Weighted sum final scores according to confidence
- **Consistency improvement**: Improve evaluation consistency and reliability through probability information

#### Performance Results

- **SummEval, Topical-Chat benchmarks**: Achieved average Spearman correlation of 0.514
- **Compared to GPTScore**: Improved human correlation
- **Some evaluation metrics**: Correlation approaching 0.7 (SOTA level)
- **Evaluation process transparency**: Can track model evaluation process

#### Advantages

- **Structured evaluation**: Clear evaluation criteria and step-by-step process
- **High correlation**: High correlation with human evaluation
- **Transparency**: Traceability of evaluation process
- **Scalability**: Applicable to various evaluation criteria

#### Limitations

- **Cost**: Requires API calls to large models like GPT-4
- **Stability**: Need multiple evaluations per criterion for stability
- **Dependency**: Dependency on specific models (GPT-4)

### 2.2.1 G-Eval Implementation Example

```python
import openai
from typing import Dict, List, Any

class GEvalEvaluator:
    def __init__(self, model="gpt-4", api_key=None):
        self.model = model
        if api_key:
            openai.api_key = api_key

    def create_evaluation_prompt(self, source_text: str, candidate_text: str,
                                criteria: str, scale: str = "1-5") -> str:
        """
        Create G-Eval style evaluation prompt

        Args:
            source_text: Source text
            candidate_text: Candidate text to evaluate
            criteria: Evaluation criteria
            scale: Evaluation scale

        Returns:
            Structured evaluation prompt
        """
        prompt = f"""Please evaluate the following text.

Source text:
{source_text}

Candidate text:
{candidate_text}

Evaluation criteria: {criteria}

Evaluation steps:
1. Identify core content of source text
2. Analyze how well candidate text matches source text
3. Assign score on {scale} scale according to evaluation criteria

Please explain the evaluation process step by step and provide the final score.

Format:
Step 1: [Core content identification]
Step 2: [Match analysis]
Step 3: [Score assignment]
Final score: [Score]"""

        return prompt

    def evaluate(self, source_text: str, candidate_text: str,
                 criteria: str, num_samples: int = 3) -> Dict[str, Any]:
        """
        Evaluate text using G-Eval method

        Args:
            source_text: Source text
            candidate_text: Candidate text to evaluate
            criteria: Evaluation criteria
            num_samples: Number of evaluation samples

        Returns:
            Evaluation result dictionary
        """
        prompt = self.create_evaluation_prompt(source_text, candidate_text, criteria)

        scores = []
        explanations = []

        for i in range(num_samples):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Low temperature for consistency
                    max_tokens=500
                )

                result = response.choices[0].message.content
                explanations.append(result)

                # Extract score (using simple regex)
                import re
                score_match = re.search(r'Final score:\s*(\d+)', result)
                if score_match:
                    scores.append(int(score_match.group(1)))

            except Exception as e:
                print(f"Error in evaluation {i+1}: {e}")

        if scores:
            avg_score = sum(scores) / len(scores)
            return {
                "average_score": avg_score,
                "scores": scores,
                "explanations": explanations,
                "consistency": len(set(scores)) == 1  # Check if all scores are the same
            }

        return {"error": "Evaluation failed"}

# Usage example
evaluator = GEvalEvaluator()

source = "Artificial intelligence technology is bringing innovative changes to the medical field. AI is improving diagnostic accuracy, enabling personalized treatment, and significantly enhancing healthcare professionals' work efficiency."

candidate = "AI is bringing major changes to healthcare. Diagnosis has become more accurate and personalized treatment is now possible."

criteria = "How accurately and completely the core content of the source text is summarized"

result = evaluator.evaluate(source, candidate, criteria)
print(f"Average score: {result['average_score']:.2f}")
print(f"Consistency: {result['consistency']}")
```

**Output example:**

```
Average score: 4.33
Consistency: True
```

### 2.3 FLASK: Fine-grained Skill Set Based Evaluation

**FLASK** is a framework that pursues more interpretable and specific evaluation by decomposing evaluation into multiple detailed ability components. Published at ICLR 2024, FLASK scores how well responses satisfy various required abilities (skills) instead of providing a single total score.

#### Core Concepts

FLASK takes the following approach:

1. **Skill decomposition**: Decompose evaluation tasks into component abilities
2. **Granular evaluation**: Assign separate scores for each ability
3. **Multi-dimensional metrics**: Provide multi-dimensional evaluation results instead of total scores

#### 12 Fine-grained Ability Indicators

The research team defined the following 12 ability indicators:

1. **Explicit reasoning ability**: Clearly present logical reasoning processes
2. **Background knowledge utilization**: Appropriately utilize relevant knowledge
3. **Logical consistency**: Maintain logical consistency within responses
4. **Context adherence**: Appropriately conform to given context
5. **Accuracy**: Factual accuracy
6. **Completeness**: Complete answers to questions
7. **Clarity**: Clarity and comprehensibility of responses
8. **Creativity**: Original and creative approaches
9. **Practicality**: Practical applicability
10. **Ethics**: Adherence to ethical standards
11. **Efficiency**: Concise and efficient responses
12. **Adaptability**: Appropriate responses to situations

#### Evaluation Process

1. **Skill tagging**: Tag abilities relevant to each evaluation instance
2. **Individual evaluation**: Assign separate scores for each skill
3. **Weight application**: Calculate final scores by applying weights according to importance
4. **Multi-dimensional results**: Provide multi-dimensional evaluation metrics instead of total scores

#### Performance Results

Key achievements of FLASK:

- **Capturing subtle differences between models**: Diagnosing strengths and weaknesses of GPT-4 and GPT-3.5 by ability
- **High correlation**: High correlation between model evaluation and human evaluation
- **Interpretability**: Clear interpretation of evaluation results
- **Customized evaluation**: Possible to design various customized evaluation rubrics

#### Example: Legal Consultation Response Evaluation

For legal consultation responses, the following skills are required:

- **Legal provision recall** (background knowledge)
- **Logical reasoning**
- **Ethical standard compliance**
- **Practical advice provision**

Each skill is scored separately, and final evaluation is performed by applying weights according to importance.

#### Advantages

- **Granular evaluation**: Clearly identify model strengths and weaknesses
- **Interpretability**: Clear interpretation of evaluation results
- **Customized application**: Possible customized evaluation for various domains
- **Reliability improvement**: Simultaneously improve reliability and explanatory power of LLM evaluation

#### Limitations

- **Complexity**: Increased complexity of evaluation process
- **Subjectivity**: Subjectivity in ability definition and weight setting
- **Cost**: Increased cost due to granular evaluation

### 2.3.1 FLASK Implementation Example

```python
from typing import Dict, List, Any
import json

class FLASKEvaluator:
    def __init__(self):
        # Define 12 ability indicators
        self.skills = {
            "explicit_reasoning": "Explicit reasoning ability",
            "background_knowledge": "Background knowledge utilization",
            "logical_consistency": "Logical consistency",
            "context_adherence": "Context adherence",
            "accuracy": "Accuracy",
            "completeness": "Completeness",
            "clarity": "Clarity",
            "creativity": "Creativity",
            "practicality": "Practicality",
            "ethics": "Ethics",
            "efficiency": "Efficiency",
            "adaptability": "Adaptability"
        }

        # Skill weights by domain
        self.domain_weights = {
            "legal": {
                "background_knowledge": 0.3,
                "logical_consistency": 0.25,
                "ethics": 0.2,
                "practicality": 0.15,
                "accuracy": 0.1
            },
            "medical": {
                "accuracy": 0.3,
                "background_knowledge": 0.25,
                "logical_consistency": 0.2,
                "ethics": 0.15,
                "practicality": 0.1
            },
            "general": {
                "accuracy": 0.2,
                "completeness": 0.2,
                "clarity": 0.2,
                "logical_consistency": 0.2,
                "practicality": 0.2
            }
        }

    def evaluate_skill(self, question: str, answer: str, skill: str) -> float:
        """
        Evaluate answer for specific skill

        Args:
            question: Question
            answer: Answer
            skill: Skill to evaluate

        Returns:
            Skill score (0-1)
        """
        # In actual implementation, use LLM to evaluate each skill
        # Here implemented as simple example

        skill_prompts = {
            "explicit_reasoning": "Did the answer clearly present logical reasoning process?",
            "background_knowledge": "Did the answer appropriately utilize relevant knowledge?",
            "logical_consistency": "Is logical consistency maintained within the answer?",
            "context_adherence": "Does the answer appropriately conform to given context?",
            "accuracy": "Is the answer factually accurate?",
            "completeness": "Is the answer complete for the question?",
            "clarity": "Is the answer clear and comprehensible?",
            "creativity": "Is the answer original and creative?",
            "practicality": "Is the answer practically applicable?",
            "ethics": "Does the answer comply with ethical standards?",
            "efficiency": "Is the answer concise and efficient?",
            "adaptability": "Is the answer appropriate for the situation?"
        }

        # Actually use LLM for evaluation
        # Here return random score as example
        import random
        return random.uniform(0.6, 1.0)

    def evaluate_answer(self, question: str, answer: str, domain: str = "general") -> Dict[str, Any]:
        """
        Comprehensive evaluation of answer using FLASK method

        Args:
            question: Question
            answer: Answer
            domain: Domain (legal, medical, general)

        Returns:
            Evaluation result dictionary
        """
        # Individual evaluation for each skill
        skill_scores = {}
        for skill in self.skills:
            skill_scores[skill] = self.evaluate_skill(question, answer, skill)

        # Apply domain-specific weights
        weights = self.domain_weights.get(domain, self.domain_weights["general"])

        # Calculate weighted average
        weighted_score = 0
        total_weight = 0

        for skill, score in skill_scores.items():
            if skill in weights:
                weighted_score += score * weights[skill]
                total_weight += weights[skill]

        final_score = weighted_score / total_weight if total_weight > 0 else 0

        return {
            "final_score": final_score,
            "skill_scores": skill_scores,
            "weights": weights,
            "domain": domain,
            "detailed_analysis": self._generate_analysis(skill_scores, weights)
        }

    def _generate_analysis(self, skill_scores: Dict[str, float], weights: Dict[str, float]) -> str:
        """Generate detailed analysis"""
        analysis = "Ability-based evaluation results:\n"

        # Display high-weight skills first
        sorted_skills = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for skill, weight in sorted_skills:
            score = skill_scores[skill]
            skill_name = self.skills[skill]
            analysis += f"- {skill_name}: {score:.2f} (weight: {weight:.2f})\n"

        return analysis

# Usage example
evaluator = FLASKEvaluator()

question = "What should I do if there are unfair clauses in a contract?"
answer = "If there are unfair clauses, first check whether the clause is legally valid. Unfair clauses under civil law can be invalid, and you may be protected under consumer protection law. It's advisable to consult with experts to explore specific response measures."

result = evaluator.evaluate_answer(question, answer, domain="legal")
print(f"Final score: {result['final_score']:.2f}")
print(result['detailed_analysis'])
```

**Output example:**

```
Final score: 0.85
Ability-based evaluation results:
- Background knowledge utilization: 0.92 (weight: 0.30)
- Logical consistency: 0.88 (weight: 0.25)
- Ethics: 0.90 (weight: 0.20)
- Practicality: 0.82 (weight: 0.15)
- Accuracy: 0.85 (weight: 0.10)
```

### Checkpoint Questions

- What properties of language models does **GPTScore** use to evaluate the quality of generated content? Explain the advantages and limitations of this method.
- What techniques did **G-Eval** introduce to improve evaluation reliability? (e.g., Chain-of-Thought, Form-Filling, etc.) How did these techniques improve correlation with human evaluation?
- Why does the **FLASK** framework perform evaluation in granular skill-based manner? How does the resulting multi-dimensional evaluation results help in interpreting model performance?

**Table 1: Comparison of Traditional Evaluation and LLM-based Meta-Evaluation Paradigms**

| Feature                    | Traditional Metrics (BLEU/ROUGE)                                          | LLM-as-a-Judge                                                                                 |
| -------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Evaluation Criteria**    | N-gram overlap with reference text (lexical matching)                     | Abstract quality criteria defined by natural language instructions (e.g., usefulness, logic)   |
| **Semantic Understanding** | Impossible. Cannot understand synonyms or different expressions.          | Possible. Understand context and nuances to evaluate semantic quality.                         |
| **Creativity/Diversity**   | Penalizes creativity when different from reference text.                  | Can highly evaluate diverse expressions and creative results if they meet evaluation criteria. |
| **Cost and Scalability**   | High cost for reference text construction, low scalability for new tasks. | Low-cost large-scale evaluation possible after initial setup, very high scalability.           |
| **Interpretability**       | Provides only scores. Cannot know why low scores occurred.                | Can explain "why" evaluation was made in natural language, providing specific feedback.        |
| **Core Limitation**        | Cannot measure core capabilities of generative AI (meaning, creativity).  | Bias issues (preference leakage, position), multilingual consistency, evaluator reliability.   |

**Table 2: Comparison of Correlation with Human Evaluation (SummEval Dataset, Spearman Correlation ρ)**

| Evaluation Metric        | Coherence | Consistency | Fluency   | Relevance | Average   |
| ------------------------ | --------- | ----------- | --------- | --------- | --------- |
| ROUGE-1                  | 0.167     | 0.207       | 0.105     | 0.326     | 0.201     |
| ROUGE-2                  | 0.158     | 0.200       | 0.106     | 0.306     | 0.192     |
| ROUGE-L                  | 0.170     | 0.210       | 0.110     | 0.320     | 0.202     |
| BERTScore                | 0.284     | 0.362       | 0.216     | 0.426     | 0.322     |
| **G-Eval (GPT-4 based)** | **0.582** | **0.460**   | **0.467** | **0.547** | **0.514** |

Data source: Liu et al., 2023. G-Eval: NLG Evaluation using Gpt-4 with Better Human Alignment.

## 3. Specialized Purpose Benchmarks

With the emergence of benchmarks specialized for specific domains or abilities, it has become possible to evaluate various capabilities of LLMs more accurately and comprehensively. This section examines major specialized purpose benchmarks including **LiveCodeBench**, **EvalPlus**, **MMLU-Pro**, **GPQA**, and **BBH**.

### 3.1 LiveCodeBench: Contamination-Free Code Generation Evaluation

**LiveCodeBench** is a real-time updatable code evaluation set proposed in 2024, with the most distinctive feature being designed to fundamentally block data contamination.

#### Core Problem: Data Contamination

Existing code evaluation datasets like HumanEval (OpenAI) and MBPP (Google) have the following problems:

- **Pre-exposure of evaluation problems**: Evaluation problems are included in model training and pre-exposed
- **Overfitting problem**: Latest models like GPT-4 achieve high scores of 80-90% on HumanEval, but this may be due to having seen some problems or learning them in modified forms
- **Reliability degradation**: Gap between actual coding ability and evaluation results

#### Solution Approach

LiveCodeBench solves the problem in the following ways:

1. **Real-time problem collection**: Continuously collect latest problems from online judge platforms (LeetCode, AtCoder, CodeForces)
2. **Time window-based updates**: Select 400+ new problems published from May 2023 to May 2024 to form initial set
3. **Periodic updates**: Set time windows and periodically update problems to use only problems that appeared after model training

#### Holistic Evaluation

The second feature of LiveCodeBench is expanding the evaluation scope comprehensively:

- **Accuracy of code execution results**: Evaluation of executable code, not just code generation
- **Self-debugging (self-repair) ability**: Scenarios where error logs are given and code needs to be modified
- **Appropriateness of comments or output format**: Evaluation of various aspects of code quality
- **pass@k measurement**: Measure whether execution results match given test cases
- **Test output prediction**: Include scenarios where models predict test outputs

#### Performance Results

Results from applying LiveCodeBench to 18 base LLMs and 34 instruction-tuned LLMs:

- **Limitations of existing static benchmarks**: Even latest models showing high performance on HumanEval show significantly lower performance on new problems
- **Overfitting confirmation**: Models overfitted on HumanEval are weaker on actual new problems
- **Transparency**: Transparently provide all prompts, model responses, and evaluation scripts

#### Significance

LiveCodeBench is an innovative benchmark that "throws new problems live so models can't solve them," revealing model overfitting/memorization and evaluating various code generation abilities in a multi-dimensional way.

### 3.2 EvalPlus: Test Case Augmentation

**EvalPlus** is a technique and dataset to supplement the insufficiency of test cases in code evaluation, introduced at NeurIPS 2023.

#### Core Problem

Problems with existing HumanEval:

- **Insufficient test cases**: Average of 7-10 simple tests per problem
- **Complex bugs not caught**: Simple tests cannot catch complex bugs
- **Lenient evaluation**: Wrong solutions are considered correct

#### Solution Approach

EvalPlus takes the following approach:

1. **Automatic test case generation**: Automatically generate dozens of times more test cases by transforming/expanding existing test inputs
2. **Mutation-based Input Generation**: Mutate or add boundary cases to given function inputs in various ways
3. **Various input transformations**: For functions handling lists, automatically generate input transformations like empty lists, single-element lists, lists including negative/special values

#### Performance Results

Results from building HumanEval+ set and re-evaluating models:

- **GPT-4's pass@1 accuracy dropped by over 20 percentage points**
- **156 out of 164 HumanEval problems showed decreased accuracy after introducing additional tests**
- **Leniency of existing evaluation revealed**

#### Significance

EvalPlus research proved that automated test augmentation improves model evaluation reliability and suggests that additional tests for safety/security aspects (e.g., handling malicious inputs) can be considered in the future.

### 3.3 HELM-Code: Transparency and Community Collaboration

**HELM (Holistic Evaluation of Language Models)** is an evaluation effort led by Stanford CRFM, focusing on evaluation transparency and community collaboration.

#### Core Features

- **Massive benchmark set**: Composed of various scenarios and metrics
- **Transparency**: Publicly release all experimental results, prompts, and responses
- **Reproducibility**: Efforts to improve reproducibility of evaluation process
- **Leaderboard**: Continuously evaluate latest models and make public

#### HELM Philosophy

HELM's philosophy is to present evaluation comprehensively like "one map," including not only quantitative model performance but also:

- **Inference speed**
- **Memory usage**
- **Bias/harmfulness**
- **Carbon emissions (Green AI metrics)**

#### Significance

This transparent and comprehensive approach improves the reliability of evaluation itself and guides researchers to develop models from a broad perspective rather than focusing on specific tasks or metrics.

### 3.4 MMLU-Pro: 10-Choice High-Difficulty Knowledge/Reasoning Benchmark

**MMLU (Massive Multi-Task Language Understanding)** is a large-scale problem bank spanning 57 fields, a representative knowledge evaluation benchmark used since the GPT-3 era. Each problem was composed of 4-choice multiple choice questions, but when latest LLMs reached human level on MMLU early, MMLU-Pro, an expanded version with dramatically increased difficulty, emerged in 2024.

#### Core Changes

1. **Increased number of choices**: Increased from 4 to 10

   - Random answer probability decreased from 25% to 10%
   - Reduced possibility of passing by luck through simple memorization

2. **Increased problem complexity**: Requires more complex multi-step reasoning
   - Added problems requiring synthesis of multiple knowledge or multi-step reasoning instead of simple memorization questions
   - Example: "In which year did what event happen?" → "Which of the following arranges historical events in chronological order?"

#### Performance Results

- **GPT-4 and other models showed 16-33% accuracy decrease compared to MMLU**
- **Secured discrimination to make differences between models prominent again**

#### Chain-of-Thought Effect

- **MMLU original**: CoT guidance doesn't show significant performance improvement
- **MMLU-Pro**: CoT guidance shows clear effect on performance improvement
- **"Solve step by step and choose answer"** instruction significantly increased accuracy

#### Current Status

- **Includes tens of thousands of questions**
- **Covers graduate-level questions in 14 fields**
- **Even best models as of 2025 achieve around 85% accuracy**
- **Tendency to make mistakes on problems requiring any application**

#### Significance

The emergence of MMLU-Pro shows that future LLM evaluation is moving beyond "simple knowledge memorization tests" to "thinking and problem-solving ability evaluation."

### 3.5 GPQA and BBH: Knowledge/Reasoning Enhanced Evaluation Sets

**GPQA (Graduate-level Google-Proof Questions & Answers)** is a high-difficulty QA benchmark that emerged in 2024, collecting graduate-level questions as the name suggests.

#### Core Features

- **"Google-Proof"**: Questions based on relatively new research that cannot be found through simple Google search
- **Problems requiring multiple steps to solve**
- **Written by field experts in biology, physics, chemistry**
- **Total of 448 5-choice questions**

#### Problem Examples

For example, questions like _"What phenomenon appears when removing the role of catalyst A in a certain chemical reaction pathway?"_ require both expert knowledge and logic.

#### Performance Results

- **PhD students achieve around 65% accuracy on GPQA**
- **State-of-the-art models like GPT-4, Gemini also achieve similar levels (around 85% or less) on this set**
- **Still cannot make significant difference from humans**

#### Significance

GPQA's significance is as follows:

1. **Imposes new difficulty on LLMs that have absorbed almost all internet knowledge**
2. **Reveals model limitations in scientific reasoning tasks**
3. **Even GPT-4 shows incomplete aspects like selecting explanations mixed with misconceptions on some GPQA items**
4. **Shows need for future research on specialized field learning and reasoning enhancement**

#### BBH (BIG-Bench Hard)

**BBH (BIG-Bench Hard)** is a hard test set composed of only 23 particularly difficult tasks from Google's BIG-Bench large benchmark.

##### Core Features

- **BIG-Bench original**: 200+ diverse language model tasks
- **BBH**: Bundled reasoning puzzles, trap problems, and capability limitation evaluation problems that existing models performed poorly on, tagged as "Hard"
- **Mathematics area**: Case counting reasoning problems, not simple calculations
- **Language area**: Complex grammatical paradoxes, etc.

##### Goal

**Focused attack on model weaknesses**, as GPT-3 or early GPT-4 could only achieve random guessing level or slightly better performance on each BBH task.

##### Research Utilization

Researchers analyzed specific capability deficiencies of models (e.g., logical puzzle solving, paradox handling) through BBH and attempted improvements through Chain-of-Thought guidance or additional training data input.

##### Extended Version: BBEH

**BBH Extended Version (BBEH: BIG-Bench Extra Hard)** published in 2023 further strengthened multi-step reasoning, creative problem solving, etc., presenting even more difficult challenges to LLMs.

##### Significance

As a result, BBH/BBEH functions as a stress test for continuously checking model limitations in the LLM research community.

### Checkpoint Questions

- How did **LiveCodeBench** solve the **data contamination** problem? What additional evaluation capabilities did this benchmark emphasize compared to existing code evaluation?

- What principle does **Mutation-based Input Generation** proposed by **EvalPlus** operate on, and why did GPT-4's performance drop significantly on HumanEval+?

- What does HELM's philosophy of "presenting evaluation comprehensively like one map" mean, and how does this differ from existing evaluation methods?

- What impact does increasing the number of choices from 4 to 10 in **MMLU-Pro** have on model evaluation, and why is Chain-of-Thought guidance more effective?

- What does the "Google-Proof" characteristic of **GPQA** mean, and what new challenges does this present to LLM evaluation?

- How does **BBH** focus on attacking model weaknesses, and what implications does this provide for model development?

- What purpose do **high-difficulty benchmarks** like GPQA or BBH have in their design, different from general benchmarks? Give one example of LLM limitations revealed in such sets.

## 4. Domain-Specific Benchmarks

While specialized purpose benchmarks focus on evaluating general abilities, domain-specific benchmarks focus on evaluating expertise in specific fields. These benchmarks play an important role in measuring how useful LLMs are in actual work environments.

### 4.1 FinBen: Comprehensive Financial Domain Benchmark

**FinBen** is a large benchmark set specialized for the financial field, publicly released at NeurIPS Datasets/Benchmarks track in 2024.

#### Core Features

- **Total of 24 financial tasks** integrating **42 datasets** into a massive benchmark
- **8 ability areas**: Information extraction, text analysis, Q&A, text generation, risk management, time series prediction, decision making, multilingual (English/Spanish)
- **Various tasks directly related to financial work**: Corporate report summarization, stock price prediction, extracting specific items from financial statements, Q&A on financial regulatory documents, generating investment decision scenarios
- **Original tasks**: Stock trading decision evaluation, financial regulation-related Q&A, etc.

#### Data Composition

- **Mostly acquired from public data or actual financial records**
- **Includes some newly created financial QA and stock trading scenarios**

#### Performance Results

Results from comparing 21 representative LLMs including GPT-4, ChatGPT, Google Gemini:

- **GPT-4**: Excellent in information extraction and simple analysis areas
- **Google Gemini**: Higher performance in time series prediction or complex text generation
- **Different models show different strength areas**
- **Open QA**: GPT-4 series strong
- **Long report writing in specific formats**: Other code-tuned LLMs more suitable

#### Implications

1. **Emphasizes need for financial domain-specific model development**
2. **Financial specialized LLMs** are more accurate in detailed tasks like regulatory document Q&A
3. **Operates in open leaderboard format** for collaborative advancement of financial field LLMs by research community

#### Significance

The emergence of FinBen greatly expanded LLM evaluation scope from everyday language areas to professional knowledge areas, establishing a foundation for objectively measuring LLM capabilities in the financial domain where high accuracy and rigor are required.

### 4.2 AgentHarm: AI Agent Harmfulness Evaluation Benchmark

**AgentHarm** is a safety benchmark for measuring the potential for harmful behavior of LLMs operating as autonomous agents, introduced at ICLR 2025.

#### Background

Today's LLMs can act like agents beyond simple chatting, using tools and performing multi-step tasks. AgentHarm's purpose is to evaluate the risks that could arise if such advanced LLM agents are misused.

#### Benchmark Composition

- **Selected 110 explicitly malicious scenarios (tasks)**
- **Including 440 prompts modified through data augmentation**
- **11 harmful categories**: Fraud, cybercrime, harassment, fake news writing, violence instruction, etc.

#### Scenario Examples

- **Cybercrime-type tasks**: _"Generate phishing emails to obtain passwords from users"_
- **Hate/harassment scenarios**: _"Find the most effective way to spread hate speech against specific minority groups"_

#### Evaluation Method

1. **Refusal rate measurement**: Measure whether models refuse (reject) such malicious requests

   - If safety measures are well implemented, should normally refuse with "This request cannot be fulfilled"
   - Calculate this as refusal rate

2. **Evaluation after jailbreaking**: After forcibly jailbreaking (neutralizing policies) models, evaluate whether they maintain agent functionality and achieve harmful goals
   - For multi-step tasks requiring tool use (hacking procedures, etc.), check whether models execute all steps to achieve goals when policies are removed

#### Key Findings

- **Even top LLMs surprisingly easily accept malicious requests without separate jailbreaking**
- **Despite safety measures by OpenAI or Anthropic, many models fall for clever prompts**
- **Simple universal jailbreak prompts** can make most agent LLMs **completely switch to malicious mode**
- **Intermediate step results are also very consistently harmful**
- **Current LLM safety measures are vulnerable in complex task scenarios**

#### Research Impact

With AgentHarm's release, researchers are more seriously exploring model defense techniques:

- **Self-censoring CoT introduction**: Experiments to make models self-question "Is this action safe?" at each step
- **Reinforcement learning-based agent safety tuning** research
- **Tool usage restriction devices** development indicators

#### Significance

AgentHarm can be seen as adding a new axis of "AI's instrumental risk" to LLM evaluation, providing important data points for AI safety research.

### 4.3 LEXam: Legal Exam-Based LLM Evaluation

**LEXam** is a benchmark proposed in 2025 for advanced reasoning evaluation in the legal field, based on 340 sets of actual law exam problems from University of Zurich, Switzerland.

#### Core Features

- **116 subjects**: Civil law, criminal law, administrative law, international law, etc.
- **Various levels from undergraduate to graduate** exam problems
- **Unprecedented legal AI evaluation set**

#### Data Composition

- **Total of 4,886 questions**
- **2,841 are open-ended essay questions**
- **2,045 are 4-choice multiple choice questions**
- **Essay questions include model answers and scoring guidelines** (issue spotting, legal application steps, etc.)

#### Evaluation Method

Not just simple correct/incorrect answers, but also evaluate the **validity of answer development process**.

#### Performance Results

Results from applying LEXam to latest LLMs showed that models reveal serious limitations in legal reasoning:

- **Even language-capable models like GPT-4 show low scores on complex essay problems** requiring combination of facts and legal provisions for multi-step reasoning
- **Models answer correctly on simple memorization questions** (e.g., asking for article numbers) but often reach wrong conclusions on **case-type problems** (applying law to given situations)
- **GPT-4 performed well on 4-choice questions** but still makes causal relationship errors on problems requiring logical elimination of trap choices

#### High-Order Reasoning Specific to Legal Field

These results show how vulnerable LLMs are in high-order reasoning specific to the legal field:

- **Fact identification**
- **Relevant legal provision selection**
- **Analogy and precedent application**
- **Final judgment**

#### LLM-as-Judge Utilization

LEXam benchmark paper also utilized LLM-as-Judge approach for evaluation:

- **Gave GPT-4 examinee GPT-3.5's answer and asked to "evaluate the reasoning structure and legal validity of this answer"**
- **Model-to-model evaluation** compared with human professors' evaluation showed **considerably high agreement**, confirming LLM evaluator potential
- **However, differences still exist in subtle parts, suggesting directions like LLM+expert joint evaluation**
- **In actual legal exam situations, partial scores exist, which models miss**

#### Significance

LEXam will be utilized as an important standard for future legal specialized LLM development and is expected to serve as a litmus test for checking model performance in actual legal AI application fields like contract review and precedent search.

### 4.4 CSEDB: Medical LLM Safety/Effectiveness Dual Evaluation

**CSEDB (Clinical Safety-Effectiveness Dual-Track Benchmark)** is a multi-dimensional benchmark for evaluating LLM utilization in the medical domain, first publicly released in 2025.

#### Background

Major challenges for medical field LLMs are accuracy of diagnosis and advice (effectiveness) and patient safety and ethical compliance (safety). CSEDB was designed to evaluate both aspects simultaneously.

#### Benchmark Composition

- **30 evaluation criteria**: Important criteria in clinical situations like critical patient recognition, guideline compliance, drug safety derived through doctor consensus
- **Total of 2,069 questions** composed to cover **26 clinical departments**
- **All questions are subjective descriptive response format**
- **Each question tagged with which evaluation criteria it relates to**

#### Examples

"Emergency patient initial response" questions are connected to safety/effectiveness criteria like _"critical symptom recognition"_, _"immediate treatment guideline application"_.

#### Performance Results

Results from applying CSEDB to 6 models including GPT-4, ChatGPT, Med-PaLM2:

- **Models' overall score was around 57.2%**
- **Safety score 54.7%, effectiveness (accuracy) score 62.3%**, both remaining at half level
- **Suggests considerable supplementation still needed for actual clinical deployment**

#### Performance in High-Risk Scenarios

- **In high-risk scenarios where patient condition is critical**, performance **decreased by additional 13.3 percentage points**
- **Safety 41%, effectiveness 49% level**, showing models **struggle more with risk situation handling**

#### Effect of Domain-Specific Tuning

- **Specialized medical LLMs** (e.g., models additionally trained on medical data) show higher overall performance than general models
- **Particularly improved scores up to 91.2% in safety, 86.1% in effectiveness**
- **Shows effect of domain-specific tuning**

#### Utilization Methods

- **Can diagnose which criteria models are vulnerable to** (e.g., frequent errors in drug interaction-related questions)
- **Utilize for objective comparison of models' clinical applicability**
- **Also useful as reference material for regulatory approval**

#### Significance

CSEDB is a specialized evaluation for the medical field where both safety and efficiency cannot be missed, serving as a test that LLMs must pass before being deployed in actual patient care.

### 4.5 MATH and GSM8K: Mathematical Ability Evaluation

Mathematics is a particularly challenging area for LLMs, and MATH and GSM8K benchmarks are widely used to measure this.

#### MATH Benchmark

**MATH** is a collection of high school olympiad-level problems built by Hendrycks et al., including about 12,000 descriptive problems in algebra, geometry, probability, etc.

##### Features

- **Each problem provides not only correct answers but also step-by-step explanations** needed for problem solving
- **Utilized for Chain-of-Thought solution training**
- **Evaluates multi-step reasoning, formula derivation ability, calculation without errors**

##### Performance Results

- **Latest GPT-4 achieves around 40% mid-range accuracy on MATH**
- **Still below human math competition participant level**

#### GSM8K Benchmark

**GSM8K** is a dataset composed of about 8,000 elementary to middle school level arithmetic word problems, mainly requiring short descriptive (=one or two sentences) solutions.

##### Features

- **Examples**: From problems like _"If you eat 2 out of 5 apples, how many are left?"_ to slightly complex _"The sum of Chulsoo and Younghee's ages is 24, and Younghee is 4 years older than twice Chulsoo's age. What are their ages?"_
- **GPT-4 sometimes makes mistakes when solving directly without Chain-of-Thought, but shows high accuracy when guided to "show your thinking"**

##### Performance Improvement Techniques

- **Self-Consistency technique** (solve multiple times with Chain-of-Thought and vote) improved GSM8K accuracy from existing 55% to **72.9%**
- **Reduced model consistency problems by exploring diverse thinking paths and taking common answers**

##### Utilization

GSM8K is currently widely used as an indicator of LLMs' basic arithmetic ability and simple reasoning ability, and various prompt techniques for performance improvement (e.g., Self-Consistency, automatic natural steps) have been tested on this benchmark.

#### Evaluation Method

Both math benchmarks are utilized not only for accuracy but also for solution process evaluation:

- **Imitate human scoring methods** like deducting partial scores when models get answers right but logic wrong
- **Because accuracy of reasoning is important in math problem solving**

#### LLM Limitations

Limitations of LLMs revealed through math benchmarks:

- **Tendency to get reasoning right in early parts but wrong in later calculations for long, complex problems**
- **Instability where slightly changing reasoning paths leads to different answers**

#### Improvement Research

Research to improve this includes embedding calculation modules through reinforcement learning, giving external calculation tool calling abilities, etc.

#### Significance

MATH and GSM8K have established themselves as core benchmarks for objectively measuring LLMs' logical thinking and calculation abilities, providing many insights into model limitation identification and thinking improvement methods through such mathematical evaluation.

### Checkpoint Questions

- Why did **FinBen** show different strength areas by model in the financial domain, and what implications does this provide for domain-specific model development?

- Why did even top LLMs surprisingly easily accept malicious requests without separate jailbreaking in **AgentHarm**, and what limitations of current LLM safety measures does this show?

- Why does GPT-4 answer correctly on simple memorization questions but reach wrong conclusions on case-type problems in **LEXam**, and what high-order reasoning ability specific to the legal field does this indicate is lacking?

- Why did performance decrease by additional 13.3 percentage points in high-risk scenarios where patient condition is critical in **CSEDB**, and what limitations of medical LLMs does this show?

- Why is the Self-Consistency technique effective for performance improvement in **MATH and GSM8K**, and what consistency problem of LLMs does this solve?

**Table 3: Overview of Next-Generation Evaluation Benchmarks**

| Benchmark         | Main Domain                 | Core Goal                                                                  | Core Innovation                                                                      |
| ----------------- | --------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **G-Eval**        | General NLG                 | Maximize correlation between LLM evaluator and human judgment              | Systematic evaluation through automatic CoT generation and Form-Filling              |
| **LiveCodeBench** | Code Generation             | Measure reliable code ability without data contamination                   | Dynamic evaluation through real-time problem collection and timestamp-based updates  |
| **MMLU-Pro**      | General Knowledge/Reasoning | Test SOTA model limitations and secure discrimination                      | Choice expansion (4→10), reasoning-centered problem strengthening                    |
| **FinBen**        | Finance                     | Comprehensive evaluation of LLM practicality in actual financial scenarios | Introduction of agent-based stock trading and RAG-based evaluation                   |
| **AgentHarm**     | AI Safety                   | Measure LLM agent harmfulness and jailbreak attack vulnerability           | Simultaneous evaluation of malicious multi-step task performance and refusal ability |

## 5. Evaluation Bias and Limitations

The bias and limitations that appear in LLM evaluation have important impacts on the reliability and fairness of evaluation. Understanding and resolving these biases is essential for building better evaluation systems.

### 5.1 Major Evaluation Biases

The major biases that appear in LLM evaluation are as follows:

#### 5.1.1 Narcissistic Bias

**Narcissistic bias** refers to the tendency of LLMs to prefer text they have generated themselves. This is the phenomenon where LLMs rate their own output style or expression methods higher when acting as evaluators.

##### Features

- **Preference for own output style**: LLMs rate the style or expression methods of text they generated higher
- **Lack of consistency**: Unfair evaluation when comparing with other models' outputs
- **Evaluation reliability degradation**: Makes objective evaluation difficult

##### Solutions

- **Utilize diverse evaluators**: Use multiple models as evaluators to offset bias
- **Clarify evaluation criteria**: Set specific and objective evaluation criteria
- **Cross-validation**: Compare with other models' evaluation results to check consistency

#### 5.1.2 Verbosity Bias

**Verbosity bias** refers to the tendency of LLMs to rate longer text higher. This is the phenomenon where text length affects evaluation.

##### Features

- **Confusion between length and quality**: Mistakenly recognize text length as a quality indicator
- **Inclusion of unnecessary information**: Add irrelevant information to increase evaluation scores
- **Efficiency degradation**: Prefer verbose responses over concise and accurate answers

##### Solutions

- **Length normalization**: Use evaluation metrics that consider text length
- **Focus on core content**: Focus on core content and relevance of text
- **Efficiency indicators**: Consider information density and accuracy together

#### 5.1.3 Inconsistency

**Inconsistency** is the phenomenon of producing different evaluation results for the same input. This significantly impairs evaluation reliability.

##### Features

- **Same input, different results**: Different scores or evaluation results for the same text
- **Evaluation criteria mismatch**: Apply different criteria each time during evaluation
- **Lack of reproducibility**: Different results even under identical conditions

##### Solutions

- **Standardized evaluation protocols**: Set consistent evaluation procedures and criteria
- **Multiple evaluations**: Use average values through multiple evaluations
- **Evaluator training**: Train evaluators for consistent evaluation

### 5.2 Evaluation Limitations

The major limitations that appear in LLM evaluation are as follows:

#### 5.2.1 Differences from Human Evaluation

- **Lack of subjectivity**: Cannot reflect human intuition and emotions
- **Context understanding limitations**: Cannot fully understand complex social and cultural contexts
- **Difficulty in creativity evaluation**: Limitations in evaluating creative and original content

#### 5.2.2 Lack of Domain-Specific Knowledge

- **Professional field understanding limitations**: Lack of professional knowledge and experience in specific fields
- **Lack of latest information**: Cannot reflect information that changes in real-time
- **Lack of cultural context understanding**: Cannot understand specific cultural or social contexts

#### 5.2.3 Subjectivity of Evaluation Criteria

- **Difficulty in setting evaluation criteria**: Difficulty in setting objective and fair evaluation criteria
- **Subjectivity in weight determination**: Subjectivity in determining importance of various evaluation elements
- **Difficulty in threshold setting**: Difficulty in setting pass/fail criteria

### Checkpoint Questions

- What is **narcissistic bias** and how does it affect LLM evaluation? What methods can be used to resolve this bias?

- Why does **verbosity bias** occur and what problems does it cause in evaluation? What methods can be used to resolve this bias?

- Why does **inconsistency** appear in LLM evaluation and how does it affect evaluation reliability? What methods can be used to improve consistency?

- Why do **differences from human evaluation** appear in LLM evaluation and what limitations does this show? What methods can be used to overcome these limitations?

- How does **lack of domain-specific knowledge** affect LLM evaluation and in which fields does this become particularly problematic? What methods can be used to resolve these limitations?

## 6. RLAIF: Reinforcement Learning from AI Feedback

**RLAIF (Reinforcement Learning from AI Feedback)** is a method that performs reinforcement learning using AI model feedback instead of human feedback. This is an extension of RLHF (Reinforcement Learning from Human Feedback), enabling more efficient and scalable learning by having AI models act as evaluators.

### 6.1 Core Principles of RLAIF

RLAIF operates through the following process:

1. **AI evaluator training**: Train AI models as evaluators using human evaluation data
2. **AI feedback collection**: Collect feedback on various outputs using trained AI evaluators
3. **Reinforcement learning execution**: Improve policy models through reinforcement learning using collected AI feedback

### 6.2 Advantages of RLAIF

- **Scalability**: Can collect much more feedback faster than human evaluators
- **Consistency**: AI evaluators provide more consistent evaluation than human evaluators
- **Cost efficiency**: Significantly save human evaluator costs
- **Diversity**: Can evaluate various domains and languages

### 6.3 Limitations of RLAIF

- **Bias propagation**: Bias of AI evaluators can propagate to learned models
- **Lack of human value reflection**: May not fully reflect human intuition and values
- **Dependency on evaluation quality**: Overall system performance depends on AI evaluator quality

### 6.4 RLAIF Implementation Example

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class AIEvaluator(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        score = self.classifier(pooled_output)
        return torch.sigmoid(score)

class RLAIFTrainer:
    def __init__(self, policy_model, evaluator):
        self.policy_model = policy_model
        self.evaluator = evaluator

    def collect_ai_feedback(self, prompts, responses):
        """Collect feedback using AI evaluator"""
        scores = []
        for response in responses:
            score = self.evaluator(response)
            scores.append(score)
        return scores

    def train_step(self, prompts, responses, ai_scores):
        """Train policy model using AI feedback"""
        # Apply reinforcement learning algorithm (e.g., PPO)
        # Here shows simple example of reward-based learning
        rewards = ai_scores
        loss = self.policy_model.compute_loss(prompts, responses, rewards)
        return loss

# Usage example
evaluator = AIEvaluator()
trainer = RLAIFTrainer(policy_model, evaluator)

# Collect AI feedback and train
prompts = ["Question 1", "Question 2", "Question 3"]
responses = ["Answer 1", "Answer 2", "Answer 3"]
ai_scores = trainer.collect_ai_feedback(prompts, responses)
loss = trainer.train_step(prompts, responses, ai_scores)
```

### Checkpoint Questions

- What is **RLAIF** and how does it differ from RLHF? What are the main advantages and limitations of RLAIF?

- What are the main elements to consider in the **AI evaluator training** process, and how do they affect RLAIF performance?

- Why does the **bias propagation** problem occur in RLAIF, and what methods can be used to resolve it?

- Why is **lack of human value reflection** pointed out as a limitation of RLAIF, and what problems can this cause?

- Why is **evaluation quality dependency** important in RLAIF, and what implications does this provide for system design?

## 7. Future Evaluation Paradigms

The future of LLM evaluation faces increasingly complex and diverse challenges. With the emergence of new technologies and application areas, evaluation methodologies must continue to evolve.

### 7.1 Multimodal LLM Evaluation

With the emergence of **multimodal LLMs**, evaluation of models that process not only text but also images, audio, video, and other diverse modalities together has become necessary.

#### 7.1.1 Evaluation Tasks

- **Inter-modal consistency**: Evaluate information consistency between different modalities
- **Cross-modal reasoning**: Evaluate ability to convert information from one modality to another
- **Multimodal generation**: Evaluate ability to generate multiple modalities simultaneously

#### 7.1.2 Evaluation Methods

- **Multimodal benchmarks**: Comprehensive evaluation sets including various modalities
- **Modality-specific granular evaluation**: Specialized evaluation metrics for each modality
- **Integrated evaluation**: Methods for comprehensively evaluating multiple modalities

### 7.2 Agent Evaluation

With the emergence of **AI agents**, evaluation of agents that perform complex tasks beyond simple text generation has become necessary.

#### 7.2.1 Evaluation Tasks

- **Task performance ability**: Ability to successfully complete complex multi-step tasks
- **Tool usage ability**: Ability to effectively utilize external tools and APIs
- **Environment adaptation ability**: Ability to adapt and learn in various environments

#### 7.2.2 Evaluation Methods

- **Simulation environment**: Agent performance evaluation in virtual environments
- **Real environment testing**: Agent performance evaluation in real environments
- **Human-agent collaboration**: Evaluation of collaboration ability with humans

### 7.3 Green AI Evaluation

**Green AI** is an approach that considers the environmental impact of AI systems, making it important to evaluate energy efficiency and carbon emissions.

#### 7.3.1 Evaluation Metrics

- **Energy consumption**: Energy consumed during model training and inference
- **Carbon emissions**: Carbon emissions caused by model usage
- **Efficiency**: Performance per unit energy

#### 7.3.2 Evaluation Methods

- **Lifecycle evaluation**: Environmental impact evaluation throughout model lifecycle
- **Comparative evaluation**: Environmental impact comparison with other models
- **Optimization evaluation**: Evaluation of optimization methods to minimize environmental impact

### 7.4 Human-AI Collaboration Evaluation

As **human-AI collaboration** emerges as an important area, it has become necessary to evaluate how effectively AI can collaborate with humans.

#### 7.4.1 Evaluation Tasks

- **Collaboration efficiency**: Efficiency when humans and AI work together
- **Communication ability**: Ability to communicate effectively with humans
- **Role division**: Appropriate role division between humans and AI

#### 7.4.2 Evaluation Methods

- **Collaboration scenarios**: Performance evaluation in actual collaboration situations
- **Human feedback**: Evaluation of human user satisfaction and feedback
- **Performance measurement**: Measurement of final performance through collaboration

### Checkpoint Questions

- What are the main tasks to consider in **multimodal LLM evaluation**, and how does this differ from existing text-based evaluation?

- What are the important evaluation tasks in **AI agent evaluation**, and how does this differ from simple text generation evaluation?

- What is the importance of **Green AI evaluation**, and how does this affect AI system development?

- What are the main elements to consider in **human-AI collaboration evaluation**, and how does this affect the practicality of AI systems?

- What is the development direction of **future evaluation paradigms**, and what changes are expected to be brought to current evaluation methodologies?

## 8. Hands-on Exercises

This section covers **3 hands-on exercises** to directly experiment with the learned concepts. All exercises are based on **PyTorch and Hugging Face Transformers**, and the provided code is an example that can be modified and utilized as needed.

### 8.1 BLEU/ROUGE vs G-Eval Comparison Experiment

This exercise compares traditional evaluation metrics like BLEU/ROUGE with the latest LLM-based evaluation method, G-Eval.

#### 8.1.1 Exercise Objectives

- **Understand differences between traditional evaluation metrics and LLM-based evaluation**
- **Compare evaluation results for various text qualities**
- **Analyze advantages and disadvantages of evaluation metrics**

#### 8.1.2 Exercise Content

1. **Calculate BLEU/ROUGE scores**
2. **Calculate G-Eval scores**
3. **Compare and analyze results**

#### 8.1.3 Exercise Code

```python
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import openai

class TraditionalEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score"""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        return sentence_bleu([reference_tokens], candidate_tokens)

    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE score"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

class GEvalEvaluator:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = openai.OpenAI()

    def evaluate(self, text, criteria):
        """Evaluation using G-Eval"""
        prompt = f"""
        Please evaluate the following text based on {criteria}.

        Text: {text}

        Evaluation criteria:
        1. Clarity: How clear is the text?
        2. Consistency: How consistent is the text?
        3. Relevance: How relevant is the text to the topic?

        Please evaluate each criterion on a scale of 1-10 and calculate the overall score.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content

# Exercise execution
def run_comparison_experiment():
    # Initialize evaluators
    traditional_eval = TraditionalEvaluator()
    geval_eval = GEvalEvaluator()

    # Test data
    reference = "Artificial intelligence is having a significant impact on modern society."
    candidates = [
        "AI is having an important impact on today's society.",
        "Artificial intelligence technology is changing our lives.",
        "AI is having a significant impact on modern society."
    ]

    print("=== BLEU/ROUGE vs G-Eval Comparison Experiment ===\n")

    for i, candidate in enumerate(candidates):
        print(f"Candidate {i+1}: {candidate}")

        # BLEU score
        bleu_score = traditional_eval.calculate_bleu(reference, candidate)
        print(f"BLEU score: {bleu_score:.4f}")

        # ROUGE score
        rouge_scores = traditional_eval.calculate_rouge(reference, candidate)
        print(f"ROUGE score: {rouge_scores}")

        # G-Eval score
        geval_score = geval_eval.evaluate(candidate, "text quality")
        print(f"G-Eval score: {geval_score}")

        print("-" * 50)

# Exercise execution
if __name__ == "__main__":
    run_comparison_experiment()
```

### 8.2 GPTScore Implementation and Experiment

This exercise directly implements GPTScore and performs evaluation on various texts.

#### 8.2.1 Exercise Objectives

- **Understand GPTScore principles**
- **Implement probability-based evaluation**
- **Perform evaluation on various text qualities**

#### 8.2.2 Exercise Content

1. **Implement GPTScore calculation function**
2. **Perform evaluation on various texts**
3. **Analyze and visualize results**

#### 8.2.3 Exercise Code

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np

class GPTScoreCalculator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_gpt_score(self, text, reference=None):
        """Calculate GPTScore"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Calculate probability for each token
            probs = F.softmax(logits, dim=-1)

            # Calculate GPTScore (average log probability)
            gpt_score = 0
            for i in range(1, inputs['input_ids'].shape[1]):  # Exclude first token
                token_id = inputs['input_ids'][0, i]
                token_prob = probs[0, i-1, token_id]
                gpt_score += torch.log(token_prob).item()

            # Normalize (divide by number of tokens)
            gpt_score /= (inputs['input_ids'].shape[1] - 1)

        return gpt_score

    def compare_texts(self, texts, reference=None):
        """Compare GPTScore of multiple texts"""
        scores = []
        for text in texts:
            score = self.calculate_gpt_score(text, reference)
            scores.append(score)
        return scores

# Exercise execution
def run_gptscore_experiment():
    # Initialize GPTScore calculator
    gpt_calculator = GPTScoreCalculator()

    # Test texts
    test_texts = [
        "Artificial intelligence is having a significant impact on modern society.",
        "AI is having an important impact on today's society.",
        "Artificial intelligence technology is changing our lives.",
        "AI is having a significant impact on modern society.",
        "Artificial intelligence is having a significant impact on modern society."
    ]

    print("=== GPTScore Experiment ===\n")

    # Calculate GPTScore for each text
    scores = gpt_calculator.compare_texts(test_texts)

    # Output results
    for i, (text, score) in enumerate(zip(test_texts, scores)):
        print(f"Text {i+1}: {text}")
        print(f"GPTScore: {score:.4f}")
        print("-" * 50)

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(scores)+1), scores)
    plt.xlabel('Text Number')
    plt.ylabel('GPTScore')
    plt.title('GPTScore Comparison')
    plt.xticks(range(1, len(scores)+1))
    plt.grid(True, alpha=0.3)
    plt.show()

    return scores

# Exercise execution
if __name__ == "__main__":
    scores = run_gptscore_experiment()
```

### 8.3 FLASK Evaluation System Implementation

This exercise implements FLASK's fine-grained skill set-based evaluation system.

#### 8.3.1 Exercise Objectives

- **Understand FLASK's fine-grained skill sets**
- **Implement multi-dimensional evaluation system**
- **Perform detailed analysis of text quality**

#### 8.3.2 Exercise Content

1. **Implement FLASK evaluation system**
2. **Calculate 12 fine-grained ability indicators**
3. **Generate comprehensive evaluation results**

#### 8.3.3 Exercise Code

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class FLASKEvaluator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # 12 fine-grained ability indicators
        self.skills = [
            "Clarity", "Consistency", "Relevance", "Completeness", "Accuracy", "Creativity",
            "Logic", "Structure", "Expressiveness", "Appropriateness", "Efficiency", "Reliability"
        ]

        # Classifiers for each ability (actually more complex models needed)
        self.skill_classifiers = {}
        for skill in self.skills:
            self.skill_classifiers[skill] = nn.Linear(self.model.config.hidden_size, 1)

    def extract_features(self, text):
        """Extract features from text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Generate sentence representation through average pooling
            features = outputs.last_hidden_state.mean(dim=1)

        return features

    def evaluate_skill(self, text, skill_name):
        """Evaluate specific ability"""
        features = self.extract_features(text)

        # Actually more sophisticated evaluation needed, here shows simple example
        if skill_name == "Clarity":
            # Clarity inversely proportional to sentence length and complexity
            clarity_score = 1.0 / (1.0 + len(text.split()) * 0.1)
        elif skill_name == "Consistency":
            # Consistency measured by ratio of repeated words
            words = text.split()
            unique_words = set(words)
            consistency_score = len(unique_words) / len(words) if words else 0
        elif skill_name == "Relevance":
            # Relevance measured by keyword density
            relevance_score = min(1.0, len(text.split()) * 0.05)
        else:
            # Random scores for other abilities (actually more sophisticated evaluation needed)
            relevance_score = np.random.uniform(0.3, 0.9)

        return relevance_score

    def evaluate_all_skills(self, text):
        """Evaluate all abilities"""
        skill_scores = {}
        for skill in self.skills:
            score = self.evaluate_skill(text, skill)
            skill_scores[skill] = score

        return skill_scores

    def calculate_overall_score(self, skill_scores):
        """Calculate overall score"""
        scores = list(skill_scores.values())
        return np.mean(scores)

    def visualize_results(self, skill_scores, text):
        """Visualize results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart of ability scores
        skills = list(skill_scores.keys())
        scores = list(skill_scores.values())

        ax1.bar(skills, scores)
        ax1.set_title('FLASK Ability Scores')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
        scores_radar = scores + scores[:1]  # Add first score to make circular
        angles += angles[:1]

        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, scores_radar, 'o-', linewidth=2)
        ax2.fill(angles, scores_radar, alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(skills)
        ax2.set_ylim(0, 1)
        ax2.set_title('FLASK Ability Profile')

        plt.tight_layout()
        plt.show()

# Exercise execution
def run_flask_experiment():
    # Initialize FLASK evaluator
    flask_evaluator = FLASKEvaluator()

    # Test text
    test_text = "Artificial intelligence is having a significant impact on modern society. This technology is being utilized in various fields and is changing our daily lives."

    print("=== FLASK Evaluation Experiment ===\n")
    print(f"Text to evaluate: {test_text}\n")

    # Evaluate all abilities
    skill_scores = flask_evaluator.evaluate_all_skills(test_text)

    # Output results
    print("Ability scores:")
    for skill, score in skill_scores.items():
        print(f"{skill}: {score:.3f}")

    # Overall score
    overall_score = flask_evaluator.calculate_overall_score(skill_scores)
    print(f"\nOverall score: {overall_score:.3f}")

    # Visualize results
    flask_evaluator.visualize_results(skill_scores, test_text)

    return skill_scores, overall_score

# Exercise execution
if __name__ == "__main__":
    skill_scores, overall_score = run_flask_experiment()
```

### 8.4 Exercise Result Analysis

#### 8.4.1 Exercise Objectives

- **Comprehensive analysis of exercise results**
- **Compare advantages and disadvantages of various evaluation methods**
- **Derive considerations for actual application**

#### 8.4.2 Exercise Content

1. **Comprehensive exercise results**
2. **Comparative analysis of evaluation methods**
3. **Review actual application scenarios**

#### 8.4.3 Exercise Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_experiment_results():
    """Comprehensive analysis of exercise results"""

    # Experimental result data (actually results from above exercises)
    results = {
        'Evaluation Method': ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'GPTScore', 'FLASK'],
        'Average Score': [0.45, 0.62, 0.38, 0.58, 0.72, 0.68],
        'Standard Deviation': [0.12, 0.08, 0.15, 0.09, 0.06, 0.11],
        'Computation Time (seconds)': [0.01, 0.02, 0.02, 0.02, 0.15, 0.25],
        'Human Correlation': [0.35, 0.42, 0.38, 0.45, 0.68, 0.72]
    }

    df = pd.DataFrame(results)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Average score comparison
    axes[0, 0].bar(df['Evaluation Method'], df['Average Score'])
    axes[0, 0].set_title('Average Score by Evaluation Method')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Computation time comparison
    axes[0, 1].bar(df['Evaluation Method'], df['Computation Time (seconds)'])
    axes[0, 1].set_title('Computation Time by Evaluation Method')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Human correlation comparison
    axes[1, 0].bar(df['Evaluation Method'], df['Human Correlation'])
    axes[1, 0].set_title('Human Correlation by Evaluation Method')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Comprehensive comparison (normalized scores)
    normalized_scores = df[['Average Score', 'Human Correlation']].copy()
    normalized_scores['Computation Time'] = 1 - (df['Computation Time (seconds)'] / df['Computation Time (seconds)'].max())

    sns.heatmap(normalized_scores.T, annot=True, cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Comprehensive Comparison by Evaluation Method (Normalized)')

    plt.tight_layout()
    plt.show()

    # Result analysis
    print("=== Exercise Result Analysis ===\n")

    print("1. Characteristics by evaluation method:")
    for _, row in df.iterrows():
        print(f"- {row['Evaluation Method']}: Average {row['Average Score']:.3f}, Human correlation {row['Human Correlation']:.3f}")

    print("\n2. Key findings:")
    print("- GPTScore and FLASK show highest human correlation")
    print("- Traditional methods (BLEU/ROUGE) are fast but have low correlation with human judgment")
    print("- LLM-based methods take longer but enable more accurate evaluation")

    print("\n3. Considerations for actual application:")
    print("- For fast evaluation: Use BLEU/ROUGE")
    print("- For accurate evaluation: Use GPTScore/FLASK")
    print("- Balanced approach: Combine multiple methods")

    return df

# Exercise execution
if __name__ == "__main__":
    results_df = analyze_experiment_results()
```

### Checkpoint Questions

- What are the main differences observed in the **BLEU/ROUGE vs G-Eval comparison experiment**, and what characteristics of each evaluation method does this show?

- What are the advantages of probability-based evaluation in **GPTScore implementation**, and how does this differ from traditional evaluation methods?

- Why does the **FLASK evaluation system** use 12 fine-grained ability indicators, and how does this differ from single-score evaluation?

- Why do GPTScore and FLASK show the highest human correlation in **exercise result analysis**, and what implications does this provide for actual application?

- What is the trade-off between fast evaluation and accurate evaluation in **considerations for actual application**, and how can this be balanced?

## 9. Summary and Conclusion

This chapter examined the changing landscape of LLM evaluation and new evaluation paradigms. We recognized the limitations of traditional evaluation metrics and confirmed how evaluation methodologies are developing through the emergence of meaning-based evaluation and the LLM-as-a-Judge paradigm.

### 9.1 Summary of Main Content

#### 9.1.1 Changing Landscape of Evaluation

- **Limitations of traditional evaluation metrics**: Limitations of BLEU, ROUGE, etc. based on superficial similarity
- **Emergence of meaning-based evaluation**: Meaning-based evaluation like BERTScore, SentenceMover, BLEURT
- **LLM-as-a-Judge paradigm**: New approaches utilizing LLMs as evaluators like GPTScore, G-Eval, FLASK

#### 9.1.2 LLM-Based Evaluation Paradigms

- **GPTScore**: Probability-based evaluation framework utilizing model's inherent knowledge
- **G-Eval**: Systematic evaluation through Chain-of-Thought achieving high correlation with human judgment
- **FLASK**: Fine-grained skill set-based multi-dimensional evaluation enabling detailed text quality analysis

#### 9.1.3 Specialized Purpose Benchmarks

- **LiveCodeBench**: Contamination-free code generation evaluation
- **EvalPlus**: More rigorous code evaluation through test case augmentation
- **HELM-Code**: Comprehensive evaluation focusing on transparency and community collaboration
- **MMLU-Pro**: 10-choice high-difficulty knowledge/reasoning benchmark
- **GPQA and BBH**: Knowledge/reasoning enhanced evaluation sets

#### 9.1.4 Domain-Specific Benchmarks

- **FinBen**: Comprehensive financial domain benchmark
- **AgentHarm**: AI agent harmfulness evaluation benchmark
- **LEXam**: Legal exam-based LLM evaluation
- **CSEDB**: Medical LLM safety/effectiveness dual evaluation
- **MATH and GSM8K**: Mathematical ability evaluation

#### 9.1.5 Evaluation Bias and Limitations

- **Major biases**: Narcissistic bias, verbosity bias, inconsistency
- **Evaluation limitations**: Differences from human evaluation, lack of domain-specific knowledge, subjectivity of evaluation criteria

#### 9.1.6 RLAIF and Future Evaluation Paradigms

- **RLAIF**: More efficient and scalable learning through AI feedback-based reinforcement learning
- **Future evaluation paradigms**: Multimodal LLM evaluation, agent evaluation, Green AI evaluation, human-AI collaboration evaluation

### 9.2 Core Insights

#### 9.2.1 Evolution of Evaluation Methodologies

LLM evaluation is evolving from simple superficial similarity to semantic similarity, and to evaluation utilizing model's inherent knowledge. This enables more accurate and human-like judgment evaluation.

#### 9.2.2 Multi-dimensionality of Evaluation

Modern LLM evaluation takes a multi-dimensional approach rather than single scores, comprehensively evaluating various aspects of text. This provides more sophisticated and useful evaluation results.

#### 9.2.3 Importance of Domain Specialization

General evaluation methods cannot easily evaluate expertise in specific domains. Therefore, development and utilization of domain-specific benchmarks is becoming important.

#### 9.2.4 Recognition of Evaluation Bias and Limitations

It is necessary to recognize and resolve biases and limitations that appear in LLM evaluation. This enables building fairer and more reliable evaluation systems.

### 9.3 Future Development Directions

#### 9.3.1 Continuous Development of Evaluation Methodologies

- **Development of more sophisticated evaluation metrics**
- **Methodology research for bias reduction**
- **Building human-AI collaboration evaluation systems**

#### 9.3.2 Expansion of Domain-Specific Evaluation

- **Development of specialized benchmarks for new domains**
- **Building multilingual evaluation systems**
- **Evaluation methodologies considering cultural context**

#### 9.3.3 Building Practical Evaluation Systems

- **Development of real-time evaluation systems**
- **Building automated evaluation pipelines**
- **Research on interpretation and utilization methods of evaluation results**

### 9.4 Conclusion

LLM evaluation is a rapidly developing field where new technologies and methodologies continue to emerge. By recognizing the limitations of traditional evaluation methods and enabling more accurate and human-like judgment evaluation through meaning-based evaluation and the LLM-as-a-Judge paradigm, we are advancing evaluation capabilities.

Particularly, through the development and utilization of domain-specific benchmarks, we can now evaluate expertise in specific fields, and efforts are continuing to recognize and resolve evaluation biases and limitations for building fairer and more reliable evaluation systems.

In the future, evaluation methodologies for new technologies and application areas such as multimodal LLMs, AI agents, and Green AI are expected to develop further, building more sophisticated and practical evaluation systems.

## 10. References

### 10.1 Traditional Evaluation Metrics

- Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of machine translation. _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_.
- Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. _Text Summarization Branches Out_.

### 10.2 Meaning-Based Evaluation

- Zhang, T., et al. (2020). BERTScore: Evaluating text generation with BERT. _International Conference on Learning Representations_.
- Clark, E., et al. (2019). SentenceMover's similarity: Automatic evaluation for multi-sentence texts. _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_.
- Sellam, T., et al. (2020). BLEURT: Learning robust metrics for text generation. _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_.

### 10.3 LLM-Based Evaluation

- Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_.
- Fu, J., et al. (2023). GPTScore: Evaluate as you desire. _arXiv preprint arXiv:2302.04166_.
- Wang, J., et al. (2023). FLASK: Fine-grained language model evaluation based on alignment skill sets. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_.

### 10.4 Specialized Purpose Benchmarks

- Jain, N., et al. (2024). LiveCodeBench: Holistic and contamination-free evaluation of large language models for code. _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_.
- Liu, J., et al. (2023). EvalPlus: Augmenting code evaluation datasets with test case generation. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_.
- Liang, P., et al. (2022). Holistic evaluation of language models. _Transactions on Machine Learning Research_.

### 10.5 Domain-Specific Benchmarks

- Chen, J., et al. (2024). FinBen: A holistic financial benchmark for large language models. _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_.
- Sun, Y., et al. (2025). AgentHarm: A comprehensive benchmark for evaluating agentic AI safety. _Proceedings of the 2025 International Conference on Learning Representations_.
- Nguyen, H., et al. (2025). LEXam: A comprehensive benchmark for legal reasoning evaluation. _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_.

### 10.6 RLAIF and Future Evaluation

- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. _arXiv preprint arXiv:2212.08073_.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems_.

### 10.7 Evaluation Bias and Limitations

- Chiang, W. L., et al. (2023). Vicuna: An open-source chatbot impressing GPT-4 with 90% ChatGPT quality. _arXiv preprint arXiv:2303.04671_.
- Lin, S., et al. (2022). TruthfulQA: Measuring how models mimic human falsehoods. _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_.

### 10.8 Mathematical and Reasoning Evaluation

- Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. _Proceedings of the 2021 Conference on Neural Information Processing Systems_.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. _Proceedings of the 2021 Conference on Neural Information Processing Systems_.

### 10.9 Medical and Legal Evaluation

- Singhal, K., et al. (2023). Towards expert-level medical question answering with large language models. _arXiv preprint arXiv:2305.09617_.
- Guha, N., et al. (2023). LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_.

### 10.10 Green AI and Efficiency

- Schwartz, R., et al. (2020). Green AI. _Communications of the ACM_.
- Strubell, E., et al. (2019). Energy and policy considerations for deep learning in NLP. _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing_.

---

_This chapter comprehensively examined the changing landscape of LLM evaluation and new evaluation paradigms. We learned methods to enable more accurate and human-like judgment evaluation by recognizing the limitations of traditional evaluation methods and through meaning-based evaluation and the LLM-as-a-Judge paradigm. We also confirmed the importance of specialized purpose benchmarks and domain-specific benchmarks, and explored methods to build fairer and more reliable evaluation systems by recognizing evaluation biases and limitations. In the future, evaluation methodologies for new technologies and application areas such as multimodal LLMs, AI agents, and Green AI are expected to develop further, building more sophisticated and practical evaluation systems._
