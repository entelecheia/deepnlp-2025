# Team Project Guidelines

## 1. Overview and Objectives

The team project provides students with hands-on experience in solving real-world problems by applying deep learning-based natural language processing (NLP) techniques learned throughout the semester. Each team consists of approximately 4 students who collaborate to train large language models (LLMs) on real data and develop application agents that utilize these models. Through this process, students integrate theoretical learning to implement complete NLP models and applications while developing problem-solving abilities and project management capabilities through team collaboration.

This guideline presents project execution methods and evaluation criteria to facilitate smooth implementation. It outlines requirements and deliverables for project execution, including midterm and final presentation requirements. Students should refer to this guideline to proceed with their projects systematically and produce creative yet achievable results.

**Checkpoint Questions:**

- What are the main objectives of the team project?
- How does this project integrate theoretical learning with practical implementation?
- What key skills will students develop through this project?

## 2. Project Topics and Goals

Each team should autonomously select their project topic, which must focus on solving natural language processing-related problems through LLMs. The project goal is to train a language model specialized for the chosen problem (through fine-tuning or additional pre-training) using the team's selected domain dataset and base LLM model, then implement it as a functioning LLM-based agent. Examples include an LLM for summarizing Korean legal documents, a medical consultation chatbot, or a knowledge search agent based on user queries. When selecting topics, teams should consider their areas of interest, data availability, model scale, and difficulty level.

Project goals should be defined as specifically and measurably as possible. This helps clarify the project scope and establish evaluation criteria. Each team should clearly describe "what problem to solve and how," and plan in advance what metrics to use for evaluating model performance. The chosen topic must be achievable within one semester with limited resources, so setting realistic scope is important. Teams should select topics that can achieve improved performance or new functionality compared to existing methods by utilizing various techniques covered in class (e.g., PEFT-based fine-tuning, RAG, DPO, etc.). After topic selection, teams should obtain approval from the supervising professor and pursue a balance between creativity and practicality.

**Checkpoint Questions:**

- What criteria should teams consider when selecting their project topic?
- How should project goals be defined to ensure successful completion?
- What techniques learned in class can be applied to improve model performance?

## 3. Team Composition and Role Division

Teams are composed of approximately 4 members, with students of diverse capabilities balanced as much as possible. Within teams, roles can be divided into project leader, data manager, model manager, and agent implementation manager. For example, one person can serve as project manager handling schedule management and overall coordination, another as data engineer specializing in data collection and preprocessing, another as model architect selecting models and building training pipelines, and the last as agent developer implementing user interfaces or system integration. Role division should be determined autonomously based on team members' strengths, ensuring all team members understand and participate in the overall project.

Active participation of all team members is essential. Each member should take responsibility for their role while sharing progress through regular team meetings and solving problems together. Teams should use collaboration tools such as messengers, email, Notion/Google Docs for smooth communication. Since all team members must participate in midterm and final presentations, equal contribution should be made in preparing presentation materials and practice sessions. If any team member is overloaded with specific tasks, roles should be readjusted or other team members should help maintain collaboration and balance. While team performance evaluation is applied equally, individual scores may be adjusted based on individual contribution when necessary. If problems arise or team conflicts occur, early consultation with the supervising professor should be sought to find solutions.

**Checkpoint Questions:**

- What are the key roles that should be assigned within a team?
- How can teams ensure equal participation from all members?
- What strategies can teams use to maintain effective collaboration?

## 4. Development Environment and Resource Conditions

The school provides high-performance GPU server environments for this project. Each team can use one NVIDIA H100 GPU (SXM, 80GB HBM memory) to perform large-scale model training. The H100 GPU is optimized for large-scale computation with the latest architecture, enabling efficient training of models with billions of parameters. However, since GPU resources allocated per team are limited, it is important to efficiently utilize GPU memory and computation time. Strategies such as using Mixed Precision (FP16/BF16), adjusting Gradient Accumulation, optimizing batch size, and reducing memory usage through PEFT techniques like LoRA/QLoRA learned in class are recommended. Rather than fine-tuning entire large models from start to finish, maximum performance should be extracted within limited resources through partial fine-tuning or reusing pre-trained weights.

The software environment is based on PyTorch and Hugging Face Transformers libraries, and frameworks covered in class such as DSPy, Haystack, and CrewAI can be used as needed. The development server has major deep learning frameworks and libraries installed, and teams work in assigned computing accounts or container environments. All experiments should be conducted in the provided server environment as a principle, and external cloud or personal equipment should only be used with prior consultation for special reasons. Each team will be allocated a certain amount of storage for data storage and experiment result backup. Important model weights or deliverables should be backed up regularly, and code version management should be thorough through GitHub repositories. When using GPUs, responsible usage should be demonstrated by following scheduled times or allocated resources for fairness with other teams, and terminating processes after use to return resources.

**Checkpoint Questions:**

- What computational resources are available for this project?
- How can teams optimize GPU memory usage during model training?
- What best practices should teams follow for code version management?

## 5. Schedule and Deliverables

The project proceeds according to a systematic schedule plan from the beginning of the semester to the final presentation. Major milestones and deliverables are as follows:

- Weeks 1-2: Complete team formation and project topic selection. Team members brainstorm ideas and create a list of possible topics, then consult with the supervising professor to finalize the topic. Deliverables: Team roster and project topic proposal (submit simple overview).

- Weeks 3-4: Data collection and base model selection phase. Secure or build datasets suitable for the chosen topic and decide on the base language model to use (e.g., Llama 2, KoGPT, BERT, etc.). Perform data preprocessing and initial exploratory analysis, and test run the selected model with small sample data. Deliverables: Dataset documentation (data source, size, characteristics) and model selection report (reasons for selection and expected advantages/disadvantages).

- Weeks 5-6: Model training preparation and prototype implementation. Complete code writing for full-scale model fine-tuning or pre-training and hyperparameter settings. Also implement a simple prototype of the LLM-based agent (e.g., simple dialogue scenario processing test). Identify issues arising at this stage and seek solutions. Deliverables: Training script draft, agent prototype (model output samples for example inputs).

- Week 7: Midterm presentation preparation. Organize project plans and create midterm presentation slides. Each team member reflects their responsible part in the slides and refines delivery content through presentation practice. Simultaneously, start some model training and include intermediate results (e.g., initial validation set performance) in presentation materials if obtained. Deliverables: Midterm presentation PPT file draft.

- Week 8: Midterm project plan presentation (replaces midterm exam). Teams present and receive feedback (detailed requirements refer to section 6 below). Deliverables: Final midterm presentation slides (PPT) submission.

- Weeks 9-12: Model development and agent completion phase. Complete full-scale model training/fine-tuning by reflecting midterm presentation feedback. Continuously evaluate performance during training and perform data augmentation, parameter adjustment, etc., as needed. Simultaneously improve LLM agent functionality and build user interface or demo demonstration environment. During this period, conduct regular team meetings to check progress and solve problems as they arise. Deliverables: Trained final model (weight files), initial version of agent application.

- Weeks 13-14: Testing and final presentation preparation. Validate the completed model with various test cases to produce performance metrics (e.g., accuracy, F1, BLEU, etc., appropriate for the project). Perform comparative evaluation with other public models or existing methods to interpret the meaning of results. Stabilize agent operation and prepare scenarios needed for demonstration demos. Begin final report writing and create final presentation slides. Deliverables: Test result summary, final presentation slide draft, project report draft.

- Week 15: Final presentation and result submission (replaces final exam). Teams present final results and demonstrate LLM agent demos. After presentation, submit all final deliverables (detailed requirements refer to section 7 below). Deliverables: Final presentation slides (PPT), final project report (document file), GitHub code repository link and usage instructions.

The above schedule is recommended, and detailed plans may vary by team. What's important is adhering to the milestones of midterm and final presentation weeks. Each team should create and manage their own detailed schedule, and major deliverables must be submitted by the deadline. Deliverable submission will be conducted through LMS upload or email submission, as announced later.

**Checkpoint Questions:**

- What are the key milestones in the project timeline?
- What deliverables are required for each phase of the project?
- How should teams manage their schedule to meet all deadlines?

## 6. Midterm Presentation Requirements

The midterm project presentation is an opportunity for each team to share their planned project and progress, conducted during the midterm exam period. Presentation time is approximately 10 minutes per team, with an additional 5 minutes for Q&A (total within 15 minutes). All team members should participate evenly in the presentation, with each person speaking for 2-3 minutes or more recommended. The midterm presentation should clearly convey the following content:

- Project Overview: Problem definition and importance of the chosen topic. Explain what question or task to solve and why it's important, providing background. If necessary, briefly mention existing research or cases in related fields to clarify the project's motivation.

- Dataset Introduction: Explain the source, scale, and characteristics of the data to be used. Present input/output formats and quality included in the data, preprocessing plans, potential problems (e.g., class imbalance, noise, etc.), and response measures.

- Base Model and Techniques: Identify the base LLM model to be used for fine-tuning or pre-training and logically present the reasons for selection. Also explain the training techniques to be applied. Specifically mention which of the PEFT techniques (e.g., LoRA), RLHF series techniques (e.g., DPO), prompt strategies, etc., learned in class will be utilized. It's good to show that the team sufficiently understands the model structure or algorithmic characteristics.

- Project Implementation Plan: Present the development schedule and task division plan for the remaining period. Share schedules of when and what to complete for each major stage (data preparation → model training → agent integration → testing/evaluation → report writing). Using role division tables or Gantt charts for visual presentation is effective. Briefly explain each team member's responsibilities and contributions so far, showing that everyone is performing their roles.

- Progress to Date: Report completed work and initial achievements up to the midterm point. This can include data exploration results, test results on small samples of the model, or demo videos/screenshots of agent prototypes. Share interesting results discovered in initial experiments or problems faced to seek audience feedback.

- Expected Difficulties and Countermeasures: Honestly identify expected challenges in completing the project (model training time issues, data shortage, performance limitations, etc.) and explain the team's strategy to solve them. If additional support or advice is needed from professors or mentors, this can be mentioned at this time.

Midterm presentation materials (PPT) should be concise while containing core information, using diagrams, figures, and tables rather than text to aid understanding. Structure the flow so listeners can easily grasp the project's intent and plan, use professional terms only when necessary and explain as simply as possible. After the presentation, sincerely listen to and record feedback from professors and colleagues to reflect in subsequent project progress. Since the midterm presentation has the nature of process evaluation toward final results, solid planning and preparation are evaluated more than perfect results. Presentation attitude, time compliance, and team coordination are reflected in evaluation, so rehearsal should be conducted to control presentation time and improve delivery effectiveness.

**Checkpoint Questions:**

- What are the key components that must be included in the midterm presentation?
- How should teams structure their presentation to effectively communicate their project plan?
- What feedback should teams seek during the midterm presentation?

## 7. Final Presentation and Submission Requirements

The final project presentation is conducted during the final exam period at the end of the semester, with approximately 20 minutes of presentation and 10 minutes of Q&A per team (30 minutes total allocated per team). All team members must participate in the presentation, dealing with more in-depth content and results than the midterm presentation. The final presentation should systematically convey content covering the entire project process. Major elements to be included in the presentation are:

- Problem Definition and Approach Reintroduction: Briefly reintroduce the project's topic and goals to help audience understanding (as some may not have attended the midterm presentation). Summarize the initially set problem and solution approach in one or two slides and provide an overview of the project's overall structure.

- Data and Model Methodology: Explain in detail the final state of the dataset used (data size after preprocessing, etc.) and model architecture/training settings. For example, share specific setting values such as "fine-tuned KoGPT-6B model with Korean news summarization data of 500,000 sentences using LoRA technique, learning rate 2e-5, 3 epochs," model structure or fine-tuning strategy, and time/epoch information spent on training. Also mention any special considerations or tuning know-how during model training (e.g., techniques to prevent collapse, memory optimization application, etc.).

- LLM Agent Implementation: Introduce how the agent developed using the trained model operates. Present a system architecture diagram showing how user input is processed by the model and how model output is reflected in the application. If there are special features such as external tool or API integration, or multimodal input, explain these. Demonstrate the agent's usage scenarios with examples, showing actual operation screens or result predictions in demo format. If possible, conducting a live demo during the on-site presentation to show interaction with the agent is recommended. (However, prepare videos or screenshots as backup for unexpected errors during live demos)

- Performance Evaluation Results: Present the final model's performance evaluated with various metrics. Report how much the model achieved compared to the success criteria set in the project goals and present comparison results with existing models or baselines. Visualize appropriate metrics such as accuracy, precision/recall, BLEU, ROUGE, or human evaluation scores with tables or graphs. Don't just list performance numbers but add interpretation and analysis of results. For example, perform advantages/disadvantages evaluation such as "our model excelled in controlling summary sentence length compared to existing methods but showed limitations in factual accuracy." Error analysis is also important, so selecting several representative failure cases and providing error case analysis adds depth to the project.

- Conclusion and Future Tasks: Summarize the overall project results while self-evaluating how much achievement was made compared to initial goals. Honestly describe expected parts, lessons learned, and limitations. Furthermore, suggest directions for future improvement or expansion. For example, add developmental discussion such as "performance improvement is expected by expanding data or combining multimodal elements," "future user evaluation is needed to verify the model's usefulness." Briefly mention lessons learned or feelings from team collaboration and conclude.

After the final presentation, all final project deliverables must be submitted. Items to be submitted are:

- Presentation Slide File: PPT or PDF file used for the final presentation. (Updated final version compared to midterm presentation slides)

- Project Report: Final report document containing project progress and results. The report is generally written in about 10 pages and should preferably include the following items: Introduction (background and objectives), Data and Methodology, Experiments and Results, Analysis and Discussion, Conclusion, References. The report should describe more detailed content than presentation materials, implementation details, code explanations, experimental environment (e.g., hardware specifications, library versions), etc. Especially if other research or open source is utilized, sources should be specified and included in references.

- GitHub Repository Link: Submit the URL of the team's code repository. The repository should be equipped with source code, model training scripts, README.md (including usage instructions), license and reference materials, etc. For code reproducibility, important hyperparameter settings or execution methods should be specified in README, and if the dataset can be made public, provide some samples or download paths. (If data cannot be made public due to copyright reasons, only describe usage methods in the report)

- Other Deliverables: Final model weight files obtained from model training may be omitted from submission if the file size is large, but teams should store them safely as professors may request them for result verification when necessary. Also, agent demo videos (if live demo was not conducted or prepared as supplementary material) can be submitted together.

Final deliverables must all be submitted within the specified deadline after the final presentation (e.g., by midnight on the presentation day), and late submission may result in point deduction or non-submission treatment. Submission methods and detailed formats will be announced later, generally in the form of LMS upload or email submission. The final presentation and deliverable submission are the culmination of the project, comprehensively evaluating all efforts and results, so all requirements should be faithfully satisfied.

**Checkpoint Questions:**

- What are the key components of the final presentation?
- What deliverables must be submitted after the final presentation?
- How should teams prepare for the final presentation and demo?

## 8. Evaluation Criteria

This team project evaluation consists of 30% midterm presentation, 50% final results, and 20% attendance and participation, totaling 100%. Detailed evaluation indicators are as follows:

- Attendance and Participation (20%): Evaluated based on class attendance rate throughout the semester and activeness in class and team activities. Factors such as frequent absences or tardiness, attitude toward participating in class discussions or Q&A, and workshop practice participation are comprehensively reflected. For team projects, collaboration contribution within the team may also be partially included in participation. For this purpose, team member feedback or self-evaluation may be received frequently during project progress, and if a specific student's contribution is extremely low, it may affect the participation score. It's important to show sincere attitude and communication efforts.

- Midterm Presentation (30%): Evaluation score for the project plan presentation conducted during the midterm exam period. Evaluation items include clarity of planning, appropriateness and creativity of topics, specificity and feasibility of plans, validity of data/model selection, and team members' presentation skills and preparation. For example, whether the problem definition is clear and distinct, whether goals and plans are realistic with well-established step-by-step strategies, whether efforts to utilize content learned in class are visible, whether all team members participate in presentation and sincerely answer questions, etc. Since midterm presentation evaluation also has the nature of feedback for improving subsequent projects, attention should be paid to comments as well as scores to supplement insufficient parts.

- Final Project Results (50%): Comprehensive evaluation score for final presentation and submitted deliverables. This includes final presentation content (presentation materials and delivery effectiveness), completeness of project deliverables (model performance and innovation, agent implementation quality), report thoroughness, code quality and reproducibility, Q&A response, etc. Specifically, the difficulty and challenge level of the problem to be solved, appropriateness and creativity of the approach used, goal achievement level (whether model performance approaches or achieves initial goals), depth of result analysis and interpretation, clarity and logical development of presentation, actual implementation level through demos, etc., are major evaluation factors. Additionally, harmony and effort balance among team members may be qualitatively judged. Submitted project reports are evaluated based on content accuracy, completeness, document structure, reference notation, etc., and GitHub code is also reflected in evaluation based on execution ease, comment and README organization, version management utilization, etc. The 50% final result score is basically given equally to all team members, but individual scores may be adjusted when necessary based on the aforementioned participation or separate peer evaluation.

In summary, evaluation criteria encompass not only the excellence of deliverables themselves but also the thoroughness of the process and teamwork. Even if innovative and difficult challenges were attempted but goals were not perfectly achieved, learning and creativity in the attempt process can be highly evaluated, while conversely, selecting easy topics and only achieving high performance may be a deduction factor in terms of challenge spirit. Therefore, it's important to well appeal in the final presentation and report what intention the team project was designed with and what efforts were made.

**Checkpoint Questions:**

- What are the three main components of the project evaluation?
- How are individual contributions evaluated within team performance?
- What factors can lead to point deductions in the evaluation?

## 9. Important Considerations

The following matters should be noted and followed during project execution:

- Ethical AI and Responsible Research: AI ethics standards should be kept in mind from data collection to model development and result utilization during project progress. If data contains personal information, measures such as de-identification must be taken and usage permission must be obtained. Care should be taken to ensure that outputs generated during model training do not contain socially biased or harmful content, and potential misuse possibilities when the final agent interacts with users should also be considered. Refer to AI ethics and regulatory content (EU AI Act, etc.) covered in class to consider responsible AI implementation plans from the project design stage.

- Learning Material Utilization and Plagiarism Prohibition: Utilizing or expanding content practiced in workshops or code covered in final assignments for projects is encouraged. However, when utilizing publicly available code or pretrained models written by others, sources must be specified. References (papers, blogs, GitHub repositories, etc.) should be disclosed in reports and code comments to avoid plagiarism disputes. Submitted code and reports will undergo plagiarism checks, and if misconduct is detected, strict measures will be taken according to school regulations. Be careful not to submit code not written by oneself as one's own, and as it's a team project, all team members should understand the content and implementation.

- Efficient Time Management: LLM model training and agent development may take more time than expected. Sufficient time should be allocated for each stage including large-scale data processing, model fine-tuning, parameter exploration, error debugging, and result organization. Working intensively just before deadlines may cause quality degradation and unexpected errors leading to presentation problems. Especially since GPU resources are limited, teams should plan training schedules systematically. If necessary, flexibly operate schedules by team members taking turns using night or weekend hours, and prepare intermediate backups for important training to prepare for unexpected accidents.

- Communication and Collaboration: Active communication among team members is the key to project success. Regularly share progress and help each other when difficulties arise. Ensure all team members know the full scope of the project, and don't let only a few specific people write all code or make decisions. Regular meetings should be held at least once a week to check issues, and it's recommended to keep simple meeting minutes.

- Technical Considerations: When working in the provided development environment, pay special attention to data backup and environment setting storage. Store important experiment results separately and manage code change history through Git to prevent accidentally losing or damaging code. If new libraries or tools need to be installed, check compatibility with other team members and utilize tools already verified in class when possible. If operations exceed GPU memory, the kernel may crash, so check memory usage before experiments and adjust token length or batch size when necessary. Also, when preparing final demos, check internet connection or API usage in advance to prepare for operation in offline environments. For example, if internet connection is absolutely necessary for demonstration, prepare backup videos as Plan B.

- Presentation and Document Writing: Time compliance during presentation is essential, and each person's allocated portion should be confirmed through practice. Slides should be composed with consistent templates, sufficiently large fonts, and easy-to-see diagrams, pursuing organized design through team agreement on colors or fonts. Presenters should make eye contact with the audience and speak clearly, and when questions arise, listen carefully and answer sincerely after consulting with team members. Reports should be written in formal technical report format and proofread to avoid spelling and grammar errors. Also, add references when necessary to specify theoretical foundations or tool references of the project.

- Other Matters: Final deliverables are not utilized for purposes other than education, and excellent projects may be shared through department websites or report collections. If photos or videos were taken during project execution, they can be utilized for future department events, so it's good for teams to keep records. Finally, work with passion within the range that doesn't harm health, but complete the project enjoyably based on consideration and respect among team members. If the project is completed while following various considerations, valuable experience of taking one step closer to solving real-world problems can be gained through this team assignment. I hope everyone achieves good results.

**Checkpoint Questions:**

- What ethical considerations should teams keep in mind during project execution?
- How can teams ensure efficient time management throughout the project?
- What technical considerations are important when working in the development environment?
