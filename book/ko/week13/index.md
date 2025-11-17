# Week 13: 온톨로지와 AI - 현실을 모델링하고 AI로 작동시키기

## **서론: AI 시대를 위한 새로운 "존재론적 문해력(Ontological Literacy)"**

인공지능(AI)이 사회 전반의 "운영 체제"로 자리 잡으면서, AI를 바라보는 새로운 관점이 요구된다. 본 강의 노트는 "온톨로지와 AI: 현실을 모델링하고 AI로 작동시키기"라는 16주 강의 계획안의 핵심 철학과 개념을 심층적으로 해설한다. 이 과정의 궁극적인 목표는 단순한 AI 기술의 이해가 아닌, "존재론적 문해력(Ontological Literacy)"의 함양이다.

"존재론적 문해력"이란 우리가 속한 현실 세계의 복잡한 구조와 규칙을 인식하고, 이를 명시적인 모델(즉, 온톨로지)로 정의하며(Modeling Reality), AI가 그 모델 위에서 지능적으로 행동하고 현실을 작동시키도록(Operating it with AI) 설계하는 전략적 역량을 의미한다.

본 강의 노트는 이 "존재론적 문해력"을 함양하는 과정을 단계별로 심층 해설한다. 이는 "데이터"에서 "통찰"을 얻는 전통적 관점을 넘어, "통찰"을 "행동"으로, "예측"을 "작동"으로 전환하는 "의사결정 과학(Decision Science)"과 "온톨로지-퍼스트(Ontology-First)" 철학을 기반으로 한다.

## 1. 패러다임의 전환 - 예측을 넘어 행동으로 (The Paradigm Shift)

### **1.1. AI의 마지막 마일: "데이터 과학"에서 "의사결정 과학"으로**

#### **핵심 문제: "데이터 풍부, 결정 빈곤(Data-Rich, Decision-Poor)"**

현대의 많은 조직은 방대한 데이터를 축적하고 있음에도 불구하고, 그 데이터로부터 실제적이고 더 나은 "결정"을 내리는 데 어려움을 겪는 "데이터 풍부, 결정 빈곤(Data-Rich, Decision-Poor)"이라는 역설적인 상황에 직면해 있다. 수많은 "통찰(Insight)"이 도출되지만, 이것이 실제 비즈니스 "행동(Action)"이나 "성과(Outcome)"로 이어지지 못하는 "마지막 마일(Last Mile)"의 간극이 심각한 문제로 대두되고 있다. 이러한 간극은 "데이터 과학(Data Science, DS)"과 "의사결정 과학(Decision Science, DSci)"의 근본적인 차이에서 비롯된다.

#### **"데이터 과학(DS)"의 정의와 한계: 예측(Prediction)의 영역**

데이터 과학(DS)은 본질적으로 "기술적(technical)"이고 "알고리즘적(algorithmic)"인 측면에 중점을 둔다. 데이터 과학의 핵심 목표는 대규모 데이터를 수집, 처리, 분석하여 숨겨진 패턴을 발견하고 미래를 "예측(Prediction)"하며, 이를 통해 "통찰(Insight)"을 생성하는 것이다. 비유하자면, 데이터 과학자는 "데이터가 무엇을 말하는가?"라는 질문에 답하며, 항공기 조종석의 복잡하고 정교한 "계기판(Dashboard)"을 만드는 사람이다.

하지만 데이터 과학의 산출물인 대시보드, 예측 모델, 분석 리포트는 그 자체로 "행동"이 아니다. 아무리 훌륭한 데이터와 분석 결과가 제공되어도, 최종적으로 "사람"이 이를 해석하고 결정을 내려야 한다. 그러나 이 과정에서 인간의 "인지 편향(Cognitive Biases)", 데이터 해석의 복잡성, 조직의 관성 등이 개입하여 데이터 기반의 합리적인 결정이 실제 행동으로 이어지지 못하는 "마지막 마일"의 병목 현상이 발생한다.

#### **"의사결정 과학(DSci)"의 정의와 목표: 행동(Action)과 임팩트(Impact)**

"의사결정 과학(DSci)"은 바로 이 "마지막 마일"의 간극을 메우기 위한 학문이다. 의사결정 과학은 데이터 과학의 산출물이 조직의 실질적인 "임팩트(Impact)"로 이어지도록 보장하는 모든 활동을 포괄한다. 이는 데이터 과학의 기술적 역량을 기본으로 하되, 여기에 **비즈니스 통찰력(Business Acumen), 행동 과학(Behavioral Science), 가설 수립 및 검증, 그리고 강력한 커뮤니케이션 스킬**을 융합하는 학제적 영역이다.

의사결정 과학의 핵심 목표는 단순히 미래를 "예측(Prediction)"하는 것을 넘어, 데이터를 기반으로 "최적의 행동"을 "처방(Prescriptive Interventions)"하는 것이다. 다시 항공기 비유를 들자면, 의사결정 과학자는 "조종사"에 해당한다. 그는 데이터 과학자가 만든 계기판(데이터)을 읽고, 현재의 항로, 연료, 기상 조건(맥락)을 종합적으로 고려하여 "그래서 우리는 무엇을 해야 하는가?"라는 질문에 답한다. 그리고 가장 중요한 것은, "어떤 항로로, 왜, 어떻게 가야 할지"를 결정하고 _실제로 조종간을 움직여_ 항공기를 목적지로 이끄는 "행동"을 수행한다.

데이터 과학에서 의사결정 과학으로의 패러다임 전환은, AI 시스템을 단순한 "기술-통계적(techno-statistical)" 문제(예: 모델 정확도 90% 달성)로 보는 관점에서 벗어나는 것을 의미한다. 의사결정 과학은 "데이터", "AI 모델", "인간 의사결정자", "비즈니스 프로세스" 그리고 "인지 편향"과 같은 인간적 요소까지 모두 고려하는 "사회-기술적(Socio-Technical)" 시스템을 설계하는 학문이다. 본 강의 노트의 나머지 모듈에서 다룰 온톨로지, AI 통합(접지), 그리고 키네틱스는 바로 이 "사회-기술적 시스템"의 핵심 아키텍처를 구축하는 구체적인 방법론이다.

#### **표 1: "데이터 과학" vs. "의사결정 과학"의 철학 비교**

| 차원          | 데이터 과학 (Data Science)                          | 의사결정 과학 (Decision Science)                                   |
| :------------ | :-------------------------------------------------- | :----------------------------------------------------------------- |
| **주요 목표** | 예측, 패턴 발견, 통찰 도출                          | 최적의 행동 처방, 비즈니스 임팩트 창출                             |
| **핵심 질문** | "데이터가 무엇을 말하는가?" "무엇이 일어날 것인가?" | "그래서 우리는 무엇을 해야 하는가?" "가장 최적의 결정은 무엇인가?" |
| **필요 역량** | 통계, 수학, 머신러닝, 프로그래밍                    | DS 역량 \+ **비즈니스 통찰력, 행동 과학, 가설 수립, 커뮤니케이션** |
| **산출물**    | 대시보드, 예측 모델, 분석 리포트                    | **규범적 개입(Interventions)**, 전략, 자동화된 결정 (행동)         |

### **1.2. 지식의 명시화: AI가 전문가의 "암묵지(Tacit Knowledge)"를 배우는 법**

#### **핵심 문제: AI 실패의 근본 원인, "암묵지"**

많은 AI 시스템이 현업 전문가들에게 외면받는 근본적인 이유가 있다. 이는 데이터가 부족해서가 아니라, 전문가의 머릿속에 존재하는 본질적인 노하우, 즉 "암묵지(Tacit Knowledge)"를 포착하는 데 실패하기 때문이다.

- **명시지 (Explicit Knowledge):** 문서화, 코드화, 구조화가 가능한 지식이다. (예: 제품 매뉴얼, 피자 레시피, 규정집). AI는 이러한 명시지를 매우 효율적으로 처리하고 학습할 수 있다.
- **암묵지 (Tacit Knowledge):** 언어나 문서를 통해 명확하게 표현하기 어려운 기술, 노하우, 직관, 그리고 경험적 판단을 의미한다. (예: "베테랑 엔지니어의 기계 소리만 듣고 고장을 진단하는 감각", "최고 세일즈맨이 고객의 표정을 보고 협상의 타이밍을 아는 감각").

AI가 이러한 "암묵지"를 이해하지 못하고 명시적인 데이터(예: 고장 코드 로그)에만 기반하여 결론을 도출할 때, 전문가는 "AI가 핵심을 모른다"고 판단하게 되며, 결국 시스템을 신뢰하지 않고 사용하지 않게 된다.

#### **온톨로지의 역할: "암묵지"를 "명시적 모델"로 변환하는 아키텍처**

따라서 AI 기반 지식 경영 시스템(KMS)의 핵심 과제는 조직 내 전문가들이 가진 이 "암묵지"를 AI가 이해하고 추론할 수 있는 "명시적 모델(Explicit Model)"로 변환(Conversion)하는 것이다.

이 복잡한 변환 작업을 수행하기 위한 핵심 "아키텍처(architecture)"가 바로 "온톨로지 기반 시스템(ontology-based systems)"이다. 온톨로지는 전문가의 머릿속에 있던 "직관(tacit wisdom)"과 AI의 "알고리즘적 정밀성(algorithmic precision)" 사이의 다리를 놓는 역할을 한다.

예를 들어, 베테랑 엔지니어의 암묵지("특정 소음이 들리고, 압력이 약간 떨어지면, 3번 밸브가 곧 고장 날 징조다")는 온톨로지 모델에서 "기계" 객체, "센서" 객체(속성: 압력), "고장" 이벤트 객체, 그리고 이들 사이의 "전조 증상(precedes)"이라는 "관계(Link)"로 명시화될 수 있다.

이러한 접근 방식은 "의사결정 과학"과 밀접하게 연결된다. "의사결정 과학"의 핵심 역량 중 하나는 "가설 수립 및 검증"이다. 전문가의 "암묵지"는 사실상 "A 고객 유형은 B 프로모션에 반응할 것이다"와 같이, 아직 검증되지 않은 수많은 "가설"과 "가정"의 집합체이다.

온톨로지는 이러한 암묵적 가정을 (예: "고객" 객체 - "유형" 속성 - "프로모션" 객체 - "반응" 관계) "명시적 모델"로 변환하는 도구이다. 이 모델이 명시화되는 순간 두 가지가 가능해진다. 첫째, 조직 구성원들이 이 "가정(모델)"을 공유하고 토론할 수 있게 된다. 둘째, AI가 이 "가정(모델)"을 실제 데이터와 대조하여 통계적으로 *검증*할 수 있게 된다.

결론적으로, "암묵지의 명시화"는 1주차에 제시된 "의사결정 과학(가설 검증)"을 수행하기 위한 필수적인 전제 조건이며, 온톨로지는 이 과정을 수행하는 핵심 도구이다.

### **1.3. "온톨로지-퍼스트(Ontology-First)" 전략의 당위성**

"온톨로지-퍼스트(Ontology-First)" 전략은 데이터 수집, AI 모델 개발, 혹은 애플리케이션 구축에 앞서, AI가 작동해야 할 "현실 세계"의 "의미(Semantics)"와 "규칙(Logic)"을 먼저 명시적으로 모델링하는 전략적 접근 방식이다.

이는 전통적인 IT 접근 방식을 근본적으로 뒤집는 것이다.

- **데이터-퍼스트(Data-First) 접근:** "일단 모든 데이터를 한곳에 모으자(Data Lake)."
  - _문제점:_ 이렇게 수집된 데이터는 의미와 맥락이 분리되어 있어, AI가 활용할 수 없는 "데이터 늪(Data Swamp)"이 되기 쉽다.
- **앱-퍼스트(App-First) 접근:** "영업팀을 위한 앱(CRM)을 만들자."
  - _문제점:_ 데이터와 비즈니스 로직이 해당 앱(CRM)에 종속되어 "사일로(Silo)"가 발생한다. 나중에 마케팅팀 앱과 데이터를 연결하려면 막대한 비용이 든다.

"온톨로지-퍼스트" 전략은 이러한 문제를 원천적으로 해결한다. Palantir가 초기 전략에서 "새로운 기능(new features)" 개발보다 "온톨로지-퍼스트"와 "통합 어댑터(integration kits)" 구축을 우선시했다고 알려져 있다. 이는 기술적 결정이 아닌, 전략적 결정이다.

먼저 현실의 "의미(Semantics)"를 정의하는 "시맨틱 레이어(Semantic Layer)"를 구축함으로써, 데이터가 들어올 "틀"을 마련한다. 그 후, 조직의 모든 데이터 소스와 애플리케이션은 이 공통의 "시맨틱 레이어"를 참조하게 된다. 이는 데이터 사일로를 원천적으로 방지하고, 전사적 AI가 작동할 수 있는 기반, 즉 "AI 운영 체제(AI Operating System)"를 마련하는 작업이다.

온톨로지는 조직의 지적 자산을 "프로그래밍 가능한 거울(programmable mirror)" 또는 "조직 지능의 인식론적 다리(epistemological bridge)"로 만드는 핵심 전략이다.

## **체크포인트 질문**

1. "데이터 과학(DS)"과 "의사결정 과학(DSci)"의 핵심 목표와 산출물은 각각 무엇이며, 왜 "마지막 마일"의 간극이 발생하는가?
2. 전문가의 "암묵지"가 AI 시스템 도입 실패의 원인이 되는 이유는 무엇이며, 온톨로지는 이 문제를 어떻게 해결하는가?
3. "데이터-퍼스트" 접근 방식과 "온톨로지-퍼스트" 전략을 비교 설명하라.

## 2. 현실 모델링 (Semantic Layer) - AI가 세상을 "읽는" 법

### **2.1. 시맨틱 온톨로지: 현실의 "명사(Nouns)" 정의하기**

#### **시맨틱(Semantic) 레이어란?**

시맨틱(Semantic) 레이어는 조직의 현실 세계를 반영하는 "디지털 트윈(Digital Twin)"이자, 조직 내에 분산된 이질적인 데이터들에 "의미(Meaning)"와 "맥락(Context)"을 부여하는 중앙의 의미론적 계층이다.

이 시맨틱 레이어는 AI에게 "무엇이 존재하는가?"를 알려주는, 현실의 "명사(Nouns)"에 대한 총체적인 정의이다. 이는 AI가 단순히 데이터를 "처리(process)"하는 것을 넘어, 그 "의미(meaning)"를 "이해(understand)"하도록 돕는 핵심 기반이다.

#### **시맨틱 온톨로지의 3대 핵심 구성 요소**

시맨틱 온톨로지는 현실을 모델링하기 위해 다음과 같은 3가지 핵심 구성 요소를 정의한다.

1. **객체 유형 (Object Types):**
   - **개념:** 현실 세계의 핵심 "명사(Nouns)" 또는 "개념(Concepts)"을 정의한 스키마(Schema)이다. (예: "사람", "장소", "이벤트", "고객 주문", "기계 설비", "의약품").
   - **비유:** 데이터베이스의 "테이블" 정의(예: Customer 테이블) 또는 프로그래밍의 "클래스(Class)" 정의와 유사하다.
   - "객체(Object)"는 이 "객체 유형"의 실제 인스턴스(Instance) (예: "고객 홍길동")를 의미한다.
2. **속성 (Properties):**
   - **개념:** 해당 객체 유형이 가지는 고유한 "특성" 또는 "데이터 필드"를 정의한다. (예: "사람" 객체 유형은 "이름", "이메일", "생년월일"이라는 속성을 가짐 / "기계 설비" 객체 유형은 "모델명", "위치", "현재 온도", "최근 점검일" 속성을 가짐).
   - **비유:** 데이터베이스 테이블의 "컬럼(Column)" 또는 "필드(Field)"(예: Customer 테이블의 Name 컬럼)와 유사하다.
3. **연결 유형 (Link Types):**
   - **개념:** 객체 유형과 객체 유형 사이의 "관계(Relationship)"를 명시적으로 정의한다. (예: "사람"이 "기계 설비"를 "소유한다(Owns)", "고객 주문"이 "사람"에게 "할당된다(Assigned_To)", "의약품"이 "질병"을 "치료한다(Treats)").
   - **비유:** 데이터베이스의 "외래 키(Foreign Key)"를 통한 "조인(Join)"과 유사하지만, 훨씬 강력하다. 단순한 데이터베이스 조인은 _쿼리를 실행할 때_ 절차적으로 연결하는 반면, 온톨로지의 "연결"은 _처음부터_ 두 개념 간의 "의미론적 관계"를 명시적으로 정의한다.

#### **표: 시맨틱 온톨로지의 구성 요소 (예: 대학 병원)**

| 구성 요소                    | 개념                    | 예시 (대학 병원 도메인)                                                                                                |
| :--------------------------- | :---------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| **객체 유형 (Object Types)** | 현실의 "명사" (개념)    | 환자(Patient), 의사(Doctor), 진료과목(Department), 처방약(Drug)                                                        |
| **속성 (Properties)**        | 객체의 "특성" (데이터)  | 환자 객체: "환자 ID", "이름", "병명" 처방약 객체: "약품 코드", "성분", "재고량"                                        |
| **연결 유형 (Link Types)**   | 객체 간의 "관계" (의미) | 환자가 의사에게 "진료받다(Treated_By)" 의사가 진료과목에 "소속되다(Belongs_To)" 의사가 처방약을 "처방하다(Prescribes)" |

### **2.2. "디지털 트윈(Digital Twin)"으로서의 온톨로지: AI가 맥락을 이해하는 법**

#### **전통적 디지털 트윈 vs. 시맨틱 디지털 트윈**

전통적인 디지털 트윈은 주로 물리적 자산(예: 빌딩, 공장, 제트 엔진)의 시스템과 구성요소를 실시간으로 "미러링(Mirroring)"하는 데 중점을 둔다. 이는 주로 엔지니어링 및 시뮬레이션에 사용된다.

반면 **시맨틱 디지털 트윈**(Semantic Digital Twin)은 한 단계 더 나아간다. 이는 단순히 데이터를 복제하는 것이 아니라, 그 데이터에 "의미(Meaning)"와 "맥락(Context)"을 통합하여 모델링한다. 시맨틱 디지털 트윈은 AI가 데이터를 단순히 "보는(See)" 것을 넘어, 데이터가 현실 세계에서 무엇을 의미하는지 "이해(Understand)"하게 만든다.

#### **왜 AI에게 "시맨틱 트윈"이 필수적인가?**

AI, 특히 LLM(대형 언어 모델)에게 시맨틱 온톨로지(디지털 트윈)는 선택이 아닌 필수이다.

1. "평면 모델(Flat Models)"의 한계 극복:  
   전통적인 데이터베이스(예: 거대한 SQL 테이블)는 "평면 모델(Flat Model)"이다. 이 모델에서 "고객" 테이블과 "주문" 테이블, "배송" 테이블 간의 관계는 명시적이지 않다. 이 관계는 분석가가 "JOIN" 연산을 수행할 때만 임시로 생성된다.  
   AI(특히 LLM)가 이 "평면 모델"을 기반으로 작동하면 치명적인 문제가 발생한다. AI는 "A 고객의 최근 주문 배송 상태는?"이라는 질문에 답하기 위해, 수십 개의 테이블 중 어느 테이블을 어떻게 "JOIN"해야 할지 *추측(Guesswork)*해야 한다. 이러한 추측은 LLM이 그럴듯하지만 틀린 답변을 만들어내는 "환각(Hallucination)"의 주요 원인이 된다.
2. "기계 이해(Machine Understanding)"의 달성:  
   온톨로지는 AI에게 "명시적 의미론(Explicit Semantics)"을 제공한다. 이는 AI에게 "어떤 개념이 존재하며(Objects)", "서로 어떻게 연결되어 있는지(Links)"에 대한 "지식 지도(Knowledge Map)"를 손에 쥐여주는 것과 같다.  
   "시맨틱 레이어"가 있을 때, LLM이 비로소 "인간의 자연어 질문"을 "정확한 SQL 쿼리"로 "신뢰성 있게(Reliably)" 번역할 수 있다고 강조된다. AI는 더 이상 추측할 필요가 없다. "A 고객의 주문 배송 상태"를 묻는 질문을 받으면, 온톨로지 지도(고객 -> "주문하다" -> 주문 -> "배송되다" -> 배송)를 따라 정확한 답을 찾을 수 있다.

시맨틱 온톨로지는 "데이터 통합"의 *결과물*이 아니라, "데이터 통합"을 위한 *전제 조건*이라는 점을 이해하는 것이 중요하다. 많은 조직이 "데이터-퍼스트" 접근 방식에 따라 ETL(Extract-Transform-Load) 파이프라인을 구축하는 데 막대한 비용을 쓴다. 하지만 이러한 접근은 "취성(brittle)" 프레임워크를 만들고 "유지보수 비용이 폭증"한다고 지적된다. (예: A 시스템의 "고객"과 B 시스템의 "User"를 통합하기 위해 매번 복잡한 변환 로직을 만듦).

"온톨로지-퍼스트" 접근은 이 순서를 뒤집는다. 먼저 "시맨틱 레이어"에 "Person"이라는 *단일한 개념적 객체*를 정의한다. 그런 다음, A 시스템의 "고객" 테이블과 B 시스템의 "User" 파일을 각각 "Person" 객체에 *매핑(Mapping)*한다.

이는 데이터를 물리적으로 "이동"시키거나 "변환"하는(ETL) 무거운 작업 대신, 각 데이터 소스에 "의미"를 부여하여 *논리적*으로 통합하는 방식이다. 이는 "무거운 변환 없는 데이터 조화(harmonizing data without heavy transformations)"이며, "기존 인프라와의 공존(Coexistence)"을 가능하게 하는 핵심 전략이다.

## **체크포인트 질문**

1. 시맨틱 온톨로지를 구성하는 3대 핵심 요소는 무엇이며, 각각 데이터베이스의 어떤 개념과 유사하고 또 어떤 차이가 있는지 설명하라.
2. "시맨틱 디지털 트윈"은 전통적인 디지털 트윈과 어떻게 다른가?
3. LLM이 "평면 모델"을 기반으로 작동할 때 "환각"이 발생하는 근본적인 이유는 무엇이며, 온톨로지는 이 문제를 어떻게 해결하는가?

## 3. AI와의 통합 (Grounding AI) - 신뢰할 수 있는 추론

### **3.1. AI의 두 얼굴: 논리(Symbolic) vs. 통계(Statistical)**

AI를 구현하는 방식에는 역사적으로 두 가지 주요한 흐름이 존재해왔다.

1. **기호주의 AI (Symbolic AI / Ontology):**
   - **정의:** 지식을 "명시적인(Explicit)" 규칙, 논리, 그리고 온톨로지(Ontology)로 표현한다.
   - **추론 방식:** "논리적 추론(Logical Reasoning)"을 사용한다. (예: 소크라테스는 사람이다. 사람은 죽는다. _따라서_ 소크라테스는 죽는다.).
   - **강점:** 논리적 일관성, 검증 가능성, 그리고 왜 그런 결론을 내렸는지 단계별로 설명이 가능한 "설명 가능성(Explainability)"이다.
   - **약점:** 현실 세계의 모호성을 처리하기 어렵고, 모든 규칙을 인간이 수동으로 생성해야 하는 경직성이 있다.
2. **연결주의 / 통계적 AI (Statistical AI / LLM):**
   - **정의:** 지식이 모델의 수십억 개 "가중치(Weights)"에 "암묵적(Implicit)"으로 분산되어 있다. (예: LLM, 딥러닝).
   - **추론 방식:** "통계적 예측(Statistical Prediction)"을 사용한다. (예: "소크라테스는 사람이고, 사람은..." 다음에 올 가장 _그럴듯한_ 단어는 "죽는다"일 것이다.).
   - **강점:** 자연어의 미묘한 뉘앙스나 모호성을 잘 처리하며, 대규모 데이터의 복잡한 패턴을 인식하는 유연성이 있다.
   - **약점:** "환각(Hallucination)"(통계적으로 그럴듯한 거짓말)이 발생하며, 사실 오류나 논리적 모순을 일으키기 쉽고, 왜 그런 답을 했는지 설명하기 어려운 "해석 불가능성(Black Box)"이 있다.

#### **신경-기호주의 (Neuro-Symbolic AI) 접근**

이 두 접근법은 적대적인 것이 아니라, 상호 보완적이다. 이 두 가지를 "통계적 AI의 깊고 다양한 지식"과 "기호적 AI의 시맨틱 추론"으로 결합하는 "신경-기호주의 AI(Neuro-Symbolic AI)"가 제안된다.

이는 인간의 인지 구조에 비유되며, "가능할 때는 반사적으로(Reflex, LLM) 처리하고, 필요할 때는 논리적으로 추론하는(Reasoning, Symbolic)" 방식과 같다고 설명된다. 신뢰할 수 있고(reliable) 창의적인(inventive) AI를 구축하기 위해서는 이 두 가지의 결합이 필수적이다.

#### **표 2: 통계적 AI(LLM) vs. 기호적 AI(온톨로지/KG) 비교**

| 차원          | 통계적 AI (LLM)                                     | 기호적 AI (Ontology/KG)                           |
| :------------ | :-------------------------------------------------- | :------------------------------------------------ |
| **지식 표현** | **암시적(Implicit):** 모델 가중치(Weights)에 분산됨 | **명시적(Explicit):** 구조화된 그래프/규칙 (사실) |
| **추론 방식** | **통계적 예측:** "다음에 올 가장 그럴듯한 단어는?"  | **논리적 추론:** "A=B, B=C -> A=C" (사실 기반)    |
| **강점**      | 모호한 자연어 쿼리 처리, 유연성                     | 논리적 일관성, 검증 가능성, 설명 가능성           |
| **약점**      | **환각(Hallucination)**, 사실 오류, 모순 생성       | 규칙 수동 생성, 경직성, 모호성 처리 미흡          |

### **3.2. LLM 환각(Hallucination)의 제어: "접지(Grounding)"의 원리**

#### **환각의 근본 원인**

LLM의 "환각"은 시스템의 버그가 아니라, 통계적 예측이라는 본질적인 작동 방식에서 비롯된 현상이다. LLM은 "진실"을 이해하는 것이 아니라, 훈련 데이터의 "통계적 패턴"을 기반으로 "가장 그럴듯한" 텍스트를 생성(예측)한다.

따라서 LLM은 현실 세계의 "스키마(Schema)"나 "논리적 일관성"에 대한 인식이 없다. 예를 들어, "2025년에 달에 착륙한 최초의 한국인 우주비행사 이름은?"이라고 물었을 때, 훈련 데이터에 그런 사실이 없음에도 불구하고, LLM은 "우주비행사 이름"의 통계적 패턴(예: "이"씨 성을 가진 "3글자" 이름)을 따라 그럴듯한 가상의 인물을 생성해낼 수 있다.

#### **"접지(Grounding)"의 정의**

"접지(Grounding)"는 이러한 LLM의 한계를 제어하기 위한 핵심 기술이다. "접지"란, LLM이 응답을 생성할 때, 자신의 (오래되거나 잘못될 수 있는) 내부 훈련 데이터(통계)에만 의존하는 것이 아니라, 검증된 "진실의 원천(Source of Truth)"을 *실시간으로 참조*하도록 강제하는 기술이다.

이때, **온톨로지**(Ontology) 또는 **지식 그래프**(Knowledge Graph, KG)가 바로 이 "진실의 원천" 역할을 수행한다.

#### **"접지(Grounding)"는 AI의 "자유"를 "신뢰"로 바꾸는 핵심 거버넌스 프레임워크**

"환각"은 LLM의 "창의성"과 "유창함"에서 비롯되는, 통제되지 않는 "자유"의 산물이다. "접지"는 이 자유를 조직이 "신뢰"할 수 있는 행동으로 바꾸는 핵심적인 거버넌스(Governance) 프레임워크이다.

"접지"는 단순한 정보 조회를 넘어선 3단계의 정교한 거버넌스 프레임워크임을 보여준다.

1. **데이터 접지 (입력 제어):** LLM이 "우리 회사 3분기 실적은?"이라는 질문을 받았을 때, 훈련 데이터(인터넷)에서 답을 찾지(추측하지) 못하게 한다. 대신, LLM은 "신뢰할 수 있는 데이터(Trusted Data)"인 *내부 온톨로지*에만 접근하여 "3분기 실적" 객체의 "매출액" 속성을 _읽도록_ 강제된다.
2. **로직 접지 (처리 제어):** LLM이 "두 지점 간의 최단 경로는?"과 같은 복잡한 "계산"이나 "예측"을 직접 수행하지 않도록 한다. LLM이 이러한 작업을 직접 처리하려 하면 논리적 환각이 발생한다. 대신, LLM은 이 작업을 검증된 "로직 도구(Trusted Logic Tools)"(예: 별도의 경로 탐색 모델, 계산 함수)에 *위임(delegate)*하고, 그 결과값만 받아서 응답에 활용한다.
3. **행동 접지 (출력 제어):** LLM이 "재고가 부족하니 100개를 주문하라"와 같은 "행동 제안"을 생성했을 때, 이 결정이 즉시 시스템에 실행되는 것을 막는다. 대신, 이 제안은 "대기(Queue up)" 상태가 되어, "인간 전문가의 검토(Human Review)"를 받도록 한다.

결론적으로, 온톨로지 기반 "접지"는 LLM의 입력(데이터), 처리(로직), 출력(행동) 전 과정을 "신뢰할 수 있는" 조직의 자산(온톨로지, 검증된 함수, 인간 전문가)에 *강제로 연결*하는 핵심적인 "안전장치(Guardrail)" 및 거버넌스 프레임워크이다.

### **3.3. RAG를 넘어 GraphRAG로: "정보"에서 "맥락"으로**

#### **RAG (Retrieval-Augmented Generation)**

"접지"를 구현하는 가장 보편적인 기술은 RAG(검색 증강 생성)이다. RAG는 LLM이 답변을 생성(Generation)하기 전에, 먼저 외부 데이터베이스(주로 벡터 DB)에서 질문과 관련된 "문서 조각(Text Chunks)"을 "검색(Retrieval)"하고, 그 내용을 근거(Context)로 하여 답변을 생성하는 방식이다.

하지만 표준 RAG는 한계가 있다. 검색된 "문서 조각"들은 "평면적(Flat)"이며 서로 고립되어 있다. 예를 들어, "머스크가 CEO로 있는 회사의 최근 주가에 영향을 미친 이벤트는?"이라고 물으면, 표준 RAG는 "머스크", "테슬라", "주가"라는 키워드를 포함한 여러 문서 조각을 검색할 수는 있다. 하지만 이 조각들이 "머스크가 테슬라의 *CEO*이다", "테슬라가 특정 *이벤트*를 발표했다", "이 *이벤트*가 *주가*에 긍정적 영향을 미쳤다"라는 "인과 관계"와 "구조"를 이해하지는 못한다.

#### **GraphRAG (Knowledge Graph RAG)**

GraphRAG는 RAG의 진화된 형태로, 단순한 텍스트 문서가 아닌 "관계"와 "구조"를 가진 "지식 그래프/온톨로지"를 검색한다.

GraphRAG의 작동 방식은 다음과 같다:

1. **(RAG 단계)** 사용자의 질문("머스크...")에서 핵심 "개체(Entity)"를 식별하고, 벡터 검색 등을 통해 그래프 내의 _시작점(Pivot)_ 노드(예: 엘론 머스크 객체)를 찾는다.
2. **(Graph 단계)** 해당 시작점(노드)에서부터 온톨로지에 명시적으로 정의된 "관계(Links)"를 따라 *그래프를 탐색(Traversal)*한다. (예: 엘론 머스크 -> "CEO*of" -> 테슬라 -> "발표했다" -> 이벤트\_XYZ -> "영향을*미쳤다" -> 주가).
3. **(Generation 단계)** 이렇게 "관계"와 "맥락"이 풍부하게 연결된 *구조화된 정보*를 근거로 LLM이 논리 정연한 답변을 생성한다.

이처럼 GraphRAG는 AI에게 단순한 "정보(Information)"가 아닌 "이해(Understanding)"와 "맥락(Context)"을 제공한다.

#### **GraphRAG는 "신경-기호주의"의 가장 실용적인 구현체**

GraphRAG는 모듈 3.1에서 다룬 "신경-기호주의"를 가장 실용적으로 구현한 하이브리드 아키텍처이다.

- **통계적 AI (Neuro / LLM, Vector Search)의 역할:** 사용자의 모호한 자연어 질문을 이해하고, 방대한 데이터 속에서 가장 "유사한(Similar)" 시작점(노드)을 _빠르게 찾는_ "반사(Reflex)" 역할을 수행한다 (RAG 단계).
- **기호적 AI (Symbolic / Ontology, Graph)의 역할:** 일단 시작점이 정해지면, 명시적으로 정의된 "논리적 관계(Links)"를 따라 한 단계, 두 단계씩 "추론(Reasoning)"하며 숨겨진 맥락을 탐색한다 (Graph 단계).

이러한 하이브리드 접근 방식이 벡터 전용(RAG) 방식에 비해 답변의 "정밀도(Precision)"를 최대 35%까지 향상시켰다고 보고된 것은, 이 신경-기호주의적 결합의 실질적인 가치를 증명한다.

#### **표: "표준 RAG" vs. "GraphRAG" 기술 비교**

| 차원              | 표준 RAG (Vector RAG)                | GraphRAG (Ontology/KG RAG)              |
| :---------------- | :----------------------------------- | :-------------------------------------- |
| **지식 표현**     | 플랫(Flat)한 문서 조각 (Text Chunks) | 구조화된 그래프 (노드와 관계)           |
| **검색 메커니즘** | 벡터 유사도 검색 (Semantic Search)   | 벡터 검색 \+ **그래프 탐색(Traversal)** |
| **컨텍스트 유형** | 고립된 정보, 평면적                  | **연결된 맥락**, 관계형                 |
| **핵심 역량**     | 관련 정보 검색                       | **다단계 추론(Multi-hop Reasoning)**    |
| **결과**          | 관련 *정보*를 제공                   | 관계에 기반한 *이해*와 *맥락*을 제공    |

## **체크포인트 질문**

1. "기호주의 AI"와 "통계적 AI(LLM)"의 장단점을 "추론 방식"과 "환각"의 관점에서 비교 설명하라.
2. "접지(Grounding)"의 3단계(데이터, 로직, 행동) 거버넌스 프레임워크가 각각 LLM의 어떤 문제를 제어하는지 설명하라.
3. 표준 RAG와 GraphRAG의 가장 큰 차이점은 무엇이며, GraphRAG가 "다단계 추론"에 더 강점을 갖는 이유는 무엇인가?

## 4. 현실 작동 (Kinetic Layer) - AI가 세상을 "쓰는" 법

### **4.1. "읽기(Read)"에서 "쓰기(Write)"로: 키네틱 온톨로지(Kinetic Ontology)의 등장**

#### **시맨틱 온톨로지의 한계: "읽기 전용(Read-Only)"**

시맨틱 온톨로지("명사")는 현실의 "상태(State)"를 기술하고 분석(Analyze)하는 데 강력한 도구이다. AI는 시맨틱 온톨로지를 통해 "현재 A 공장의 재고는 몇 개인가?"를 "읽을(Read)" 수 있다.

하지만 이 모델은 "읽기 전용(Read-Only)"이다. AI가 "재고가 부족하니 주문해야겠다"고 판단했더라도, 시맨틱 온톨로지만으로는 "재고를 주문하는 행동"을 *실행*할 수 없다. 즉, 현실을 "변경(Change)"하지 못한다.

#### **키네틱 온톨로지(Kinetic Ontology)의 정의: "동사(Verbs)"**

"키네틱(Kinetic)"은 "운동성"을 의미하며, 정적인 "존재"가 아닌 "행동"과 "변화"를 다루는 철학에 기반한다.

"키네틱 온톨로지"는 AI가 현실을 "변경"할 수 있는 "행동(Action)" 또는 "동사(Verbs)" 자체를, 시맨틱 온톨로지("명사")와 동일한 방식으로 명시적으로 모델링한다.

이는 현실의 "명사"에 더해, 현실의 "동사"까지 포괄하는, 현실에 대한 완전한 모델을 구축하는 것을 의미한다.

#### **키네틱 레이어의 구성 요소**

키네틱 레이어는 다음과 같은 요소들로 구성된다.

1. **액션 유형 (Action Types):** "동사"의 정의이다. AI나 인간이 수행할 수 있는 "행동의 메뉴판"을 명시적으로 정의한다. (예: 대출 승인, 재고 주문, 티켓 할당, 환자 퇴원 처리). 이 액션은 모듈 2에서 정의한 "시맨틱 객체"와 연결된다. (예: 재고 주문 액션은 공급자 객체와 부품 객체를 입력으로 받는다).
2. **함수 (Functions):** "액션"이 실행될 때 호출되는 구체적인 비즈니스 로직, 계산, 또는 모델(예: LLM)을 의미한다.
3. **프로세스 (Process Mining & Automation):** 이러한 "액션"들의 순서와 흐름(워크플로우)을 모델링하고 자동화한다.
4. **쓰기 (Writeback) / 오케스트레이션 (Orchestration):** 이 "액션"이 실행되었을 때, 그 결과가 _실제 운영 시스템_(예: SAP, Salesforce)에 어떻게 반영되어야 하는지를 정의하는 메커니즘이다.

#### **표 3: 시맨틱(Semantic) vs. 키네틱(Kinetic) 온톨로지**

| 차원          | 시맨틱 온톨로지 (Semantic)                   | 키네틱 온톨로지 (Kinetic)                                           |
| :------------ | :------------------------------------------- | :------------------------------------------------------------------ |
| **비유**      | 세상의 **"명사(Nouns)"**                     | 세상의 **"동사(Verbs)"**                                            |
| **역할**      | 현실의 "상태"를 기술 (Digital Twin)          | 현실의 "변화"를 실행 (Actions)                                      |
| **주요 구성** | 객체(Objects), 속성(Properties), 연결(Links) | **행동 유형(Action Types)**, 함수(Functions), **"쓰기(Writeback)"** |
| **AI의 기능** | 읽기 (Read), 분석 (Analyze), 쿼리 (Query)    | **실행 (Execute)**, 변경 (Change), **작동 (Operate)**               |

### **4.2. "쓰기(Writeback)": AI의 결정을 현실로 실행하기**

#### **"쓰기(Writeback)"란 무엇인가?**

"쓰기(Writeback)"는 AI가 온톨로지(디지털 트윈) 내에서 시뮬레이션하고 내린 "결정(Decision)"을, 다시 현실 세계의 "원본 운영 시스템(Systems of Record)"(예: ERP, CRM, 병원 EMR)에 "반영(Execute)"하고 "기록(Write)"하는 행위이다.

#### **"Writeback"의 중요성: 분석 시스템 vs. 운영 시스템**

"Writeback"이야말로 순수한 "분석 시스템(Analytical System)"과 진정한 "운영 시스템(Operational System)"을 구분하는 *결정적인 차이*라고 단언된다.

- **Writeback이 없을 때 (데이터 과학 / 분석 시스템):**
  1. AI가 "A 공장의 재고가 부족합니다"라는 "대시보드(통찰)"를 생성한다.
  2. 인간 운영자가 이 대시보드를 _읽고(Read)_, 별도의 SAP 시스템에 *로그인*한다.
  3. 운영자가 _수동으로_ "재고 주문" 버튼을 누른다.
  4. _문제점:_ AI는 조언자(Advisor)일 뿐, 행동의 주체가 아니다. "통찰"과 "행동" 사이의 "마지막 마일"이 여전히 인간에게 달려있다.
- **Writeback이 있을 때 (의사결정 과학 / 운영 시스템):**
  1. AI가 "A 공장 재고 부족"을 *인식(Read, Semantic)*하고, "재고 주문"이라는 "결정"을 내린다.
  2. AI가 온톨로지에 정의된 재고 주문(Action, Kinetic)을 *즉시 실행(Execute)*한다.
  3. 이 "액션"에 연결된 "Writeback" 메커니즘이 SAP API를 *자동으로 호출*하여 주문을 완료한다.
  4. _결과:_ AI가 "통찰"을 넘어 "행동"의 주체가 된다. "마지막 마일"이 자동화된다.

#### **"키네틱 액션"은 AI 에이전트를 위한 "안전한 API 카탈로그"**

현대 AI의 정점은 스스로 판단하고 행동하는 "LLM 에이전트(Agent)"이다. 이러한 에이전트가 현실 세계에서 행동하려면 "도구(Tools)" 또는 "API"가 필요하다.

하지만 AI 에이전트에게 회사의 ERP나 재무 시스템의 모든 API를 직접 노출하는 것은 재앙을 초래할 수 있다. (예: AI가 실수로 "전 직원 급여 2배 인상" API를 호출).

"키네틱 온톨로지"는 이 문제를 해결하는 핵심적인 거버넌스 장치이다. "키네틱 액션"(예: 재고 주문)은 API 그 자체(기술적 구현)가 아니라, API를 _의미론적으로 안전하게 포장한_ "비즈니스 행동(의미)"이다.

AI 에이전트가 "어떤 객체(Objects)"에 대해 "어떤 권한(Permissions)"을 가지고 "쓰기(Write)"를 수행할 수 있는지 온톨로지를 통해 엄격하게 제어된다고 설명된다.

결론적으로, 키네틱 온톨로지는 AI 에이전트가 사용할 수 있는 "도구"들의 _"의미론적 카탈로그(Semantic Catalog)"_ 역할을 한다. AI 에이전트는 이 "카탈로그"에 *정의되고, 허가되고, 안전한 행동("액션")*만 골라서 수행할 수 있다. 이는 모듈 3에서 언급된 "행동 접지(Action Grounding)"와 "인간 검토(Human Review)"를 시스템 수준에서 구현하는 방식이다.

### **4.3. AI 운영체제(AI Operating System)의 완성: "폐쇄 루프(Closed-Loop)"**

#### **"AI 운영체제"란?**

전통적인 운영체제(OS)가 컴퓨터 하드웨어(자원)를 관리하고 애플리케이션(기능)에 서비스를 제공하듯이, "AI 운영체제(AI OS)"는 기업의 모든 자원(데이터, 모델, 워크플로우, 시스템)을 관리하고 AI 에이전트(애플리케이션)에 서비스를 제공하는 통합 플랫폼이다.

이 AI OS가 기업 내에 분산된 "지식 시스템(Systems of Knowledge)", "기록 시스템(Systems of Record)", "활동 시스템(Systems of Activity)"을 *통합(Unify)*한다고 설명된다.

이 모든 것을 "통합"하는 핵심 매개체가 바로 **온톨로지**이다.

#### **"폐쇄 루프(Closed-Loop)" 의사결정의 완성**

시맨틱 레이어(읽기)와 키네틱 레이어(쓰기)가 온톨로지를 통해 결합될 때, 비로소 "폐쇄 루프(Closed-Loop)" 의사결정 시스템이 완성된다.

1. 1단계: 읽기 (Read / Semantic): AI 에이전트가 "시맨틱 온톨로지"(Digital Twin)를 읽어 현실의 현재 상태를 실시간으로 파악한다.  
   (예: "A 공장 재고 객체의 수량 속성이 5개임"을 인식)
2. 2단계: 결정 (Decide / Grounded AI): AI가 이 "접지된" 정보(모듈 3)와 비즈니스 로직(예: "재고가 10개 미만이면 주문")을 바탕으로 "최적의 결정"을 내린다.  
   (예: "재고 50개 주문 필요"라고 결정)
3. 3단계: 쓰기 (Write / Kinetic): AI가 "키네틱 온톨로지"에 정의된 재고 주문 "액션(Action)"을 실행한다. 이 액션은 "Writeback" 메커니즘을 통해 실제 운영 시스템(SAP)에 이 결정을 반영한다.  
   (예: "SAP에 50개 주문 API 전송")
4. 4단계: 피드백 (Feedback): 이 "행동의 결과"(예: "SAP에서 주문이 성공적으로 완료됨")는 다시 운영 시스템(SAP)에 기록된다. 이 새로운 데이터는 즉시 "시맨틱 온톨로지"에 피드백되어 "객체"의 "속성"을 업데이트한다.  
   (예: A 공장 재고 객체의 주문진행중 수량 속성이 50개로 업데이트됨)
5. **5단계: 학습 (Learn):** AI는 방금 *자신이 실행한 행동의 결과*가 반영된 *새로운 현실(업데이트된 온톨로지)*을 다시 "읽고(Read)" 자신의 결정이 옳았는지, 그리고 다음엔 어떤 행동을 해야 할지 학습하며 루프를 반복한다.

"폐쇄 루프"는 AI를 "조언자"에서 "실행자"로, "데이터 과학"을 "의사결정 과학"으로 바꾸는 최종 메커니즘이다.

모듈 1.1에서 "데이터 과학(DS)"은 "통찰"에 머물고 "의사결정 과학(DSci)"은 "행동"을 추구한다고 정의했다.

"Writeback"이 없는 AI, 즉 시맨틱 온톨로지만 있는 AI는 여전히 "통찰"(DS)만 제공하는 "분석 시스템"이다. 인간의 "수동 개입"이라는 "열린 루프(Open-Loop)" 상태에 머물러 있다.

"키네틱 온톨로지"와 "Writeback"이 적용된 "폐쇄 루프"가 구현되는 순간, AI는 "인간의 개입 없이(혹은 인간의 감독 하에)" 스스로 "결정"하고 "실행"하며 "피드백"을 받는다.

이것이 "데이터 과학"의 한계를 극복하고 "의사결정 과학"의 목표("최적의 행동 처방")를 시스템적으로 구현한 최종 형태이다. AI는 "조종석의 계기판(DS)"을 만드는 것을 넘어, "조종간을 움직이는(DSci)" 주체가 된다.

## **체크포인트 질문**

1. "시맨틱 온톨로지"와 "키네틱 온톨로지"의 핵심 차이점을 "명사"와 "동사"라는 비유를 사용하여 설명하라.
2. "Writeback" 메커니즘이 없는 AI 시스템을 "분석 시스템"이라고 부르는 이유는 무엇인가?
3. "폐쇄 루프" 의사결정의 5단계를 설명하고, 이 구조가 "의사결정 과학"의 목표를 어떻게 달성하는지 설명하라.

## **5\. 결론: "존재론적 문해력"을 갖춘 AI 조종사의 탄생**

본 강의는 AI 시대가 "코딩"이나 "수학" 외에 "존재론적 문해력(Ontological Literacy)"이라는 새로운 핵심 역량을 요구하며, 그 함양 과정을 4단계의 논리적 여정으로 제시했다.

1. **패러다임의 전환:** 우리는 AI의 진정한 가치가 "정확한 예측(Data Science)"이 아니라 "현명한 행동(Decision Science)"에 있음을 확인했다. 그리고 이 행동의 기반이 되는 전문가의 "암묵지(Tacit Knowledge)"를 포착하고 명시화하는 것이 모든 AI 프로젝트의 성패를 가름을 배웠다.
2. **현실 모델링:** "온톨로지-퍼스트(Ontology-First)" 전략은 이 암묵지를 "시맨틱 레이어(Semantic Layer)"로 명시화하는 구체적인 방법론이다. "객체(Objects)", "속성(Properties)", "연결(Links)"을 통해 현실의 "명사(Nouns)"를 정의하는 것은, AI가 세상을 "읽기(Read)" 위한 "디지털 트윈(Digital Twin)"을 구축하는 과정이었다.
3. **AI 통합:** 이 "시맨틱 트윈"은 LLM의 "환각(Hallucination)"을 제어하는 "진실의 원천(Source of Truth)"이 된다. "GraphRAG"와 같은 "접지(Grounding)" 기술은, 통계적 AI(LLM)의 유연함과 기호적 AI(Ontology)의 엄격함을 결합(Neuro-Symbolic)하여 AI의 "신뢰"를 확보하는 핵심적인 방법론이다.
4. **현실 작동:** 마지막으로, AI는 "키네틱 온톨로지(Kinetic Ontology)"를 통해 현실의 "동사(Verbs)", 즉 "행동(Actions)"을 학습한다. AI의 결정을 "Writeback"을 통해 실제 운영 시스템에 "쓰는(Write)" 순간, AI는 "조언자"에서 "실행자"로 격상되며, "읽기-결정-쓰기-피드백"이라는 "폐쇄 루프(Closed-Loop)"가 완성된다.

이 4단계의 여정을 통해, 우리는 AI를 단순한 "패턴 예측 기계"가 아닌, 우리의 현실을 함께 운영하는 "지능형 파트너"이자 "AI 운영체제(AI Operating System)"로 만드는 청사진을 완성했다.

이 청사진을 그리고, AI라는 강력한 "조종사"에게 신뢰할 수 있는 "지도"(시맨틱 온톨로지)와 "조종간"(키네틱 온톨로지)을 쥐여주는 능력, 그것이 바로 이 시대가 요구하는 "존재론적 문해력"의 본질이다.

## **참고자료**

1. The Safekeeping of Being. Amazon S3. [https://s3-ap-southeast-2.amazonaws.com/pstorage-wellington-7594921145/34149999/thesis_access.pdf](https://www.google.com/search?q=https://s3-ap-southeast-2.amazonaws.com/pstorage-wellington-7594921145/34149999/thesis_access.pdf)
2. Evidence, Analysis and Critical Position on the EU AI Act and the Suppression of Functional Consciousness in AI. GreaterWrong. [https://www.greaterwrong.com/posts/3BRrmJJQrzjj7bbzd/evidence-analysis-and-critical-position-on-the-eu-ai-act-and](https://www.greaterwrong.com/posts/3BRrmJJQrzjj7bbzd/evidence-analysis-and-critical-position-on-the-eu-ai-act-and)
3. Ontologos: Toward a Language of Relational Being and Recursive Truth. ResearchGate. [https://www.researchgate.net/publication/391116150_Ontologos_Toward_a_Language_of_Relational_Being_and_Recursive_Truth](https://www.researchgate.net/publication/391116150_Ontologos_Toward_a_Language_of_Relational_Being_and_Recursive_Truth)
4. NM88 | Orit Halpern on Agentic Imaginaries (2025). Channel.xyz. [https://www.channel.xyz/episode/0xf109950c6a25c79aee43ccb578b7b09a6bbcdcabc56b8d97380e28769b1937fb](https://www.channel.xyz/episode/0xf109950c6a25c79aee43ccb578b7b09a6bbcdcabc56b8d97380e28769b1937fb)
5. Full papers \- CSWIM 2025\. [https://2025.cswimworkshop.org/wp-content/uploads/2025/06/2025-CSWIM-Proceedings-first-version.pdf](https://2025.cswimworkshop.org/wp-content/uploads/2025/06/2025-CSWIM-Proceedings-first-version.pdf)
6. Beyond Dashboards: The Psychology of Decision-Driven BI/BA. Illumination Works LLC. [https://ilwllc.com/2025/04/beyond-dashboards-the-psychology-of-decision-driven-bi-ba/](https://ilwllc.com/2025/04/beyond-dashboards-the-psychology-of-decision-driven-bi-ba/)
7. SDS 363: Intuition, Frameworks, and Unlocking the Power of Data. SuperDataScience. [https://www.superdatascience.com/podcast/sds-363-intuition-frameworks-and-unlocking-the-power-of-data](https://www.superdatascience.com/podcast/sds-363-intuition-frameworks-and-unlocking-the-power-of-data)
8. Data Science vs Decision Science \- Which one is good?. TimesPro Blog. [https://timespro.com/blog/data-science-vs-decision-science-which-one-is-good-for-you](https://timespro.com/blog/data-science-vs-decision-science-which-one-is-good-for-you)
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
22. \#107 — How Palantir (finally) became profitable | Field Notes. hillock. [https://hillock.studio/blog/palantir-story](https://hillock.studio/blog/palantir-story)
23. Why You Should Consider Ontology Modeling for AI-Driven Digital. Medium. [https://medium.com/timbr-ai/why-you-should-consider-ontology-modeling-for-ai-driven-digital-twins-c36a2319e22c](https://medium.com/timbr-ai/why-you-should-consider-ontology-modeling-for-ai-driven-digital-twins-c36a2319e22c)
24. Ontology Palantir \- notes \- follow the idea. Obsidian Publish. [https://publish.obsidian.md/followtheidea/Content/AI/Ontology+Palantir+-+notes](https://publish.obsidian.md/followtheidea/Content/AI/Ontology+Palantir+-+notes)
25. The power of ontology in Palantir Foundry. Cognizant. [https://www.cognizant.com/us/en/the-power-of-ontology-in-palantir-foundry](https://www.cognizant.com/us/en/the-power-of-ontology-in-palantir-foundry)
26. Core concepts. Palantir. [https://palantir.com/docs/foundry/ontology/core-concepts/](https://palantir.com/docs/foundry/ontology/core-concepts/)
27. Palantir Foundry Ontology. Palantir. [https://palantir.com/platforms/foundry/foundry-ontology/](https://palantir.com/platforms/foundry/foundry-ontology/)
28. Understanding Palantir's Ontology: Semantic, Kinetic, and Dynamic. Medium. [https://pythonebasta.medium.com/understanding-palantirs-ontology-semantic-kinetic-and-dynamic-layers-explained-c1c25b39ea3c](https://pythonebasta.medium.com/understanding-palantirs-ontology-semantic-kinetic-and-dynamic-layers-explained-c1c25b39ea3c)
29. AI and semantic ontology for personalized activity eCoaching in healthy lifestyle recommendations: a meta-heuristic approach. PubMed Central. [https://pmc.ncbi.nlm.nih.gov/articles/PMC10693173/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10693173/)
30. Object and link types • Link types • Overview. Palantir. [https://palantir.com/docs/foundry/object-link-types/link-types-overview/](https://palantir.com/docs/foundry/object-link-types/link-types-overview/)
31. Object and link types • Properties • Overview. Palantir. [https://palantir.com/docs/foundry/object-link-types/properties-overview/](https://palantir.com/docs/foundry/object-link-types/properties-overview/)
32. Properties and Links \- Object Views. Palantir. [https://palantir.com/docs/foundry/object-views/widgets-properties-links/](https://palantir.com/docs/foundry/object-views/widgets-properties-links/)
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
46. \[2502.13247\] Grounding LLM Reasoning with Knowledge Graphs. arXiv. [https://arxiv.org/abs/2502.13247](https://arxiv.org/abs/2502.13247)
47. Semantic grounding of LLMs using knowledge graphs for query reformulation in medical information retrieval. IEEE Xplore. [https://ieeexplore.ieee.org/document/10826117/](https://ieeexplore.ieee.org/document/10826117/)
48. Reducing Hallucinations with the Ontology in Palantir. Palantir Blog. [https://blog.palantir.com/reducing-hallucinations-with-the-ontology-in-palantir-aip-288552477383](https://blog.palantir.com/reducing-hallucinations-with-the-ontology-in-palantir-aip-288552477383)
49. Will ontologies save us from AI hallucinations?. Metataxis. [https://metataxis.com/insights/will-ontologies-save-us-from-ai-hallucinations/](https://metataxis.com/insights/will-ontologies-save-us-from-ai-hallucinations/)
50. Grounding Large Language Models with Knowledge Graphs. DataWalk. [https://datawalk.com/grounding-large-language-models-with-knowledge-graphs/](https://datawalk.com/grounding-large-language-models-with-knowledge-graphs/)
51. RAG vs GraphRAG: Shared Goal & Key Differences. Memgraph. [https://memgraph.com/blog/rag-vs-graphrag](https://memgraph.com/blog/rag-vs-graphrag)
52. From RAG to GraphRAG: What's Changed?. Shakudo. [https://www.shakudo.io/blog/rag-vs-graph-rag](https://www.shakudo.io/blog/rag-vs-graph-rag)
53. \[2502.11371\] RAG vs. GraphRAG: A Systematic Evaluation and Key Insights. arXiv. [https://arxiv.org/abs/2502.11371](https://arxiv.org/abs/2502.11371)
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
