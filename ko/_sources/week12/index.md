# Week 12: AI 규제와 책임 있는 AI

## 강의 개요

자연어 처리를 위한 딥러닝 12주차 강의에 오신 것을 환영한다. 오늘은 우리가 배운 NLP 기술이 현실 세계와 만나는 가장 중요하고 복잡한 교차점인 규제와 책임성 문제를 다룬다. 2025년 11월 현재, 우리는 AI 기술의 "황야 시대"가 끝나고 "법의 시대"가 시작되는 전환점에 있다.

이 강의의 목표는 두 가지다. 첫째, 2024년 8월 1일 제정되어 2025년 8월부터 본격 시행된 EU AI법을 중심으로 한 글로벌 규제 체계를 분석하여 개발자로서 준수해야 할 사항을 명확히 이해하는 것이다. 둘째, 2025년 최신 연구인 개인정보 보호 강화 기술(PETs)을 검토하여 이러한 법률을 준수하는 책임 있는 AI를 기술적으로 구현하는 청사진을 그리는 것이다.

이번 학기에는 "EU AI법 준수 LLM 서비스 설계" 과제를 완료해야 한다. 이 강의는 해당 과제를 완료하는 데 필요한 법적, 기술적, 아키텍처적 기초를 제공한다.

---

## 1. 2025년 AI 거버넌스 및 규제 현황

2025년은 AI 규제가 추상적인 윤리 지침에서 구속력 있는 법률로 전환된 첫 해다. 이러한 변화의 중심에는 EU AI법이 있다.

### 1.1. 새로운 글로벌 표준: EU AI법의 구조와 핵심

2024년 8월 1일 발효되어 2025년부터 단계적으로 시행되고 있는 EU AI법은 세계 최초의 포괄적 AI 규제법이다. 이 법률은 "CE 마크"와 마찬가지로 EU에서 AI 시스템을 배포하거나 시장에 출시하려는 모든 "제공자(Provider)"와 "배포자(Deployer)"에 적용된다. AI 시스템이 EU 내에서 개발되었든 제3국에서 개발되었든 상관없이 적용되며, "출력(output)"이 EU 내에서 사용되는 경우에도 적용된다.

#### 1.1.1. 핵심 아키텍처: 4단계 위험 기반 접근법

EU AI법의 가장 두드러진 특징은 위험 기반 접근법으로, 모든 AI 시스템을 위험 수준에 따라 4단계로 분류한다. 규제의 엄격성은 위험 수준에 비례한다.

1. **허용 불가 위험(Unacceptable Risk):** EU의 가치와 기본권에 명확한 위협을 가하는 AI 시스템이다. 이러한 시스템은 시장에 출시하거나 서비스에 투입하거나 사용하는 것이 전면 금지된다.
2. **고위험(High Risk):** 인간의 기본권, 건강, 안전 또는 사회의 핵심 기능에 중대한 영향을 미칠 수 있는 AI 시스템이다. 법률의 대부분은 이러한 고위험 시스템에 대한 의무를 다룬다.
3. **제한적 위험(Limited Risk):** 사용자가 AI와 상호작용하고 있음을 인지해야 하는 시스템에 적용된다. 챗봇이나 딥페이크 같은 시스템이 이 범주에 해당하며, 경미한 투명성 의무가 부과된다.
4. **최소 위험(Minimal Risk):** AI 기반 비디오 게임이나 스팸 필터처럼 위험이 거의 없거나 전혀 없는 대부분의 AI 애플리케이션을 포함한다. 이러한 시스템은 사실상 규제되지 않으며, 행동 강령의 자발적 준수만 권장된다.

#### 1.1.2. 2025년 2월부터 시행: "허용 불가 위험"과 NLP 관련성

금지 조항이 법적 효력을 갖는 첫 번째 조항이었다. 2025년 2월 2일부터 "허용 불가 위험" AI의 사용이 EU에서 불법이 되었다. 이는 NLP 연구자와 개발자에게 즉각적인 영향을 미친다.

NLP와 직접 관련된 주요 금지 사항은 다음과 같다:

- **사회 점수화(Social Scoring):** 공공 기관이 개인의 사회적 행동, 신뢰도 또는 개인적 특성을 기반으로 개인을 평가하거나 분류하여 불리한 처우(예: 서비스 접근 제한)를 초래하는 시스템을 금지한다.
- **조작적 AI(Manipulative AI):** 개인의 의식 밖에서 잠재의식적 기법을 사용하거나 특정 그룹(예: 연령, 장애, 사회경제적 지위 기반)의 취약점을 악용하여 개인의 행동을 실질적으로 왜곡하는 AI 시스템을 금지한다.
- **직장 및 교육 환경에서의 감정 인식:** 2025년 2월 초 유럽위원회(EC)가 발표한 가이드라인에 따르면, 직장이나 교육 기관에서 개인의 감정을 추론하는 AI 시스템이 금지된다. 매우 구체적인 의료 또는 안전상의 이유(예: 운전자 피로 감지)를 제외하고, 이는 EU 내에서 NLP 기반 "직원 감정 분석 솔루션", "학생 스트레스 모니터링", "채용 후보자 감정 분석" 제품의 상업화를 사실상 차단한다.

이 금지 목록은 NLP 연구의 특정 방향에 대한 명확한 법적·사회적 "사형 선고"와 같다. 기술적으로는 가능하지만, EU는 그러한 기술이 사회적으로 용납될 수 없다고 선언했다.

#### 1.1.3. "고위험 AI 시스템"(HRAIS)의 주요 의무

이것은 여러분이 설계할 대부분의 상업용 NLP 시스템(특히 금융, 인사, 교육 분야)이 속할 가능성이 높은 범주다. HRAIS로 분류되면 제공자(즉, 개발자)는 시장 출시 전에 엄격한 사전 적합성 평가를 통과해야 하며 다음 주요 의무를 이행해야 한다.

EU AI법에 따른 HRAIS의 7가지 주요 의무(제8조~제17조):

1. **위험 관리 시스템(제9조):** AI 시스템의 전체 생명주기 동안 위험을 식별, 평가, 완화하기 위한 지속적인 프로세스를 수립하고 문서화해야 한다.
2. **데이터 및 데이터 거버넌스(제10조):** 차별적 결과를 최소화하기 위해 고품질의 훈련, 검증, 테스트 데이터셋을 사용해야 한다. 데이터셋은 관련성이 있고 충분히 대표적이며, 가능한 한 오류가 없고 의도된 목적에 완전해야 한다. (이 조항이 GDPR과 어떻게 충돌하는지는 3.2절에서 논의한다.)
3. **기술 문서(제11조):** 당국이 시스템의 준수 여부를 평가하는 데 필요한 모든 정보(예: 아키텍처, 성능, 데이터셋 사양)를 포함한 상세한 기술 문서를 작성하고 최신 상태로 유지해야 한다.
4. **기록 보관/로그(제12조):** 시스템의 작동을 자동으로 기록하고, 결과의 추적 가능성을 보장하기 위해 로그(예: 결정의 근거)를 생성하고 저장해야 한다.
5. **투명성/배포자 정보(제13조):** 시스템을 실제로 운영할 "배포자(Deployer)"에게 시스템의 기능, 한계, 올바른 사용 방법 및 해석 방법에 대한 명확하고 충분한 정보를 제공해야 한다.
6. **인간 감독(제14조):** 시스템은 사용 중에 적절한 인간 개입 및 감독을 허용하도록 설계되어야 한다. 인간은 AI의 결정을 중단, 무시 또는 번복할 수 있어야 한다.
7. **정확성, 견고성 및 사이버보안(제15조):** 시스템은 의도된 목적에 적합한 높은 수준의 정확성을 보여주고, 오류나 외부 적대적 공격에 견고하며, 적절한 수준의 사이버보안을 갖춰야 한다.

#### 1.1.4. [중요] NLP 특화 "고위험" 사용 사례(부록 III)

그렇다면 어떤 NLP 시스템이 "고위험(HRAIS)"인가? AI법 부록 III는 기본적으로 고위험으로 간주되는 8가지 구체적인 사용 사례를 나열한다. 이 목록은 NLP 및 프로파일링 기술이 법률의 핵심 대상임을 명확히 보여준다.

- **교육 및 직업 훈련:**
  - 교육 기관에 대한 접근, 입학 또는 배정을 결정하는 시스템(예: AI 입학 담당자, AI 지원서 심사자).
  - 학생의 학습 성과를 평가하는 시스템(예: AI 기반 자동 채점, AI 튜터의 학생 성과 분석).
- **고용, 근로자 관리 및 자영업 접근:**
  - 채용 또는 선발을 위한 AI 시스템(예: 채용 광고 타겟팅, 이력서 분석 및 필터링, 면접 후보자 평가).
  - 승진 및 해고 결정, 개인적 특성이나 행동 기반 작업 할당, 성과 모니터링 및 평가를 위한 시스템.
- **필수 민간 및 공공 서비스 접근:**
  - 신용 점수화 또는 신용도 평가를 위한 AI 시스템(금융 사기 탐지는 제외).
  - 공공 기관이 공적 혜택 및 서비스(예: 사회보장, 복지)에 대한 자격을 평가, 감소 또는 취소하는 데 사용하는 시스템.
  - 생명 및 건강 보험의 위험 평가 및 가격 책정에 사용되는 시스템.
- **법 집행:**
  - 거짓말 탐지기(polygraph) 및 유사한 도구.
  - 형사 조사나 기소 중 증거의 신뢰성을 평가하는 시스템.
  - 성격 특성이나 과거 범죄 행동을 기반으로 범죄 위험을 평가하거나 개인을 프로파일링하는 시스템.

이 목록에서 명확히 알 수 있듯이, EU AI법의 고위험 및 금지 조항은 물리적 AI(로봇, 드론)보다는 인간의 언어, 행동, 특성을 평가하고 예측하여 결과적으로 개인의 기회(입학, 채용, 대출) 접근을 결정하는 프로파일링 및 자동화된 의사결정 시스템에 더 초점을 맞추고 있다. 이는 AI법이 근본적으로 "NLP 규제법"의 성격을 매우 강하게 띠고 있음을 의미한다.

### 1.2. 2025년 쟁점: 범용 AI(GPAI) 규제

2025년 8월 2일, EU AI법의 가장 논란의 여지가 있는 조항인 범용 AI(GPAI) 모델에 대한 의무가 공식적으로 발효되었다. 이 조항은 GPT-4, Llama 3, Claude 3와 같은 대규모 모델(기초 모델이라고도 함)을 개발하고 제공하는 회사(예: OpenAI, Google, Anthropic, Meta)에 직접적인 책임을 부과한다. 2025년 11월 현재, 이것은 AI 업계에서 가장 뜨거운 규제 이슈다.

#### 1.2.1. 2025년 7월 가이드라인: "GPAI" 정의

AI법의 GPAI 의무가 발효되기 직전인 2025년 7월 18일, 유럽위원회(EC)는 이러한 의무의 범위와 정의를 명확히 하는 가이드라인 초안을 발표했다.

이 가이드라인에 따르면, 누적 계산 부하가 $10^{23}$ FLOPs(초당 부동소수점 연산) 이상이고 텍스트, 오디오 또는 이미지/비디오 생성과 같은 광범위한 작업을 수행할 수 있는 모델이 "GPAI 모델"로 정의된다.

#### 1.2.2. 모든 GPAI 제공자의 의무

"시스템적 위험"이 없는 소규모 GPAI 모델의 제공자라도 $10^{23}$ FLOPs 임계값을 초과하면 다음 네 가지 주요 의무를 준수해야 한다:

1. **기술 문서:** 모델의 훈련, 테스트, 평가 프로세스 및 결과를 설명하는 상세한 기술 문서를 준비하고 유지하며, 요청 시 AI 사무국에 제공해야 한다.
2. **하류 제공자 정보:** 하류 개발자(예: HRAIS를 구축하는 스타트업)가 자체 AI법 의무(예: HRAIS 기술 문서)를 준수할 수 있도록 모델의 기능, 한계, 사용 방법에 대한 충분한 정보를 제공해야 한다.
3. **저작권 정책:** EU 저작권법을 존중하고 준수하는 정책을 수립하고 시행해야 한다(예: 데이터 수집 중 저작권 보유자의 "옵트아웃" 요청 존중).
4. **훈련 데이터 요약:** 2025년 7월 24일 AI 사무국이 발표한 공식 템플릿에 따라 모델 훈련에 사용된 데이터의 요약을 공개적으로 게시해야 한다.

#### 1.2.3. "시스템적 위험"이 있는 GPAI의 추가 의무

이것은 GPT-4, Claude 3, Gemini Ultra와 같은 최첨단(SOTA) 대규모 모델을 대상으로 하는 특별 규제다.

- **정의:** 2025년 7월 가이드라인은 누적 계산 부하가 $10^{25}$ FLOPs 이상인 GPAI 모델이 "시스템적 위험(Systemic Risk)"을 가진 것으로 추정한다.
- **추가 의무(제55조):** 이러한 시스템적 위험 모델의 제공자(예: OpenAI, Google, Anthropic)는 위의 네 가지 기본 의무를 준수해야 할 뿐만 아니라 훨씬 더 강력한 네 가지 추가 의무를 이행해야 한다:
  1. **모델 평가 수행:** 최첨단(SOTA) 표준에 따라 모델 평가를 수행한다. 여기에는 편향, 견고성, 잠재적 오용 위험을 식별하기 위한 내부 및 외부 적대적 테스트가 포함된다.
  2. **시스템적 위험 평가 및 완화:** 모델이 야기할 수 있는 EU 수준의 시스템적 위험(예: 민주적 프로세스, 공중 보건, 국가 안보에 대한 위협)을 식별, 평가하고 적절한 완화 조치를 취해야 한다.
  3. **심각한 사고 추적 및 보고:** 모델 배포 후 발생하는 심각한 사고를 추적, 문서화하고 지체 없이 EU AI 사무국 및 관련 국가 당국에 보고해야 한다. (2025년 11월 4일, EC는 이 보고를 위한 템플릿을 발표했다.)
  4. **적절한 사이버보안 보장:** 모델 자체뿐만 아니라 모델 가중치가 저장된 물리적 인프라에 대해 적절한 수준의 사이버보안 보호를 보장해야 한다.

#### 1.2.4. 2025년 7월 "실무 규약(Code of Practice)"의 역할

제공자가 이러한 복잡하고 모호한 의무(예: "적절한" 사이버보안, "최첨단" 모델 평가)를 어떻게 준수할 수 있을까? 2025년 7월 10일, EC는 산업계, 학계, 시민사회의 독립 전문가들이 초안을 작성한 "GPAI 실무 규약"을 승인하고 발표했다.

- 이 규약은 법적으로 "자발적"이다. 제공자는 다른 방법으로 준수를 입증할 수 있다.
- 그러나 제공자가 이 규약을 준수하고 서명하면 "안전 항구(safe harbor)" 또는 "적합성 추정(presumption of conformity)"이라는 강력한 이점을 얻는다. 즉, AI법(제53조, 제55조)에 따른 의무를 이행한 것으로 간주된다.
- 규약은 세 가지 장으로 구성된다: 투명성, 저작권, (시스템적 위험 모델의 경우) 안전 및 보안. 각 의무를 이행하기 위한 구체적인 조치를 명시한다.

#### 1.2.5. [중요] 2025년 "준수 위기"

2025년 11월 현재, GPAI 규제는 법적으로 발효되었지만 실행에서 심각한 준수 위기에 직면해 있다. 이는 법률의 복잡한 시행 일정 때문이다.

- **"그랜드파더링(Grandfathering)":** 2025년 8월 2일 이전에 이미 시장에 출시된 GPAI 모델의 제공자(예: GPT-4, Llama 3, Claude 3)는 2027년 8월 2일까지 2년의 유예 기간을 받는다.
- **"하류 비극(The Downstream Tragedy)":** 그러나 2025년 8월 2일 이후에 이러한 "그랜드파더링" 모델(예: GPT-4) 중 하나를 사용하여 "고위험 AI 시스템(HRAIS)"(예: 채용 솔루션)을 구축하는 하류 개발자는 HRAIS 규제(1.1.3절 참조)를 즉시 준수해야 한다.
- **"벤더 투명성 부족":** 이것이 여러분이 직면할 문제다. HRAIS 기술 문서를 작성하려면 하류 개발자는 상류 모델(GPT-4)의 훈련 데이터 요약 및 편향 테스트 결과와 같은 정보가 필요하다. 그러나 상류 제공자(OpenAI)는 2027년까지 해당 정보를 제공할 법적 의무가 없다.

이러한 규제 공백과 공급망 위기 속에서 2025년 7월에 발표된 "실무 규약"은 단순한 "자발적" 규약이 아니라 사실상 "필수 비즈니스 인증"으로 기능한다. HRAIS를 구축하는 하류 기업은 2027년까지 기다릴 수 없기 때문에, 이 규약에 자발적으로 서명하고 필요한 문서를 제공하는 제공자(예: Google, Anthropic, Mistral)의 "안전한" 상류 모델을 선택해야 한다. 2025년 현재, 이 규약에 서명하는 것이 GPAI 시장에서 "신뢰할 수 있는 파트너"임을 입증하는 유일한 방법이 되었다.

### 1.3. 대분기: 2025년 글로벌 규제 비교

2025년은 "브뤼셀 효과(Brussels Effect)"—EU의 엄격한 표준이 사실상의 글로벌 표준이 될 것이라는 기대—라는 신화가 깨진 해다. 우리는 이제 3-4개의 명확히 구별되는 규제 블록의 시대에 있다.

#### 1.3.1. 미국: "혁신 친화" 및 규제 완화

- **배경:** 2025년 1월 취임한 트럼프 행정부는 AI를 경제적·지정학적 리더십의 핵심으로 보고, 이전 바이든 행정부의 "안전하고 신뢰할 수 있는 AI" 행정명령(E.O. 14110)을 폐지했다.
- **핵심 입장:** 2025년 7월, 백악관은 "미국의 AI 행동 계획"을 발표했다. 계획의 핵심은 "과도한 규제"를 피하고 "성장 친화적 AI 정책"을 촉진하는 것이다.
- **정책:** NIST AI 위험 관리 프레임워크(NIST AI RMF 2.0)는 법적 구속력이 없는 자발적 지침으로 남아 있다. 행정부는 심지어 NIST에 RMF에서 "오정보" 및 "다양성, 형평성, 포용성(DEI)"과 같은 "이데올로기적 편향"을 제거하도록 지시했다.
- **EU와의 마찰:** 미국 행정부는 EU AI법을 "안전에 대한 불필요한 걱정"이라고 공개적으로 비판하며, 이는 미국 기술 기업에 대한 차별이며 혁신을 억압한다고 주장했다.
- **2025년 11월 현황:** 이러한 강한 압력의 결과로, 2025년 11월 7일 EU 위원회가 AI법의 일부 조항(예: HRAIS 위반에 대한 벌금)을 2027년 8월까지 연기하거나 1년의 "유예 기간"을 부여하는 것에 대해 "검토 중"임을 확인했다는 보도가 있었다.

#### 1.3.2. 한국: 혁신과 규제의 "제3의 길"

- **법적 지위:** 한국은 2025년 1월 21일 "AI 기본법"(공식 명칭: 인공지능 기본법 및 신뢰 기반 조성에 관한 법률)을 제정하여 2026년 1월 22일부터 시행 예정이다. 이것은 EU에 이어 세계 두 번째 포괄적 AI 법률이다.
- **핵심 입장:** EU의 "위험" 중심과는 다른 "균형" 접근법을 취한다. AI 산업의 "촉진"을 우선시하며("먼저 촉진, 나중에 규제"), 공공 생활, 안전, 기본권에 중대한 영향을 미치는 "고영향 AI" 시스템에만 "최소 규제"를 부과하는 것을 목표로 한다.
- **주요 의무:** "고영향 AI"는 EU의 "고위험"과 유사하게 정의된다(예: 의료, 채용, 대출 심사). 이러한 시스템은 영향 평가 수행, 위험 관리 시스템 구축, 인간 감독 보장, AI 생성 콘텐츠 "표시" 및 사용자 알림 의무가 있다.
- **차이점:** "혁신 친화적" 규제 모델로, EU보다 낮은 벌금을 부과하며 AI 산업 육성 및 데이터 인프라(예: 훈련 데이터) 지원에 더 큰 중점을 둔다.

#### 1.3.3. 중국: 국가 중심 거버넌스

- **핵심 입장:** 중국은 EU와 미국과 완전히 다른 국가 중심적이고 "사회 통제" 지향적 접근법을 보여준다. AI를 국가 경쟁력과 사회 안정 유지를 위한 핵심 도구로 본다.
- **법적 지위:** 2023년 "생성형 AI 서비스 관리 임시 조치"를 시작으로 중국은 2025년부터 강력한 규제를 시행하고 있다.
- **주요 의무:**
  1. **콘텐츠 통제:** 콘텐츠는 핵심 사회주의 가치를 반영해야 하며 불법적이거나 유해한 콘텐츠(예: 국가 안보에 대한 위협, 공산당 비판) 생성이 방지된다.
  2. **데이터 출처:** 훈련 데이터의 합법성을 보장하고 타인의 지적재산권을 존중해야 한다.
  3. **명시적 표시:** 2025년 하반기에 시행된 "표시 조치"에 따라 모든 AI 생성 콘텐츠는 "명시적"(예: 워터마크, 텍스트 알림) 및 "암묵적"(예: 메타데이터) 표시를 모두 포함해야 한다.

#### 1.3.4. Comparative Analysis of Global AI Regulations, 2025

| Feature                 | European Union (EU)                                                                                                                   | United States (US)                                                          | South Korea (ROK)                                                        | China (PRC)                                                               |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| **Core Philosophy**     | Fundamental rights protection, Trust, Human-centric                                                                                | Market innovation, Geopolitical leadership, Deregulation                 | Balance of innovation and trust ("Promote first, regulate later")     | Social stability, State control, Technological sovereignty             |
| **Legal Status**        | **Mandatory Law (Act)** (In force Aug 2024)                                                                                        | **Voluntary Guidelines** (NIST AI RMF)                                   | **Mandatory Law (Act)** (Enforced Jan 2026)                          | **Mandatory Administrative Measures** (Partially in effect)            |
| **Risk Classification** | 4-Tier Risk-Based (Unacceptable/High/Limited/Minimal)                                                                               | Unitary Risk Management Framework (RMF)                                  | 2-Tier (High-Impact / Other)                                          | No risk-based tiers (Content-based regulation)                         |
| **GenAI Regulation**    | Strong obligations for **GPAI (>$10^{23}$ FLOPs)** & **Systemic Risk (>$10^{25}$ FLOPs)** (documentation, evaluation, reporting) | **No regulation**. (Encourages open-source, removal of ideological bias) | "Labeling" and "Transparency" obligations for Generative AI           | "Labeling" and strong "Content Control" obligations for Generative AI  |
| **2025 Status**         | GPAI rules in effect (8/2), "Supply chain crisis" begins, Discussing "postponement" due to US pressure                          | "America's AI Action Plan" (7/25), Deregulation stance established       | "AI Basic Act" enacted (1/25), Preparing enforcement decrees for 2026 | "Labeling Measures" in effect, Global governance plan announced (7/25) |

이러한 "대분기"는 2025년 현재 글로벌 AI 기업에 아키텍처적 분기를 강요하고 있다. 기업은 더 이상 "일괄 적용형" 책임 있는 AI 모델을 구축할 수 없다. 이제 최소 세 가지 다른 버전을 고려해야 한다: (1) **EU 버전:** 엄격한 문서화, 편향 감사, 인간 감독이 내장된 "고신뢰" 모델, (2) **미국 버전:** 기능과 혁신에 초점을 맞춘 "고성능" 모델, (3) **중국 버전:** 강력한 콘텐츠 필터링 및 표시가 내장된 "고통제" 모델. 이것은 법무팀의 문제가 아니라 각 지역에 대해 다른 모델 아키텍처, 데이터 거버넌스, 배포 전략이 필요한 핵심 엔지니어링 과제가 되었다.

### 체크포인트 질문

- EU AI법의 네 가지 위험 단계는 무엇이며, 규제 엄격성에서 어떻게 다른가?
- 대부분의 상업용 NLP 시스템이 "고위험 AI 시스템"(HRAIS)으로 분류될 가능성이 높은 이유는 무엇인가?
- GPAI 모델과 "시스템적 위험"이 있는 GPAI 모델의 차이는 무엇인가?
- 2025년 8월 2일 이후 HRAIS를 구축하는 하류 개발자가 직면하는 "준수 위기"를 설명하라.
- EU, 미국, 한국, 중국의 규제 접근법이 핵심 철학과 법적 지위에서 어떻게 다른가?

---

## 2. 기술 심화: LLM을 위한 개인정보 보호 강화 기술(PETs)

법적 준수는 법무팀의 서류 작업으로 끝나지 않는다. 법률이 요구하는 프라이버시와 안전은 코드로 구현되어야 한다. 이제 2025년 최신 연구를 바탕으로 LLM 시대에 개인정보 보호 강화 기술(PETs)이 어떻게 진화하고 있는지 살펴본다.

### 2.1. 차등 프라이버시(DP): "데이터"가 아닌 "패턴" 학습

차등 프라이버시(DP)는 "알고리즘의 출력(예: 모델 가중치, 예측)을 보고 특정 개인의 데이터가 훈련 세트에 포함되었는지 알 수 없다"는 것을 보장하는 강력한 수학적 정의다. 프라이버시 예산($\epsilon$ 및 $\delta$) 내에서 통계적으로 보정된 노이즈를 결과에 주입하여 개인정보 노출을 방지한다.

#### 2.1.1. LLM 시대의 위협: 임베딩 역전 공격(EIAs)

검색 증강 생성(RAG) 시스템이 보편화되면서 사용자의 민감한 쿼리(예: "탈세 법률 검색")가 텍스트 임베딩으로 변환되어 클라우드 벡터 데이터베이스로 전송된다. 과거에는 이러한 임베딩이 안전한 것으로 간주되었다.

그러나 최근 연구(2025)에 따르면 임베딩 역전 공격(EIAs)이 임베딩 벡터만으로 원본 텍스트를 상당히 재구성할 수 있다. 이는 벡터 DB에 저장된 임베딩 자체가 민감한 정보 유출이 될 수 있음을 의미한다.

2025년 3월 arXiv에 게재된 "EntroGuard"와 같은 연구가 해결책을 제시한다. 이것은 클라우드로 전송되기 전에 임베딩에 DP 기반 통계적 섭동을 주입한다. 이 노이즈는 벡터 검색(RAG) 정확도를 최대한 유지하도록 설계되면서, 동시에 EIA 공격자가 원본 민감 텍스트 대신 고엔트로피(즉, 의미 없는) 텍스트를 복구하도록 강제한다.

#### 2.1.2. 2025년 주요 트렌드 1: 차등 프라이버시 합성 데이터 생성

과거에는 DP가 주로 모델 훈련 과정에서 그래디언트에 노이즈를 주입하는 프라이빗 훈련(예: DP-SGD)에 관한 것이었다. 이것은 매우 복잡하고, 큰 프라이버시 예산을 소비하며, 결과 모델의 유용성을 크게 저하시켰다.

2025년의 새로운 패러다임은 프라이빗 데이터 생성이다. 민감한 원시 데이터에 직접 모델을 훈련시키는 대신, 이 접근법은 DP를 적용하여 원본의 통계적 속성만 포함하는 합성 데이터를 생성한다. 그런 다음 이 "안전한" 합성 데이터를 모델 훈련에 사용한다.

- **Google의 접근법(2025년 3월):** Google Research는 "추론 전용" DP 합성 데이터 생성 방법을 발표했다. 여러 민감한 원시 데이터 포인트(예: 사용자 쿼리)를 가져와 프롬프트로 포맷하고 사전 훈련된 LLM에 입력한다. LLM의 다음 토큰 예측(logits)은 DP 메커니즘(예: 지수 메커니즘)을 통해 "프라이빗하게 집계"되어 합성 다음 토큰을 샘플링한다. 이 프로세스를 반복하여 원본 패턴을 따르지만 개인 프라이버시를 보호하는 합성 데이터를 생성한다.
- **Microsoft/ICLR 연구(2024):** 사전 훈련된 기초 모델(예: GPT-4)을 "블랙박스" API로 취급하고 DP 쿼리를 통해 합성 데이터를 생성하는 "훈련 없는" 접근법을 제시했다.

#### 2.1.3. 2025년 주요 트렌드 2: 프라이빗 집계 트렌드 분석

DP의 상업적 적용의 정점은 2025년 6월 발표된 Apple Intelligence에 나타났다. Apple은 "사용자 데이터를 절대 수집하지 않는다"는 강력한 프라이버시 원칙을 유지하면서 동시에 "사용자 경험을 개선"해야 하는 모순적인 도전에 직면했다.

- **Apple의 접근법:**
  1. **기기 내 처리:** Apple은 사용자의 기기 내 데이터(예: 이메일 내용, 알림)를 수집하지 않는다. "이메일 요약"과 같은 기능을 개선하는 데 필요한 분석은 사용자의 기기에서 로컬로 수행된다.
  2. **프라이빗 업데이트:** 모델 개선에 유용한 "트렌드" 또는 "업데이트"(예: 그래디언트, 특정 패턴)가 기기에서 생성된다.
  3. **DP 노이즈 주입:** 이 "업데이트"는 기기를 떠나기 전에 DP 알고리즘을 통해 "익명화"된다.
  4. **집계:** Apple의 서버는 이러한 익명화된 집계 트렌드만 수신한다. 개별 사용자의 원시 데이터는 수신하지 않는다. 이 집계 정보는 통계적으로 유의미하지만, 수학적으로 특정 개인을 식별하는 것은 불가능하다.
- **결과:** DP를 통해 Apple은 두 가지 모순적인 목표를 동시에 달성한다: (1) "사용자 데이터를 보지 않는다"는 강력한 프라이버시 마케팅 주장과 (2) "사용자 데이터로 모델을 개선한다"는 엔지니어링 목표. DP는 이러한 모순을 해결하는 법적·윤리적 방패다.

2025년 현재, DP의 주류 적용은 "모델 훈련" 자체의 복잡성에서 벗어나 "DP 합성 데이터 생성"과 "DP 트렌드 분석"으로 이동했다. 이것은 DP를 기술적 도전에서 비즈니스 프로세스 솔루션으로 격상시켰다.

### 2.2. 연합 학습(FL): 데이터 이동 없이 훈련

연합 학습(FL)은 프라이버시에 대한 근본적으로 다른 접근법이다. 데이터를 중앙 서버로 보내는 대신, 모델(또는 모델 업데이트)이 각 클라이언트(예: 스마트폰, 병원, 은행)로 전송되어 로컬 데이터로 훈련된다. 훈련된 모델 가중치(또는 그래디언트)만 서버로 다시 전송되어 집계된다. 데이터는 항상 로컬에 머문다.

#### 2.2.1. LLM의 도전: 통신 및 계산 병목

초기 FL(예: FedAvg 알고리즘)은 작은 모델(예: 모바일 키보드 예측)을 가정했다. 그러나 70B 또는 175B 파라미터를 가진 LLM(예: Llama 3, GPT-3.5)의 등장으로 전통적인 FL은 거의 불가능해졌다.

- **통신 비용:** 전체 LLM(수백 GB)을 각 클라이언트(예: 병원)로 보낸 다음 그래디언트 업데이트(수백 GB)를 서버로 다시 보내는 것은 엄청난 네트워크 대역폭이 필요하다.
- **계산 비용:** 각 클라이언트(예: 개별 병원 서버)는 70B 모델을 파인튜닝하는 데 필요한 고성능 GPU 인프라를 갖춰야 한다.
- **데이터 이질성(Non-IID):** 각 클라이언트의 데이터 분포는 매우 이질적이며(Non-IID, 독립적이고 동일하게 분포되지 않음), 단순한 FedAvg 접근법은 수렴에 실패하거나 모델 성능을 저하시킬 수 있다.

#### 2.2.2. 2025년 해결책 1: "연합 PEFT"

2025년 현재, LLM과 함께 FL을 사용하는 가장 유망한 해결책은 PEFT(파라미터 효율적 파인튜닝), 특히 LoRA(저순위 적응)와의 결합이다.

- **핵심 아이디어:** 전체 70B 모델을 연합하지 않는다. 70B 파라미터 사전 훈련 LLM은 고정된다. 훈련 가능한 파라미터 수를 대폭 줄이는 경량 LoRA 어댑터만 연합된다.
- **작동 방식:**
  1. **분배:** 서버는 거대한 "고정된" 사전 훈련 모델(예: Llama 3 70B)을 모든 클라이언트(예: 병원 A, B, C)에 한 번만 분배한다.
  2. **로컬 훈련:** 각 클라이언트(병원 A)는 작은 LoRA 어댑터(예: 원본 모델 크기의 0.1%, 약 20-100MB)만 자체 로컬 프라이빗 데이터(환자 기록)로 훈련한다. 70B 모델 본체는 건드리지 않는다.
  3. **업데이트:** 클라이언트는 70B 파라미터(수백 GB)가 아닌 20MB LoRA 어댑터 가중치만 서버로 다시 보낸다.
  4. **집계:** 서버는 전 세계 병원에서 수집한 작은 LoRA 어댑터만 "평균화"(예: FedAvg)하여 "글로벌 LoRA 어댑터"를 생성한다.
  5. 이 "글로벌 어댑터"는 다음 훈련 라운드를 위해 클라이언트로 다시 전송된다.
- 이 방법은 통신 비용을 수천 배 줄이고 Non-IID 데이터 환경에서도 강한 성능을 보여준다.

#### 2.2.3. 2025년 해결책 2: "레이어 스킵 FL"

이것은 2025년 4월 arXiv에 게재된 또 다른 PEFT 기반 접근법이다.

- **핵심 아이디어:** LoRA 어댑터를 추가하는 대신, 이 방법은 사전 훈련된 LLM의 일부 레이어를 고정(스킵)하고 선택된 특정 레이어만 파인튜닝한다.
- **성능:** LLaMA 3.2-1B 모델에 적용했을 때, 이 접근법은 통신 비용을 약 70% 줄이면서 성능 저하를 중앙 집중식 훈련의 2% 이내로 유지했다. 이것은 의료 NLP(예: i2b2, MIMIC-III 데이터셋)와 같이 프라이빗 데이터를 공유하지 않고 여러 기관이 도메인 특정 데이터에서 협력적으로 훈련하는 데 매우 실용적인 해결책임이 입증되었다.

"PEFT (LoRA) + FL"의 결합은 단순한 최적화가 아니라 패러다임 전환이다. "글로벌 일반화"와 "로컬 전문화"를 동시에 달성한다. 서버는 "글로벌 LoRA"를 통해 일반 도메인 지식(예: 의학)을 개선하는 한편, 각 클라이언트(병원)는 자체 데이터에 매우 특화된 자체 "로컬 LoRA"를 유지한다. 따라서 연합 학습은 중앙 모델을 개선하면서 각 클라이언트에 "맞춤형 프라이빗 모델"을 제공하는 이중 목적 솔루션으로 진화했다.

### 2.3. 동형 암호화(HE): "성배"의 실용화

동형 암호화(HE)는 암호화된 데이터(암호문)에서 직접 원하는 계산(덧셈, 곱셈 등)을 수행할 수 있게 하는 "꿈의" 암호화 기술이다. 암호화된 결과를 복호화하면 원본 평문 데이터에서 연산을 수행한 것과 동일한 출력을 얻는다. 이를 사용하면 클라이언트가 민감한 데이터를 암호화하여 서버로 보낼 수 있고, 서버는 원시 데이터를 전혀 보지 않고도 연산(예: LLM 추론)을 수행한 다음 암호화된 결과를 클라이언트로 반환할 수 있다.

#### 2.3.1. 실용성 장벽: 10,000배 이상의 오버헤드

HE는 LLM과 같은 대규모 신경망에 적용할 때 치명적인 결함이 있다: 엄청난 계산 오버헤드.

- 2025년 논문에 따르면 HE 기반 LLM 추론은 평문 추론보다 최소 10,000배 느리다.
- **이유:** 완전 동형 암호화(FHE)는 선형 연산(예: nn.Linear, 행렬 곱셈)에 대해 상대적으로 효율적이다. 그러나 Transformer 아키텍처의 핵심인 비선형 활성화 함수(예: ReLU, GeLU, SiLU, Softmax)에 대해서는 극도로 비효율적이다.
- 암호화된 상태에서 이러한 비선형 연산을 근사하면 암호문에 누적된 노이즈가 기하급수적으로 증가한다. 이를 재설정하려면 부트스트래핑이라는 초고비용 연산이 필요하다. LLM은 수백 개의 레이어를 가지고 있어 단일 추론에 수백 또는 수천 번의 부트스트래핑 연산이 필요할 수 있다.

#### 2.3.2. 2025년 해결책 1: HE 친화적 모델 아키텍처

이를 해결하기 위해 표준 Transformer를 단순히 암호화하는 대신 모델 아키텍처 자체를 HE 연산에 더 친화적으로 변경하는 연구가 진행 중이다.

- **비선형 함수 교체:** ReLU 또는 GeLU 활성화 함수는 HE에서 쉽게 계산할 수 있는 저차 다항식 근사로 교체된다.
- **어텐션 메커니즘 변경:** Softmax를 포함하는 복잡한 어텐션 메커니즘은 계산을 최적화하기 위해 가우시안 커널 또는 단순 다항식 어텐션으로 교체된다.
- 2024년 10월 arXiv 연구에 따르면 LoRA 파인튜닝과 가우시안 커널을 결합하면 HE 기반 Transformer의 추론 속도를 2.3배, 파인튜닝 속도를 6.94배 개선할 수 있다.

#### 2.3.3. 2025년 해결책 2: "Safhire" 하이브리드 HE 추론

2025년 9월 arXiv에 게재된 "Safhire"는 현재까지 가장 실용적인 해결책을 제시한다.

- **핵심 아이디어:** HE가 잘하는 것(선형 연산)과 잘하지 못하는 것(비선형 연산)을 분리하고 서버와 클라이언트가 작업을 공유한다.
- **작동 방식:**
  1. **클라이언트:** 입력을 암호화($Enc(x)$)하여 서버로 보낸다.
  2. **서버(암호화):** 암호화된 상태에서 HE 친화적 선형 연산(예: nn.Linear)만 수행한다. ($Enc(z) = W \cdot Enc(x) + b$)
  3. **서버:** 비선형 활성화(예: ReLU)가 필요한 시점에 암호화된 결과($Enc(z)$)를 클라이언트로 다시 보낸다.
  4. **클라이언트(평문):** $Enc(z)$를 복호화하여 $z$를 얻고, HE에 불친화적인 비선형 연산 $a = ReLU(z)$를 로컬에서 평문으로 빠르게 수행한다.
  5. **클라이언트:** 활성화된 결과 $a$를 다시 암호화($Enc(a)$)하여 서버로 보내 다음 레이어의 선형 연산을 요청한다.
- 이 "클라이언트-서버-클라이언트" 왕복은 비용이 많이 드는 부트스트래핑을 완전히 제거하여 HE 추론을 실용적인 수준으로 낮춘다.
- 이 하이브리드 접근법은 HE 트레이드오프를 "극도의 계산 오버헤드"에서 "관리 가능한 네트워크 지연 오버헤드"로 전환했다. 이것은 RAG와 같은 실제 서비스에 HE를 적용하는 문을 열었다.

### 체크포인트 질문

- 차등 프라이버시란 무엇이며, 패턴 학습을 허용하면서 개인 데이터를 어떻게 보호하는가?
- 임베딩 역전 공격(EIAs)이 RAG 시스템을 어떻게 위협하며, DP는 이 위협을 어떻게 완화하는가?
- 차등 프라이버시의 "프라이빗 훈련"(DP-SGD)과 "프라이빗 데이터 생성" 접근법의 차이를 설명하라.
- 대규모 언어 모델에 연합 학습을 적용하는 주요 도전 과제는 무엇이며, "연합 PEFT"는 이를 어떻게 해결하는가?
- 동형 암호화가 LLM에 대해 계산적으로 비용이 많이 드는 이유는 무엇이며, "Safhire" 하이브리드 접근법은 이 문제를 어떻게 해결하는가?

---

## 3. 산업 사례 연구: 도메인 특화 NLP 솔루션 설계

이제 1장의 규제와 2장의 기술을 결합하여 특정 산업 도메인에서 책임 있는 LLM 솔루션을 설계하는 구체적인 청사진을 검토한다.

### 3.1. 의료: HIPAA 준수 LLM 챗봇 설계

#### 3.1.1. 규제 및 문제

- **법률:** 미국 HIPAA(건강보험 이동성 및 책임법).
- **핵심 개념:**
  1. **PHI(보호 건강 정보):** HIPAA는 18가지 개인 식별자를 PHI로 정의한다(예: 이름, 모든 유형의 날짜, 전화번호, 주소, 의료 기록 번호 등).
  2. **BAA(비즈니스 협력자 계약):** "피적용 기관(Covered Entity)"(예: 병원)이 PHI 처리를 제3자 "비즈니스 협력자"(예: 클라우드 제공자, EMR 벤더, AI 회사)에게 위탁할 때 필요한 법적 계약이다. 이 계약은 법적으로 제3자도 HIPAA 보안 규칙을 준수하도록 보장한다.
- **문제:** OpenAI(ChatGPT) 및 Anthropic과 같은 대부분의 공개 LLM API 제공자는 표준 서비스에 대해 BAA를 서명하지 않는다. 따라서 의사가 환자 차트(PHI 포함)를 복사하여 ChatGPT 웹 인터페이스에 붙여넣고 "이 환자의 기록을 요약하라"고 요청하면, BAA 없이 PHI가 제3자에게 전송되었기 때문에 심각한 HIPAA 위반이다.

#### 3.1.2. 기술적 해결책: "비식별화 + 자체 호스팅 RAG" 아키텍처

이를 해결하는 방법은 BAA를 서명할 벤더(예: Google의 Med-PaLM 2 또는 BastionGPT와 같은 의료 특화 벤더)를 사용하거나 PHI가 기관의 방화벽을 절대 벗어나지 않도록 하는 자체 호스팅 아키텍처를 구축하는 것이다.

**HIPAA 준수 LLM 챗봇 아키텍처 청사진:**

1. **인프라:** HIPAA 준수 클라우드(예: AWS, Azure, GCP) 내에 격리된 VPC(가상 사설 클라우드)를 구축한다. 모든 서비스(LLM, DB, API)는 이 VPC 내부 네트워크 내에서만 통신한다.
2. **암호화:** HIPAA는 전송 중 및 저장 중 데이터 암호화를 요구한다. 전송 중 데이터를 보호하기 위해 TLS 1.3 이상을 사용하고, DB의 저장 중 데이터를 보호하기 위해 AES-256 및 FIPS 140-2 검증 암호화 모듈을 사용한다.
3. **데이터베이스:** 환자의 원본 PHI(예: EMR/EHR)는 이 VPC 내의 암호화된 RDBMS 또는 벡터 DB에 저장된다.
4. **LLM:** Llama 3 또는 (BAA 적용) Med-PaLM 2와 같은 모델이 VPC 내에서 자체 호스팅되어 데이터가 기관의 방화벽을 절대 벗어나지 않도록 보장한다.
5. **접근 제어:** RBAC(역할 기반 접근 제어) 및 MFA(다중 인증)를 사용하여 "최소 권한 원칙"을 적용하고 접근을 엄격히 제어하여 승인된 의료진(예: 담당 의사)만 해당 환자의 PHI에 접근할 수 있도록 한다.
6. **감사:** PHI에 대한 모든 접근 시도와 모든 AI 쿼리 및 응답은 불변 감사 로그에 기록되어야 한다.

**핵심 NLP 파이프라인: 비식별화(De-ID) 게이트웨이:**

이것은 PHI 유출을 방지하는 기술적 핵심이다.

1. **(입력 쿼리):** 의사가 챗봇에 "2025년 10월 1일(PHI) 환자 John Snow(PHI)의 심장 검사 결과를 요약하라"고 요청한다.
2. **(비식별화 필터):** 이 쿼리가 LLM으로 가기 전에 비식별화(De-identification) 엔진을 통과한다. 이 엔진은 고성능 NER(명명된 개체 인식) 모델을 사용하여 18가지 PHI 식별자를 실시간으로 감지한다.
3. **(마스킹/난독화):** 감지된 PHI는 정책에 따라 마스킹(예: `[*******]`, `<NAME>`)되거나 난독화(예: "John Snow" → "Michael Willian")된다.
4. **(익명화된 쿼리):** 익명화된 쿼리 "환자 <NAME>의 <DATE> 심장 검사 결과를 요약하라"가 RAG 시스템으로 전달된다.
5. **(RAG + LLM):** RAG 시스템은 환자의 (암호화된) 실제 기록을 검색하여 LLM에 컨텍스트를 제공하고, VPC 내의 자체 호스팅 LLM이 이 컨텍스트를 기반으로 요약을 생성한다.
6. **(출력):** 생성된 요약이 의사에게 반환된다(필요한 경우 UI가 `<NAME>`을 "John Snow"로 재식별할 수 있다).

이 "비식별화 → RAG → 자체 호스팅 LLM" 스택은 2025년 의료 AI의 표준 아키텍처다. 두 가지 핵심 문제를 동시에 해결한다: (1) HIPAA 준수를 위해 PHI를 보호하고, (2) LLM이 실제 EMR 데이터(RAG)를 기반으로 답변하도록 강제하여 LLM 환각을 방지한다.

### 3.2. 금융: GDPR 및 EU AI법 준수 신용 점수화

#### 3.2.1. 규제 및 문제

- **법률 1: EU AI법(HRAIS):** "신용 점수화"는 EU AI법 부록 III에 따라 명시적으로 "고위험"으로 분류된다. 따라서 1.1.3절에서 논의한 7가지 주요 의무(제10조 데이터 거버넌스, 제13조 투명성, 제14조 인간 감독 등)가 모두 적용된다.
- **법률 2: EU GDPR(데이터 보호):**
  - **제22조:** 개인이 "법적 또는 유사하게 중대한 효과"를 초래하는 "순수 자동화 처리에 기반한" 결정(예: AI에 의한 자동 대출 거부)의 대상이 되지 않을 권리를 규정한다. 또한 "의미 있는 정보"(즉, 설명)와 "인간 개입"에 대한 권리를 보장한다.
  - **제9조(GDPR) vs. 제10조(AI법) 충돌:** AI법 제10조는 고위험 시스템이 편향을 감지하고 수정하기 위해 "특수 범주 데이터"(예: 인종, 민족, 정치적 성향)를 처리하는 것을 "엄격하게 필요할 때" 허용한다. 그러나 GDPR 제9조는 원칙적으로 동일한 민감 데이터의 처리를 금지한다. (이 충돌을 해결하려면 GDPR 제9조의 예외(예: 명시적 동의, 실질적 공공 이익)와 AI법 제10조의 조건을 동시에 충족해야 한다.)
- **문제:** 금융 회사는 대출 연체 예측 정확도를 1%만큼이라도 높이기 위해 복잡한 블랙박스 모델(예: XGBoost, 딥러닝)을 선호한다. 그러나 모델이 복잡할수록 GDPR 제22조가 요구하는 "의미 있는 설명"을 제공하기가 더 어려워진다.

#### 3.2.2. 기술적 해결책: XAI를 준수 도구로 활용

이 딜레마를 해결하기 위해 XAI(설명 가능한 AI)는 모델 디버깅 도구뿐만 아니라 법적 요구사항을 충족하기 위한 필수 준수 엔진으로 사용된다.

- **주요 도구: SHAP 및 LIME**
  - **LIME(로컬 해석 가능한 모델 독립적 설명):** 개별 예측 주변의 가상 샘플(섭동)을 생성하고(예: "이 고객이 거부된 이유는 무엇인가?"), 해당 "로컬" 영역에서만 작동하는 간단한 "대리 모델"(예: 선형)을 구축하여 설명을 제공한다.
  - **SHAP(Shapley 가산 설명):** 협력 게임 이론의 "Shapley 값"을 사용하여 각 특성(예: 소득, 부채 비율, 연체 이력)이 최종 예측(대출 승인/거부)에 얼마나 "기여했는지"(긍정적 또는 부정적)를 정확히 계산한다.
- **실용적 적용: "불리한 조치 통지" 자동 생성**
  - GDPR 제22조와 미국 ECOA(균등 신용 기회법)는 대출이 거부된 고객에게 거부의 "구체적이고 정확한" 이유를 제공하도록 요구한다(미국에서는 이것이 "불리한 조치 통지"다).
  - XAI는 이 통지 생성을 자동화하는 데 사용된다:
    1. **(예측):** 블랙박스 모델(예: XGBoost)이 고객 A의 대출을 "거부"한다.
    2. **(XAI 실행):** 이 "거부" 직후, 고객 A의 데이터에 대해 SHAP이 실행된다. SHAP은 결정에 가장 큰 부정적 영향을 미친 특성을 계산한다(예: `debt_to_income_ratio: +0.4`, `recent_inquiries: +0.2`, `age_of_oldest_account: +0.1`).
    3. **("번역" 레이어):** 비즈니스 로직 레이어가 이 수학적 SHAP 값(예: $+0.4$)을 법적으로 준수하는 인간 언어(이유 코드)로 번역한다.
    4. **(최종 통지):** "귀하의 대출 신청이 거부되었습니다. 주요 이유는 다음과 같습니다: (1) 높은 부채 대 소득 비율(SHAP $+0.4$), (2) 최근 신용 조회가 너무 많음(SHAP $+0.2$)."

금융 AI에서 XAI(SHAP/LIME)는 더 이상 선택적 "디버깅" 도구가 아니라 GDPR 제22조를 준수하기 위한 필수 법적 준수 레이어다. 그러나 이것은 블랙박스 문제를 해결한 것이 아니라 "블랙박스 문제를 이동시킨" 것이다. 2025년 11월 현재, 새로운 법적 위험이 등장하고 있다. 법적 전투는 더 이상 모델 자체가 아니라 설명의 유효성에 관한 것이다. "왜 LIME이 아닌 SHAP을 사용했는가?", "왜 SHAP '기준선'을 평균값 대신 0으로 설정했는가?", "SHAP 값 $+0.4$를 '주요 이유'로 '번역'하는 근거는 무엇인가?" 이것들이 새로운 공격 지점이다. 즉, 엔지니어는 이제 "2차 설명 부담"을 지게 되었다. 설명 자체의 정확성과 안정성을 방어할 준비가 되어 있어야 한다.

### 3.3. 교육: FERPA 준수 AI 튜터 설계

#### 3.3.1. 규제 및 문제

- **법률:** 미국 FERPA(가족 교육권 및 프라이버시법).
- **핵심 개념:** 학생의 "교육 기록"에 포함된 PII(개인 식별 정보)를 보호한다. 여기에는 성적과 출석뿐만 아니라 학생과 AI 튜터 간의 채팅 로그, 학생의 학습 패턴에 대한 AI 분석, 성과 분석 데이터도 포함되며, 이 모든 것이 "교육 기록"으로 간주될 수 있다.
- **문제:** 학교가 학생 데이터를 제3자(AI 벤더)에게 제공할 때, 해당 벤더는 FERPA 예외 하에서 "합법적인 교육적 이익"을 가진 "학교 공무원"으로 행동해야 한다. 그러나 벤더가 이러한 학생 채팅 로그를 수집하여 자체 범용 모델 훈련에 사용하면, 계약된 "교육 목적" 범위를 벗어나기 때문에 심각한 FERPA 위반이 될 수 있다.

#### 3.3.2. 기술적 해결책: "비훈련 + RAG" 아키텍처

이 아키텍처는 2025년 11월 AWS와 파트너십을 맺고 Loyola Marymount University(LMU)가 구축한 AI Study Companion 사례 연구에서 잘 입증되었다.

- **핵심 원칙:** "귀하의 데이터로 훈련하지 않음."
- **FERPA 준수 AI 튜터 청사진:**
  1. **인프라:** 모든 인프라는 대학 자체 클라우드(예: AWS) 계정 내에 구축되어 대학이 데이터에 대한 제어권을 유지하도록 한다.
  2. **지식 베이스:** LLM은 학생의 PII 데이터가 아닌 대학 소유의 강의 자료(예: 강의 전사본, 강의 노트, 강의 계획서, 교과서, 과제 가이드)에서 학습한다. 이 자료는 전사(예: Amazon Transcribe)되고 청크로 나뉘어 S3 버킷에 저장된다.
  3. **RAG(검색 증강 생성):** RAG 인덱스는 이러한 "강의 자료"에서만 구축된다(예: Amazon OpenSearch에서).
  4. **LLM(비훈련):** 관리형 기초 모델(예: Claude 3)이 Amazon Bedrock과 같은 API를 통해 호출된다. 핵심은 이 LLM이 학생의 프롬프트나 채팅 로그에서 학습하지 않도록 하는 것이다("비훈련" 원칙). LLM은 RAG가 제공하는 "강의 자료" 컨텍스트를 기반으로만 답변을 생성한다(예: "이 강의의 3주차 노트를 기반으로 답변하되, 인터넷은 사용하지 마라").
  5. **프라이버시:** 학생의 PII(로그인 정보, 채팅 로그)는 "교육 기록"으로 취급되며 LLM 훈련에 사용되지 않고 별도의 안전한 DB에 암호화되어 저장된다. "최소 권한 원칙"이 적용되어 RAG 검색을 필터링하므로 학생은 자신이 등록한 강의 자료에 대해서만 접근하고 질문할 수 있다.

HIPAA 준수 아키텍처와 FERPA 준수 아키텍처는 놀랍도록 유사하다. 이는 두 법률(HIPAA, FERPA)의 핵심 문제가 "민감 데이터에 대한 제어"(PHI, PII)이기 때문이다. 문제는 "AI가 민감 데이터에서 학습하는 것"이므로 해결책은 "AI가 민감 데이터에서 학습하는 것을 방지하는 것"이다.

결론적으로, "자체 호스팅(또는 프라이빗 클라우드) RAG + 비식별화/PII 필터" 패턴은 2025년 현재 규제 산업(의료, 금융, 교육)을 위한 표준 LLM 아키텍처다. 이 아키텍처는 세 가지 핵심 문제를 동시에 해결한다: (1) 법적 준수(민감 데이터 격리 및 제어), (2) 프라이버시("비훈련" 원칙), (3) 신뢰성(RAG를 통한 환각 방지).

### 체크포인트 질문

- HIPAA 준수 LLM 챗봇 아키텍처의 주요 구성 요소는 무엇이며, 비식별화 게이트웨이가 중요한 이유는 무엇인가?
- GDPR 제9조와 EU AI법 제10조 간의 충돌이 신용 점수화 시스템에 어떤 도전을 만드는가?
- XAI(SHAP/LIME)가 GDPR 제22조를 위한 준수 도구로 어떻게 작동하며, XAI 사용에서 어떤 새로운 법적 위험이 발생하는가?
- FERPA 준수 AI 튜터에서 "비훈련" 원칙이 무엇이며, RAG는 이를 어떻게 가능하게 하는가?
- HIPAA, FERPA, GDPR 준수 아키텍처가 서로 다른 규제에도 불구하고 유사한 패턴을 공유하는 이유는 무엇인가?

---

## 4. 워크샵 가이드(핵심 실습/과제)

이제 1장(법률), 2장(기술), 3장(사례 연구)을 학습했다. 이를 바탕으로 "EU AI법 및 기타 관련 규제를 준수하는 LLM 서비스 설계"를 초안으로 작성한다.

### 4.1. 과제 시나리오

- **여러분은:** EU 시장 진출을 계획하는 AI 스타트업의 개발팀 리더다.
- **제품:** 아래 "고위험" 시나리오 중 하나를 선택한다.
  1. **금융:** $10^{24}$ FLOPs로 훈련된 자체 내부 GPAI 모델을 기반으로 한 EU 전역 중소기업(SME)을 위한 "신용 점수화" 솔루션.
  2. **교육:** 상용 GPAI 모델 API(예: GPT-4o, Claude 3.5)를 기반으로 EU 대학생의 작문을 평가하고 피드백을 제공하는 "AI 작문 튜터" 솔루션.
  3. **고용:** 파인튜닝된 상용 GPAI 모델(예: Llama 3)을 기반으로 대규모 EU 기업의 채용 공고에 대한 수천 개의 이력서를 분석하고 순위를 매기는 "AI HR" 솔루션.
- **과제:** 선택한 시나리오에 대해 모델 개발부터 배포까지의 단계를 다루는 "EU AI법 준수 체크리스트"를 작성하고, 해당 기술/정책 조치를 취한 이유를 제시한다.

### 4.2. EU AI법을 위한 실용적 준수 체크리스트

설계 요약서는 최소한 다음 항목에 대한 답변을 포함해야 한다.

#### 1단계: 시스템 분류

- [ ] **위험 단계 식별:** 우리 시스템이 AI법에서 어떤 단계에 속하는가?
  - (예시 답변: 시나리오 1(신용 점수화), 2(학습 성과 평가), 3(채용)은 모두 부록 III에 따라 명시적으로 "고위험"이다.)
- [ ] **GPAI 모델 식별:** 우리 시스템이 기반으로 하는 모델이 GPAI인가?
  - (예시 답변: 시나리오 1은 $10^{24}$ FLOPs > $10^{23}$ FLOPs이므로 "GPAI"다. 시나리오 2와 3은 상용 모델을 사용하므로 해당 제공자가 "GPAI 제공자"다.)
- [ ] **시스템적 위험 식별:** 기본 모델에 "시스템적 위험"이 있는가?
  - (예시 답변: 시나리오 1은 $10^{24}$ FLOPs < $10^{25}$ FLOPs이므로 시스템적 위험이 있는 것으로 추정되지 않는다. 시나리오 2와 3의 경우, 기본 모델(GPT-4o 등)이 $10^{25}$ FLOPs를 초과하면 "시스템적 위험" 모델이다.)

#### 2단계: GPAI 제공자 의무(해당되는 경우)(제53조)

(시나리오 1처럼 GPAI를 직접 개발했거나 시나리오 3처럼 기본 모델을 "실질적으로 수정"한 경우, "GPAI 제공자"가 될 수 있다.)

- [ ] **기술 문서(제53조):** 모델의 훈련/평가 프로세스를 문서화했는가?
- [ ] **저작권 정책(제53조):** EU 저작권법을 준수하는 정책을 수립했는가(예: 옵트아웃 존중)?
- [ ] **데이터 요약(제53조):** AI 사무국 템플릿을 사용하여 훈련 데이터 요약을 게시할 준비가 되어 있는가?
- [ ] **실무 규약:** 2025년 7월 GPAI "실무 규약"에 서명하여 준수를 입증할 것인가? (하류 파트너 확보에 유리함)

#### 3단계: 시스템적 위험 의무(해당되는 경우)(제55조)

(시나리오 2처럼 시스템이 >$10^{25}$ FLOPs 모델을 기반으로 하는 경우. 참고: 이 의무는 주로 "상류" 제공자(OpenAI)에게 있지만, "하류" 사용자로서 그 이행을 확인해야 한다.)

- [ ] **모델 평가(제55조):** 기본 모델 제공자가 적대적 테스트 등을 수행했는지 확인했는가?
- [ ] **위험 완화(제55조):** 식별된 시스템적 위험(예: 편향)을 완화하기 위한 조치가 있는가?
- [ ] **사고 보고(제55조):** 심각한 사고를 보고하기 위한 시스템이 있는가?
- [ ] **사이버보안(제55조):** 모델 및 인프라에 대한 사이버보안 조치가 마련되어 있는가?

#### 4단계: HRAIS 제공자 의무(필수!)(제8조~제15조)

(시나리오 1, 2, 3에 필수. 여러분은 "고위험 AI 시스템"의 제공자다.)

- [ ] **위험 관리(제9조):** AI 생명주기를 위한 위험 관리 시스템을 수립했는가? (예: 정기적 위험 평가 및 완화 계획)
- [ ] **데이터 거버넌스(제10조):**
  - [ ] (시나리오 1, 3) 훈련 데이터가 특정 성별, 인종 또는 국적에 대해 편향되지 않았음을 어떻게 입증하는가? (예: 데이터셋 대표성 분석)
  - [ ] (시나리오 2, 3) GDPR(제9조)에 따라 학생/지원자의 PII 및 민감 데이터를 처리할 합법적 동의를 받았는가?
  - [ ] (시나리오 1, 3) 편향 수정을 위해 민감 데이터(예: 인종)를 사용해야 하는 경우, AI법(제10조 제5항)과 GDPR(제9조) 간의 충돌을 어떻게 해결했는가? (예: 명시적 동의 + 엄격한 목적 제한)
- [ ] **기술 문서(제11조):** HRAIS에 대한 모든 기술 문서를 준비했는가? (아키텍처, 사용된 GPAI 모델, 데이터셋 정보, 평가 결과)
  - (경고: 2025년 11월 현재, 상류 GPAI 제공자가 이 정보를 제공하지 않을 수 있다(1.2.5절 참조). 이 "공급망 위기"를 어떻게 해결할 것인가?)
- [ ] **로깅(제12조):** 모든 시스템 결정(예: 대출 거부, 이력서 거부)과 그 근거를 추적 가능성을 위해 기록하는가?
- [ ] **투명성(제13조):**
  - [ ] (시나리오 2, 3) 시스템 운영자(교사, 인사팀)에게 명확한 사용 지침(예: "이것은 참고용이며, 최종 결정은 여러분이 내린다")과 한계(예: "이 모델은 특정 작문 유형에 약하다")를 제공하는가?
  - [ ] (시나리오 1) XAI를 사용하여 신용 평가에 대한 "거부 이유"를 제공할 수 있는가? (GDPR 제22조와 연결)
- [ ] **인간 감독(제14조):**
  - [ ] 자동화된 결정(예: 이력서 자동 거부, 대출 자동 거부)을 중단, 무시 또는 번복할 수 있는 "인간 감독" 메커니즘이 있는가?
  - [ ] (시나리오 3) "AI는 상위 10%를 추천하지만, 인간이 최종 결정을 내린다." — 이것이 AI법이 요구하는 "의미 있는" 인간 감독인가, 아니면 인간이 AI를 맹목적으로 따르는 "고무 도장(rubber-stamping)"인가? (이 설계 선택을 방어해야 한다.)
- [ ] **견고성/보안(제15조):** 시스템이 적대적 공격(예: 프롬프트 주입, 이력서에 흰색 텍스트로 키워드 추가)에 견고하며 사이버보안이 보장되는가?

### 4.3. PET 통합을 위한 의사결정 프레임워크

설계 요약서는 특정 PET를 선택(또는 선택하지 않음)한 기술적 근거를 포함해야 한다.

- **옵션 1: 프라이빗 아키텍처(기본값)**
  - **설계:** "자체 호스팅(또는 VPC) RAG + 비식별화/PII 필터" (3.1절 및 3.3절의 아키텍처).
  - **선택 이유:** 이것은 2025년 규제 준수를 위한 가장 간단하고 견고한 방법이다. 민감 데이터(PII, PHI)를 격리된 DB에 저장하고 LLM을 "상태 없는" 추론 엔진으로만 사용하여 민감 데이터로의 훈련을 피한다. 이것은 AI법 제10조(데이터 거버넌스)와 GDPR을 모두 충족하는 가장 신뢰할 수 있는 방법이다. (시나리오 2 및 3에 강력히 권장)
- **옵션 2: 차등 프라이버시(DP)**
  - **설계:** (옵션 1에 추가) 서비스 중 수집된 민감 데이터(예: 학생 작문)를 사용하여 "범용 작문 모델"을 재훈련하고 개선해야 할 때 사용한다.
  - **선택 이유:** 서비스 품질을 개선하기 위해 민감 데이터로 재훈련이 필요할 때 사용한다. Apple처럼 로컬 기기에서 DP 적용 "트렌드"만 수집하거나, Google처럼 수집된 데이터에서 "DP 합성 데이터"를 생성하여 모델을 재훈련한다.
- **옵션 3: 연합 학습(FL)**
  - **설계:** (옵션 1에 추가) 여러 기관(예: 여러 대학, 여러 은행)이 데이터를 공유하지 않고 "공통" 도메인 모델(예: 공동 신용 점수화 모델)을 구축하려 할 때 사용한다.
  - **선택 이유:** 중앙 데이터 수집이 법적으로/상업적으로 불가능할 때 사용한다. 각 기관이 자체 로컬 LoRA만 훈련하고 서버가 해당 LoRA만 집계하는 "PEFT(LoRA) + FL" 아키텍처를 사용한다. (시나리오 1에 적합할 수 있음)
- **옵션 4: 동형 암호화(HE)**
  - **설계:** 사용자가 서버(신용 점수화 모델)가 자신의 민감한 쿼리/문서(예: 개인 재무 제표)를 전혀 보지 않고 추론을 수행하기를 원할 때 사용한다.
  - **선택 이유:** 최고 수준의 "쿼리 프라이버시"가 필요할 때 사용한다. 서버가 선형 연산(암호화)을 처리하고 클라이언트가 비선형 연산(복호화)을 처리하는 "하이브리드 HE" 방법을 사용하여 실용적인 추론 속도를 달성한다. (시나리오 1의 B2C 버전에 적합)

과제에서 가장 강력하고 실용적인 해결책은 2장의 "가장 화려한" PET가 아닐 수 있다. 2025년 현재, 대부분의 규제 문제는 복잡한 암호화(HE)나 분산 학습(FL)을 요구하지 않는다. 문제는 대부분 데이터 거버넌스와 아키텍처 분리의 실패에서 비롯된다. 3장에서 본 "자체 호스팅 RAG + 비식별화 필터" 아키텍처는 규제 문제의 90%를 해결하는 "기본값"이자 "모범 사례" 청사진이다. **최고의 프라이버시 보호는 처음부터 데이터를 수집하지 않는 것(RAG)이거나 익명화하는 것(비식별화)이다.** DP, FL, HE는 이 기본 아키텍처로 해결할 수 없는 특별한 비즈니스 요구가 발생했을 때만 고려할 "2단계" 해결책이다(예: "분산 데이터로 훈련해야 한다").

---

## 참고자료

1. AI법 상위 수준 요약 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/high-level-summary/](https://artificialintelligenceact.eu/high-level-summary/)
2. EU AI법 타임라인 및 위험 단계 설명 - Trilateral Research, [https://trilateralresearch.com/responsible-ai/eu-ai-act-implementation-timeline-mapping-your-models-to-the-new-risk-tiers](https://trilateralresearch.com/responsible-ai/eu-ai-act-implementation-timeline-mapping-your-models-to-the-new-risk-tiers)
3. 핵심 이슈 3: 위험 기반 접근법 - EU AI Act, [https://www.euaiact.com/key-issue/3](https://www.euaiact.com/key-issue/3)
4. AI법 | 유럽의 디지털 미래 형성 - European Union, [https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
5. EU AI법 금지 사용 사례 | Harvard University Information Technology, [https://www.huit.harvard.edu/eu-ai-act](https://www.huit.harvard.edu/eu-ai-act)
6. 새로운 EU AI법 가이드라인: 기업에 대한 시사점은 무엇인가?, [https://www.twobirds.com/en/insights/2025/global/new-eu-ai-act-guidelines-what-are-the-implications-for-businesses](https://www.twobirds.com/en/insights/2025/global/new-eu-ai-act-guidelines-what-are-the-implications-for-businesses)
7. EU AI법: 2025년 현재 우리는 어디에 있는가? | Blog - BSR, [https://www.bsr.org/en/blog/the-eu-ai-act-where-do-we-stand-in-2025](https://www.bsr.org/en/blog/the-eu-ai-act-where-do-we-stand-in-2025)
8. AI법 탐색기 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/ai-act-explorer/](https://artificialintelligenceact.eu/ai-act-explorer/)
9. EU AI법: 빠른 가이드, [https://www.simmons-simmons.com/en/publications/clyimpowh000ouxgkw1oidakk/the-eu-ai-act-a-quick-guide](https://www.simmons-simmons.com/en/publications/clyimpowh000ouxgkw1oidakk/the-eu-ai-act-a-quick-guide)
10. EU AI법에 대해 알아야 할 사항 및 Concentric AI가 도울 수 있는 방법, [https://concentric.ai/what-you-need-to-know-about-the-eu-ai-act-and-how-concentric-ai-can-help/](https://concentric.ai/what-you-need-to-know-about-the-eu-ai-act-and-how-concentric-ai-can-help/)
11. EU AI법: AI 시스템의 다양한 위험 수준 - Forvis Mazars - Ireland, [https://www.forvismazars.com/ie/en/insights/news-opinions/eu-ai-act-different-risk-levels-of-ai-systems](https://www.forvismazars.com/ie/en/insights/news-opinions/eu-ai-act-different-risk-levels-of-ai-systems)
12. EU Artificial Intelligence Act | EU AI법의 최신 발전 및 분석, [https://artificialintelligenceact.eu/](https://artificialintelligenceact.eu/)
13. GPAI 모델 가이드라인 개요 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/gpai-guidelines-overview/](https://artificialintelligenceact.eu/gpai-guidelines-overview/)
14. 범용 AI 모델 제공자를 위한 가이드라인 | Shaping Europe's digital future, [https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers](https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers)
15. 유럽위원회, EU AI법에 따른 범용 AI 모델 의무에 대한 가이드라인 발표 | DLA Piper, [https://www.dlapiper.com/insights/publications/ai-outlook/2025/european-commission-publishes-guidelines-for-general-purpose-ai-models-under-the-eu-ai-act](https://www.dlapiper.com/insights/publications/ai-outlook/2025/european-commission-publishes-guidelines-for-general-purpose-ai-models-under-the-eu-ai-act)
16. 유럽위원회, 범용 AI 모델 제공자를 위한 가이드라인 발표, [https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20250724-european-commission-issues-guidelines-for-providers-of-general-purpose-ai-models](https://www.wilmerhale.com/en/insights/blogs/wilmerhale-privacy-and-cybersecurity-law/20250724-european-commission-issues-guidelines-for-providers-of-general-purpose-ai-models)
17. EU AI법: 인공지능에 대한 최초 규제 | Topics - European Parliament, [https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence](https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence)
18. 일반적으로 말하면: 귀하의 회사가 범용 AI 모델 제공자로서 EU AI법 준수 의무가 있는가? - Arnold & Porter, [https://www.arnoldporter.com/en/perspectives/advisories/2025/08/does-your-company-have-eu-ai-act-compliance-obligations](https://www.arnoldporter.com/en/perspectives/advisories/2025/08/does-your-company-have-eu-ai-act-compliance-obligations)
19. EU AI법 준수 체크리스트 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/](https://artificialintelligenceact.eu/assessment/eu-ai-act-compliance-checker/)
20. EU의 범용 AI 의무가 새로운 지침과 함께 발효됨 - Skadden, [https://www.skadden.com/insights/publications/2025/08/eus-general-purpose-ai-obligations](https://www.skadden.com/insights/publications/2025/08/eus-general-purpose-ai-obligations)
21. 범용 AI 실무 규약 소개 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/introduction-to-code-of-practice/](https://artificialintelligenceact.eu/introduction-to-code-of-practice/)
22. EU 위원회, 범용 AI 실무 규약 발표: 준수 의무는 2025년 8월부터 시작 - Nelson Mullins, [https://www.nelsonmullins.com/insights/blogs/ai-task-force/ai/ai-task-force-the-eu-commission-publishes-general-purpose-ai-code-of-practice-compliance-obligations-begin-august-2025](https://www.nelsonmullins.com/insights/blogs/ai-task-force/ai/ai-task-force-the-eu-commission-publishes-general-purpose-ai-code-of-practice-compliance-obligations-begin-august-2025)
23. 실무 규약 개요 | EU Artificial Intelligence Act, [https://artificialintelligenceact.eu/code-of-practice-overview/](https://artificialintelligenceact.eu/code-of-practice-overview/)
24. EU AI법에 따른 범용 AI 의무가 2025년 8월 2일부터 발효 | Insight, [https://www.bakermckenzie.com/en/insight/publications/2025/08/general-purpose-ai-obligations](https://www.bakermckenzie.com/en/insight/publications/2025/08/general-purpose-ai-obligations)
25. 범용 AI 실무 규약 | Shaping Europe's digital future, [https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai](https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai)
26. EU AI법: 범용 AI 실무 규약 · 최종 버전, [https://code-of-practice.ai/](https://code-of-practice.ai/)
27. EU AI법 하에서 AI 수정: 실무로부터의 교훈, [https://artificialintelligenceact.eu/modifying-ai-under-the-eu-ai-act/](https://artificialintelligenceact.eu/modifying-ai-under-the-eu-ai-act/)
28. 전문: 유럽에서 AI 규제: AI법과 AI 프레임워크 협약의 공동 분석 - Taylor & Francis Online, [https://www.tandfonline.com/doi/full/10.1080/20508840.2025.2492524](https://www.tandfonline.com/doi/full/10.1080/20508840.2025.2492524)
29. AI 사용 규제에 대한 EU, 미국, 중국에 대한 신뢰 - Pew Research Center, [https://www.pewresearch.org/2025/10/15/trust-in-the-eu-u-s-and-china-to-regulate-use-of-ai/](https://www.pewresearch.org/2025/10/15/trust-in-the-eu-u-s-and-china-to-regulate-use-of-ai/)
30. AI Watch: 글로벌 규제 추적기 - 미국 | White & Case LLP, [https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-united-states](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-united-states)
31. 미국의 AI 행동 계획 - The White House, [https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf](https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf)
32. 트럼프 행정부 하에서의 2025년 2월 AI 발전, [https://www.insidegovernmentcontracts.com/2025/03/february-2025-ai-developments-under-the-trump-administration/](https://www.insidegovernmentcontracts.com/2025/03/february-2025-ai-developments-under-the-trump-administration/)
33. AI 위험 관리 프레임워크 | NIST - National Institute of Standards and Technology, [https://www.nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
34. NIST AI 위험 관리 프레임워크: 더 스마트한 AI 거버넌스를 위한 간단한 가이드 - Diligent, [https://www.diligent.com/resources/blog/nist-ai-risk-management-framework](https://www.diligent.com/resources/blog/nist-ai-risk-management-framework)
35. 유럽 산업계, EU AI법에 반발 – 고용주를 위한 핵심 요약, [https://www.fisherphillips.com/en/news-insights/european-industry-pushes-back-on-the-eu-ai-act.html](https://www.fisherphillips.com/en/news-insights/european-industry-pushes-back-on-the-eu-ai-act.html)
36. 트럼프와 빅테크의 압력 속에서 EU가 AI법을 완화할 수 있다, [https://www.theguardian.com/world/2025/nov/07/european-commission-ai-artificial-intelligence-act-trump-administration-tech-business](https://www.theguardian.com/world/2025/nov/07/european-commission-ai-artificial-intelligence-act-trump-administration-tech-business)
37. 한국의 새로운 AI 법률: 조직에 대한 의미 및 준비 방법, [https://www.onetrust.com/blog/south-koreas-new-ai-law-what-it-means-for-organizations-and-how-to-prepare/](https://www.onetrust.com/blog/south-koreas-new-ai-law-what-it-means-for-organizations-and-how-to-prepare/)
38. 한국 인공지능(AI) 기본법 - International Trade Administration, [https://www.trade.gov/market-intelligence/south-korea-artificial-intelligence-ai-basic-act](https://www.trade.gov/market-intelligence/south-korea-artificial-intelligence-ai-basic-act)
39. 한국의 새로운 AI 프레임워크법: 혁신과 규제 사이의 균형, [https://fpf.org/blog/south-koreas-new-ai-framework-act-a-balancing-act-between-innovation-and-regulation/](https://fpf.org/blog/south-koreas-new-ai-framework-act-a-balancing-act-between-innovation-and-regulation/)
40. 인공지능 규제에 대한 글로벌 접근법, [https://jsis.washington.edu/news/global-approaches-to-artificial-intelligence-regulation/](https://jsis.washington.edu/news/global-approaches-to-artificial-intelligence-regulation/)
41. AI Watch: 글로벌 규제 추적기 - 중국 | White & Case LLP, [https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-china](https://www.whitecase.com/insight-our-thinking/ai-watch-global-regulatory-tracker-china)
42. 중국 - AI 규제 지평선 추적기 - Bird & Bird, [https://www.twobirds.com/en/capabilities/artificial-intelligence/ai-legal-services/ai-regulatory-horizon-tracker/china](https://www.twobirds.com/en/capabilities/artificial-intelligence/ai-legal-services/ai-regulatory-horizon-tracker/china)
43. 2025년 AI 규제: 미국, EU, 영국, 일본, 중국 등, [https://www.anecdotes.ai/learn/ai-regulations-in-2025-us-eu-uk-japan-china-and-more#:~:text=China%3A%20Generative%20AI%20Regulation,-In%20brief%3A%20What&text=In%20addition%2C%20providers%20are%20required,legal%20sourcing%20of%20training%20data.](https://www.anecdotes.ai/learn/ai-regulations-in-2025-us-eu-uk-japan-china-and-more#:~:text=China%3A%20Generative%20AI%20Regulation,-In%20brief%3A%20What&text=In%20addition%2C%20providers%20are%20required,legal%20sourcing%20of%20training%20data.)
44. AI 규제 업데이트 2025년 하반기 - FairNow, [https://fairnow.ai/ai-regulations-updates-h2-2025/](https://fairnow.ai/ai-regulations-updates-h2-2025/)
45. 중국, 글로벌 AI 거버넌스를 위한 행동 계획 발표, [https://www.ansi.org/standards-news/all-news/8-1-25-china-announces-action-plan-for-global-ai-governance](https://www.ansi.org/standards-news/all-news/8-1-25-china-announces-action-plan-for-global-ai-governance)
46. [2506.11687] 기계 학습에서의 차등 프라이버시: 기호 AI에서 LLM까지 - arXiv, [https://arxiv.org/abs/2506.11687](https://arxiv.org/abs/2506.11687)
47. 기계 학습에서의 차등 프라이버시: 기호 AI에서 LLM까지 - arXiv, [https://arxiv.org/html/2506.11687v1](https://arxiv.org/html/2506.11687v1)
48. [2503.12896] 엔트로피 기반 섭동을 통한 엔드-클라우드 협업에서 LLM 임베딩 보호 - arXiv, [https://arxiv.org/abs/2503.12896](https://arxiv.org/abs/2503.12896)
49. 차등 프라이버시 LLM 추론을 사용한 합성 데이터 생성, [https://research.google/blog/generating-synthetic-data-with-differentially-private-llm-inference/](https://research.google/blog/generating-synthetic-data-with-differentially-private-llm-inference/)
50. 혁신과 프라이버시의 교차로: 생성형 AI를 위한 프라이빗 합성 데이터, [https://www.microsoft.com/en-us/research/blog/the-crossroads-of-innovation-and-privacy-private-synthetic-data-for-generative-ai/](https://www.microsoft.com/en-us/research/blog/the-crossroads-of-innovation-and-privacy-private-synthetic-data-for-generative-ai/)
51. Apple Intelligence 기초 언어 모델 기술 보고서 2025, [https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
52. 차등 프라이버시를 사용한 Apple Intelligence 집계 트렌드 이해, [https://machinelearning.apple.com/research/differential-privacy-aggregate-trends](https://machinelearning.apple.com/research/differential-privacy-aggregate-trends)
53. 종단 간 음성 인식을 위한 차등 프라이버시 연합 학습, [https://machinelearning.apple.com/research/fed-learning-diff-privacy](https://machinelearning.apple.com/research/fed-learning-diff-privacy)
54. TechDispatch #1/2025 - 연합 학습 - European Data Protection Supervisor, [https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-06-10-techdispatch-12025-federated-learning_en](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-06-10-techdispatch-12025-federated-learning_en)
55. 연합 학습으로 의료 데이터 분석 혁신: 애플리케이션, 시스템 및 미래 방향에 대한 포괄적 조사 - PMC - PubMed Central, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/)
56. 레이어 스킵을 사용한 연합 학습: 효율적인 훈련 - arXiv, [https://arxiv.org/abs/2504.10536](https://arxiv.org/abs/2504.10536)
57. [2501.04436] LLM의 연합 파인튜닝: 프레임워크 비교 및 연구 방향 - arXiv, [https://arxiv.org/abs/2501.04436](https://arxiv.org/abs/2501.04436)
58. (PDF) 연합 대규모 언어 모델: 솔루션, 도전 과제 및 미래 방향, [https://www.researchgate.net/publication/385183939_Federated_Large_Language_Model_Solutions_Challenges_and_Future_Directions](https://www.researchgate.net/publication/385183939_Federated_Large_Language_Model_Solutions_Challenges_and_Future_Directions)
59. FedSRD: 통신 효율적 연합 대규모 언어 모델 파인튜닝을 위한 희소화-재구성-분해 - arXiv, [https://arxiv.org/html/2510.04601v2](https://arxiv.org/html/2510.04601v2)
60. 연합 학습 구현: 프라이버시 보존 AI 접근법, [https://blog.4geeks.io/implementing-federated-learning-a-privacy-preserving-ai-approach/](https://blog.4geeks.io/implementing-federated-learning-a-privacy-preserving-ai-approach/)
61. 암호화 친화적 LLM 아키텍처 - arXiv, [https://arxiv.org/pdf/2410.02486](https://arxiv.org/pdf/2410.02486)
62. HHEML: 엣지에서 프라이버시 보존 기계 학습을 위한 하이브리드 동형 암호화, [https://arxiv.org/html/2510.20243v1](https://arxiv.org/html/2510.20243v1)
63. 에이전트 기반 프라이버시 보존 기계 학습 위치 논문. 활발히 개발 중., [https://arxiv.org/html/2508.02836](https://arxiv.org/html/2508.02836)
64. 완전한 하이브리드 ML 추론의 실용적이고 프라이빗한 방법 - arXiv, [https://arxiv.org/abs/2509.01253](https://arxiv.org/abs/2509.01253)
65. 동형 암호화를 사용한 신경망을 위한 효율적인 키셋 설계 - MDPI, [https://www.mdpi.com/1424-8220/25/14/4320](https://www.mdpi.com/1424-8220/25/14/4320)
66. 동형 암호화를 사용한 프라이버시 보존 딥러닝 모델 개발: 신장 CT 영상에서의 기술적 타당성 연구 | Radiology: Artificial Intelligence - RSNA Journals, [https://pubs.rsna.org/doi/10.1148/ryai.240798](https://pubs.rsna.org/doi/10.1148/ryai.240798)
67. 암호화 친화적 LLM 아키텍처 - ICLR Proceedings, [https://proceedings.iclr.cc/paper_files/paper/2025/file/6715b4e97be055687c1ecaf33913d358-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/6715b4e97be055687c1ecaf33913d358-Paper-Conference.pdf)
68. [2410.02486] 암호화 친화적 LLM 아키텍처 - arXiv, [https://arxiv.org/abs/2410.02486](https://arxiv.org/abs/2410.02486)
69. API 기반 LLM으로 전송하기 전에 클라이언트 데이터를 암호화하는 방법? : r/LlamaIndex - Reddit, [https://www.reddit.com/r/LlamaIndex/comments/1iwzeph/how_to_encrypt_client_data_before_sending_to_an/](https://www.reddit.com/r/LlamaIndex/comments/1iwzeph/how_to_encrypt_client_data_before_sending_to_an/)
70. 임상 기록에서 보호 건강 정보의 기업 규모 비식별화를 위한 자연어 처리 - NIH, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9285160/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9285160/)
71. 2025년 의료 챗봇 개발 가이드 - MobiDev, [https://mobidev.biz/blog/healthcare-chatbot-development-guide](https://mobidev.biz/blog/healthcare-chatbot-development-guide)
72. HIPAA 준비 AI 챗봇: 의료 혁신을 위한 안전한 호스팅, [https://www.atlantic.net/hipaa-compliant-hosting/hipaa-ready-ai-chatbots-secure-hosting-for-healthcare-innovation/](https://www.atlantic.net/hipaa-compliant-hosting/hipaa-ready-ai-chatbots-secure-hosting-for-healthcare-innovation/)
73. ChatGPT는 HIPAA를 준수하는가? 2025년 업데이트 - The HIPAA Journal, [https://www.hipaajournal.com/is-chatgpt-hipaa-compliant/](https://www.hipaajournal.com/is-chatgpt-hipaa-compliant/)
74. 의료용 ChatGPT | HIPAA 준수 의료 GPT, [https://bastiongpt.com/](https://bastiongpt.com/)
75. 의료 챗봇을 위한 HIPAA 준수: 필수 가이드 - Kommunicate, [https://www.kommunicate.io/blog/a-essential-guide-to-hipaa-compliance-in-healthcare-chatbots/](https://www.kommunicate.io/blog/a-essential-guide-to-hipaa-compliance-in-healthcare-chatbots/)
76. 손쉬운 PHI 비식별화: 비식별화된 환자 데이터, [https://www.johnsnowlabs.com/effortless-de-identification-running-obfuscation-and-deidentification-in-healthcare-nlp/](https://www.johnsnowlabs.com/effortless-de-identification-running-obfuscation-and-deidentification-in-healthcare-nlp/)
77. 영어 및 독일어 전자 건강 기록 비식별화를 위한 대규모 언어 모델 - MDPI, [https://www.mdpi.com/2078-2489/16/2/112](https://www.mdpi.com/2078-2489/16/2/112)
78. PHI 식별 소프트웨어: 완전한 2025년 가이드 및 도구 - Invene, [https://www.invene.com/blog/software-to-identify-phi-complete-guide](https://www.invene.com/blog/software-to-identify-phi-complete-guide)
79. 제로샷 상용 API가 규제 등급 임상 텍스트 비식별화를 제공할 수 있는가?, [https://arxiv.org/html/2503.20794v2](https://arxiv.org/html/2503.20794v2)
80. HIPAA 지침 내 AI 애플리케이션 사례 연구 - Accountable HQ, [https://www.accountablehq.com/post/case-studies-of-ai-applications-within-hipaa-guidelines](https://www.accountablehq.com/post/case-studies-of-ai-applications-within-hipaa-guidelines)
81. 금융 서비스에서 AI 사기 탐지 준수: 보안과 고객 권리의 균형 - VerityAI, [https://verityai.co/blog/ai-fraud-detection-compliance-financial-services](https://verityai.co/blog/ai-fraud-detection-compliance-financial-services)
82. 프라이버시와 책임 있는 AI - IAPP, [https://iapp.org/news/a/privacy-and-responsible-ai](https://iapp.org/news/a/privacy-and-responsible-ai)
83. AI 보안 및 데이터 보호의 법률 및 준수, [https://www.edpb.europa.eu/system/files/2025-06/spe-training-on-ai-and-data-protection-legal_en.pdf](https://www.edpb.europa.eu/system/files/2025-06/spe-training-on-ai-and-data-protection-legal_en.pdf)
84. AI 개발자를 위한 GDPR 준수 - 실용 가이드 - Essert Inc, [https://essert.io/gdpr-compliance-for-ai-developers-a-practical-guide/](https://essert.io/gdpr-compliance-for-ai-developers-a-practical-guide/)
85. EU AI법과 GDPR: 충돌인가 정렬인가? - Taylor Wessing, [https://www.taylorwessing.com/en/global-data-hub/2025/eu-digital-laws-and-gdpr/gdh---the-eu-ai-act-and-the-gdpr](https://www.taylorwessing.com/en/global-data-hub/2025/eu-digital-laws-and-gdpr/gdh---the-eu-ai-act-and-the-gdpr)
86. (PDF) 신용 점수화에서의 설명 가능한 AI: 정확성과 투명성의 균형, [https://www.researchgate.net/publication/394998451_Explainable_AI_in_Credit_Scoring_Balancing_Accuracy_and_Transparency](https://www.researchgate.net/publication/394998451_Explainable_AI_in_Credit_Scoring_Balancing_Accuracy_and_Transparency)
87. GDPR 및 금융 규제 하에서 AI 기반 사기 탐지 - ResearchGate, [https://www.researchgate.net/publication/393870899_AI-Driven_Fraud_Detection_Under_GDPR_and_Financial_Regulations](https://www.researchgate.net/publication/393870899_AI-Driven_Fraud_Detection_Under_GDPR_and_Financial_Regulations)
88. 신용 점수화 및 대출 승인을 위한 설명 가능한 AI (XAI) - ResearchGate, [https://www.researchgate.net/publication/389847187_Explainable_AI_XAI_for_Credit_Scoring_and_Loan_Approvals](https://www.researchgate.net/publication/389847187_Explainable_AI_XAI_for_Credit_Scoring_and_Loan_Approvals)
89. Advance Journal of Econometrics and Finance Vol-3, Issue-1, 2025, [https://ajeaf.com/index.php/Journal/article/download/131/142](https://ajeaf.com/index.php/Journal/article/download/131/142)
90. TechDispatch #2/2023 - 설명 가능한 인공지능, [https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2023-11-16-techdispatch-22023-explainable-artificial-intelligence_en](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2023-11-16-techdispatch-22023-explainable-artificial-intelligence_en)
91. 금융에서의 설명 가능한 AI: 다양한 이해관계자의 요구 해결, [https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance](https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance)
92. 은행의 신용 평가를 위한 설명 가능한 AI - MDPI, [https://www.mdpi.com/1911-8074/15/12/556](https://www.mdpi.com/1911-8074/15/12/556)
93. 신용 점수화에서 투명성 향상을 위한 새로운 프레임워크: 해석 가능한 신용 점수카드를 위한 Shapley 값 활용 - NIH, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11318906/)
94. (PDF) 설명 가능한 인공지능(XAI) 기법(예: SHAP, LIME)을 복잡한 블랙박스 모델에 적용한 신용 위험 평가에서 모델 해석 가능성 및 규제 준수 향상 - ResearchGate, [https://www.researchgate.net/publication/397323127_Enhancing_Model_Interpretability_and_Regulatory_Compliance_in_Credit_Risk_Assessment_through_Explainable_Artificial_Intelligence_XAI_Techniques_eg_SHAP_LIME_applied_to_complex_Black-Box_models](https://www.researchgate.net/publication/397323127_Enhancing_Model_Interpretability_and_Regulatory_Compliance_in_Credit_Risk_Assessment_through_Explainable_Artificial_Intelligence_XAI_Techniques_eg_SHAP_LIME_applied_to_complex_Black-Box_models)
95. SHAP을 사용한 신용 점수화 딥러닝 모델 설명: 오픈 뱅킹 데이터를 사용한 사례 연구 - MDPI, [https://www.mdpi.com/1911-8074/16/4/221](https://www.mdpi.com/1911-8074/16/4/221)
96. 설명 가능한 AI를 사용하여 ECOA 불리한 조치 이유 생성, [https://www.paceanalyticsllc.com/post/ecoa-adverse-actions-and-explainable-ai](https://www.paceanalyticsllc.com/post/ecoa-adverse-actions-and-explainable-ai)
97. 정확성-해석 가능성 딜레마: 현대 기계 학습에서 트레이드오프를 탐색하기 위한 전략적 프레임워크 - Science Publishing Group, [https://www.sciencepublishinggroup.com/article/10.11648/j.ajist.20250903.15](https://www.sciencepublishinggroup.com/article/10.11648/j.ajist.20250903.15)
98. FERPA 준수를 위한 2025 AI 가이드 | Concentric AI, [https://concentric.ai/maintain-ferpa-compliance-with-concentric-ai/](https://concentric.ai/maintain-ferpa-compliance-with-concentric-ai/)
99. 인공지능과 관련된 연방 규제 미국에는 데이터 프라이버시를 포괄하는 포괄적 법률이 없다, [https://www.nea.org/sites/default/files/2025-06/5.1-ai-policy-overview-of-federal-regulations-final.pdf](https://www.nea.org/sites/default/files/2025-06/5.1-ai-policy-overview-of-federal-regulations-final.pdf)
100. AI 사례 연구: 직원과 학생이 공정하고 프라이버시 보호 방식으로 AI를 사용하도록 돕기 - MOREnet, [https://www.more.net/wp-content/uploads/2025/02/Case-Studies-in-AI.pdf](https://www.more.net/wp-content/uploads/2025/02/Case-Studies-in-AI.pdf)
101. FERPA 준수 AI 학습 동반자 구축 방법 - Ki Ecke, [https://ki-ecke.com/insights/how-to-build-a-ferpa-compliant-ai-study-companion/](https://ki-ecke.com/insights/how-to-build-a-ferpa-compliant-ai-study-companion/)
