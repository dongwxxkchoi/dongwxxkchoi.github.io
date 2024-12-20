---
layout: single
date: 2023-08-13
title: "CS224n - Lecture 15 (Add Knowledge to Language Models)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

> ✅ **Add Knowledge to Language Models  
> - Language Models (LMs)  
> - What does a LM know?  
> - Techniques to add knowledge to LMs  
>     1) Add pretrained entity embeddings  
>     2) Use an external memory  
>     3) Modify the training data  
> - Evaluating knowledge in LMs**



## Language Models

- 현재 대부분의 **language models**는 **2가지 방식으로 sequence의 probability를 학습**한다.
	1. **sequence of text의 다음 단어를 예측**

		![0](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/0.png)

	2. **masked token을 bidirectional context를 사용해 예측 (ex. BERT)**

		![1](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/1.png)


	⇒ 두 방법 모두 human annotated data 즉, labeled data가 필요 없음


	**이렇게 학습된 LMs은 다양한 task에 사용됨**

	- Summarization
	- Dialogue
	- Autocompletion
	- Machine Translation
	- Fluency evaluation
	- …
- 흔히 **text의 pretrained representations을 생성**하는데 사용

	(downstream NLP tasks를 위한, language understanding의 개념들을 encoding하는)


	⇒ 그런데, 만약 **language model이 knowledge base가 될 수 있을까?**

- “Petroni et al., EMNLP 2019” 에서 BERT-Large를 test

![2](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/2.png)


![3](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/3.png)


![4](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/4.png)


초록색 답이 정답, 빨간색 답이 오답 


⇒ 이처럼, **make sense(correct type) 즉 그럴듯 하지만, 항상 정답은 아님**

- **왜 이런 일이 일어날까?**
	- **Unseen facts** : **train 과정에서 본 적이 없을** 수 있음. LM은 **fact를 생성해낼 수 없음.**
	- **Rare facts** : 그 fact가 **rare**할 수 있음. train 과정에서 봤을 수 있으나, **기억 할만큼 충분한 example을 본 것이 아님**
	- **Model sensitivity** : LM은 **prompt에 민감**함

		예를 들어, 

		- “x was <u>made</u> in y”에선 X
		- “x was <u>created</u> in y”에선 O

		x와 y관계가 underline의 동사 즉, prompt에 따라 바뀔 수도 있음 


	이처럼 확실하게 기억을 하지 못하는 것은 LM의 가장 큰 문제점


	⇒ 최근 **LM 연구에서 가장 핫한 문제**

- 많은 downstream tasks에서 사용 knowledge

	→ **LM의 pretrained representations 기반**임


	⇒ knowledge-intensive한 tasks도 존재


	downstream task → extract the relations between two entities in a sentence


	**⇒ relation extraction**


만약 몇몇 entity간 relation을 이미 알고 있다면 더 쉬운 task


(ex. pretrained language model에서 받은)

- evaluation
- what types of task
- rich pretrained representations

⇒ stretch goal, some researchers 

- 과연 **LM**이 궁극적으로 **traditional knowledge bases를 대체**할 수 있을까?

	즉, SQL같은 지식 기반 quering 대신 **자연어 prompt로 된 LM에 query**


	⇒ 이를 위해선 LM이 **높은 수준의 recalling facts 능력**을 가져야 함 

- **traditional knowledge bases**

	![5](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/5.png)

	- node : entity

		edge : entity 간의 관계

	- 특징
		- populated by human
		- fill in relation, entity
		- structured data LM에 반영 위한 pipeline 필요

			ex. query

	- apparent entity, relation, tail entity 라고도 부름
- **Querying language models as knowlede basis**

	![6](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/6.png)

	- 특징
		- 방대한 양의 **unstructured text data**에 대해 학습
		- **natural language query를 통해 질의** 가능
	- **장점**

		flexible natural language queries  (example → U.F.O.F.의 마지막 단어?)

	- **단점**

		hard to interpret, trust, modify

		- interpret

			기존 방식은 information의 기원이 존재


			→ 왜 그 query return?


			but, LM은 명확치 x (그저 knowledge가 encoded 된 것이기 때문)

		- trust

			사실적인 prediction but incorrect 가능


			LM이 fact를 아는 것인지, bias를 통해 예측을 하는것인지?

		- modify

			기존 방식은 그저 update 가능


			LM은 분명치 않음


			→ fine-tune but 오래된 기억이 남아있는지 어떻게 알 수 있나?


![7](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/7.png)


⇒ 그래서 이를 위한 기법들이 소개되고 있음



## Techniques to add knowledge to LMs


![8](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/8.png)



### Adding pretrained entity embeddings


![9](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/9.png)

- pretrained word는 그런 **entity에 대한 notion이 없음**

	**LM**이 이런 entities의 representation **배우는 것은 어려움** 


	(ex. U.S.A = United States of America = America)


	**⇒ entity당 embedding 부여**


	→ **entity embeddings**는 이런 **world에 대한 factual knowledge를 encode하도록 pretrained**

- **Entity linking?**

	![10](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/10.png)

	- word embedding

		→ word 당 embedding 값이 ID처럼 부여됨

	- entity embedding

		→ **word 당 entity가 여러 개** 있을 수 있음


		**→ entity 당 embedding**


		(위의 Washington은 Washington State와는 다른 embedding 값 가짐)


		⇒ 이 ambiguous한 **mention 관계를 찾아내는 것**이 entity embedding의 목표

- **Entity Embedding**
	- knowledge graph embedding models (ex. TransE)
		- pretrained **entity** / pretrained **relation** embeddings

			⇒ subject embedding + relation embedding = object embedding 이도록

	- Word-entity co-occurence methods (ex. Wikipedia2Vec)
		- entity 있을 때, 어떤 words에 대해 동시-발생 많은지?
	- Transformer encodings of entity descriptions (ex. BLINK)
		- use transformer

	⇒ 그렇다면, 이렇게 **pretrained entity embeddings을 어떻게 LM에 추가할까?**

- **Add pretrained entity embeddings**

	![11](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/11.png)

	- **특정 word** $w_j$**와 특정 entity** $e_k$ **사이에 alignment가 있는 걸 안다**고 가정

		⇒ $e_k = f(w_j)$

	- $w_j$ : sequence of words의 j번째 word embedding
	- $e_k$ : 상응하는 entity embedding

	이 둘을 연결하는 방법이 필요함


	**⇒ fusion layer**


	![12](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/12.png)

	- $W_t$ 와 $W_e$ : weight matrix

		⇒ 같은 차원으로 $w_j$와 $e_k$를 mapping

	- $F$ : activation function
	- $h_j$ : 두 input이 합쳐져 하나의 hidden representation output

	그리고 그 방법 관련 model


	**⇒ ERNIE**



#### ERNIE (Enhanced Language Representation with Informative Entities)


![13](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/13.png)

- **pretrained entity embeddings 제공**
- **구조**
	- **Text-Encoder**
		- 좌하단 부분
		- Token input (**Word input**)을 받음
		- multi-layer bidirectional **Transformer encoder로 구성**
	- **Knowledge-Encoder**
		- 좌상단 부분
		- Text-Encoder를 통과한 **Token**과 **Entity input을 받음**
		- multi-headed attentions로 구성
			- 두 input 받음
		- **fusion layer 존재**

			$w_l^{(i-1)}$ → Multi-head Attention → $\tilde w_l^{(i)}$  ($l : 0~...~n$)


			$e_o^{(i-1)}$ → Multi-head Attention → $\tilde e_o^{(i)}$   ($o:0~...~m$)


			⇒ 이렇게 이전 timestep의 input들이 attention을 거쳐


			![14](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/14.png)


			⇒ weight matrix들과 product되어 single hidden state output


			![15](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/15.png)


			⇒ 이 hidden state를 통해 사용된 $w_j$와 $e_k$를 update

	- **hidden state** $h_j$

		word input인 $w_j$와 같은 index를 갖는 이유?

- **Pre-Train**

	(BERT와 유사)

	- masked language model
	- next sentence prediction

	(추가)

	- **randomly mask token-entity alignments**

		![16](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/16.png)


		⇒ **word의 entity를 masking 해놓고, entity를 추측**

	- 최종 Loss

		![17](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/17.png)

- **성과**

	![18](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/18.png)

- **강점**
	- entity와 context info의 combine
	- knowledge-driven downstream tasks에서 좋은 성능
- **단점**
	- fully annotated entities가 필요

		(even for downstream tasks)


		⇒ pre-linked input entity가 필요하다.

	- 그렇기 때문에 더 expensive


#### KnowBERT

- **BERT**에서 downstream task로 **Wikipedia 등의 다양한 지식 베이스에서 추출한 정보를 추가적으로 활용**해 **entity와 관련된 지식을 encoding**함

	→ 이후 **entity를 예측하는 entity linker (EL)을 추가**해서 pretraining 하는 방법 


	**⇒ EL을 학습시키기 떄문에, entity annotation이 필요 없음**



### Use an external memory


![19](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/19.png)

- **pretrained entity embeddings의 단점**

	knowledge base를 update 할 때


	⇒ retrain entity embeddings 필요

	- 더 direct한 방법이 없을까?

		**⇒ use external memory**

- **External memory (Key-value store)**
	- model이 직접 knowledge graph tripes 또는 context information에 접근하도록
	- 장점
		- independent of the learned model parameters

			⇒ better support injecting and updating factual knowledge


			⇒ update를 위해 key에 대한 value를 바꾼다거나 하면 됨

			- often without more pretraining!
		- more interpretable


#### KGLM (Barack's Wife Hillary: Using Knowledge-Graphs for Fact-Aware
Language Modeling)


![20](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/20.png)

- **특징**
	- **knowledge graph**

		![21](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/21.png)

		- sequence 속 words와 relevant entities가 주어졌을 때,

			→ **graph에 기반**해 **다음 word와 entity를 predict**

		- full knowledge graph가 있을 때, **관심 있는 word에 관한 local knowledge graph 생성**

			⇒ **가능한 entity를 찾으면 local knowledge graph에 추가**

		- entities는 **training 도중 known 된다**고 가정한다

			→ training을 위한 **entity annotated data가 필요**함 


			⇒ 그렇기 때문에, **ground truth local knowledge graph라 할 수 있음**

- **언제 knowledge graph를 사용**해야 할까?

	![22](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/22.png)

	- Sequence의 words들이 **LSTM cell**을 통과
	- **최종 target 전에 왔을 때 output**으로 **다음 단어**가

		1) Related entity (in the local KG) 


		2) New entity (not in the local KG) 


		3) Not an entity  인지 **여부를 분류**함


	⇒ 이 여부를 통해 knowledge graph 사용할지를 결정

- **어떻게 LM이 local KG에 있는지 아닌지를 판단?**

	![23](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/23.png)

	- **KG triple = (parent entity, relation, tail entity)**

		⇒ 여기서, **top-scoring parent, top-scoring relation**을 찾는 것


			다음 단어 예측에 **가장 useful한 triple이 무엇인지** 예측

		- **top-scoring parent**

			이 경우 parent entity는 하나로 고정 


			→ Super Mario Land

		- **top-scoring relation**

			→ Publisher


		⇒ 이를 통해 최종적으로 **Nintendo라고 판단**

	- $P(p_t) $ **= softmax**$(v_p \cdot h_t)$
		- $p_t$ : parent entity
		- $v_p$ : corresponding entity embedding
		- $h_t$ : LSTM의 hidden state

		⇒ 이렇게, parent entity의 distribution을 구할 수 있음

	- **이를 relation embedding에 대해서도 똑같이 적용!**

		**⇒ 최종 tail entity 판단**

	- **Next word 판단**

		⇒ **한 entity에 여러 개의 word 가능**


		⇒ 그 aliases 중에서 **가장 likely한 것**을 가져옴 


	![24](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/24.png)

- **다른 경우?**
	- **New Entity의 경우는?**

		![25](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/25.png)


		→ full KG에서 가장 top-scoring entity를 찾음

		- Next entity

			가장 score 높은 entity

		- Next word → 동일한 방식
	- **Entity가 아니라면?**

		![26](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/26.png)

		- **Next entity**

			없음

		- **Next word**

			가장 가능성 높은 vocabulary (일반적 방식)

- **특징**
	- GPT-2와 AWD-LSTM 보다 **fact completion task**에서 좋은 성능

		특히, GPT-2 보다 **더 specific 하게 token을 예측**할 수 있음

	- **fact 수정 / update 에 좋음!**

		**direct한 변화** 줄 수 있음


		![27](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/27.png)



#### KNN-LM


![28](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/28.png)

- next word를 예측하는 것 보다 **text sequence 사이의 similarity를 배우는 것**이 **쉽다!**

	ex. “Dickens is the author of _________”  $\approx$  “Dickens wrote _________”

- **rare facts와 long-tail sequence등**에 **더 좋은 성능**을 보이는 것을 확인!
- **text sequences를 nearest neighbor datastore에 store!**

	![29](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/29.png)

	- $\lambda$ : hyperparameter

		KNN, LM probabilities를 얼마나 반영할지 결정

- language generation에서도 배웠듯이, 새로운 domain에 대한 data를 **datastore**에 많이 생성해 놓으면, **새로운 domain에 대한 train 과정 없이 domain 정보를 제공**할 수 있음


### Modifying the training data


![30](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/30.png)

- 위에서의 방식은 **explicitly**한 방식이었음

	**→ unstructured text를 통해 implicit하게 incorporate하는 방식은 없을까?**


	**⇒ Yes. Mask** or **corrupt** the data ⇒ additional training tasks that require factual knowledge

- **장점**
	- **추가적인 memory나 computation이 필요 없음**
	- **architecture를 수정할 필요 없음**


#### WKLM (Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model)


![31](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/31.png)

- model이 **knowledge가 true인지 false인지 구별**하도록 학습
- **같은 type**의 **다른 entity**로 mention을 교체

	⇒ **negative knowledge stmts**를 생성

	- 모델은 entity가 **replaced 된건지,** **corrupted 된건지**를 예측
	- **type constraint**

		→ **문법적으로는 문제가 없는 negative sample**을 생성


		→ 모델이 **실질적인 knowledge를 배우도록 함**


		(단지 문법이 틀려서 틀렸다고 판단하는 것이 아닌)


	![32](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/32.png)

- **그림 설명**
	- 좌측 : 원래 article
	- 우측 : 수정된 article
	- 파란 글씨 : entities
	- 빨간 글씨 : 수정된 entities
- **과정**

	1) entities의 type check


	2) 그 type의 다른 entities를 check


	3) entities를 randomly sample


	4) alias를 하나 골라 replace

- **entity replacement loss 이용**

	![33](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/33.png)_-가 붙어야 하지 않나?_

	- e: entity, C: context, $\epsilon^+$ : true entity mention
	- True mention에 대해선 확률 증대
	- False mention에 대해선 확률 감

	**Total Loss**


	![34](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/34.png)

	- MLM loss는 token-level
	- entRep loss는 entity-level

		에서 정의됨


		(multi-word entities, phrases …)

- 이처럼 **entity level에서 modify를 진행**하는 것은 **LM의 지식의 양을 늘려줄 수 있다.**
- **성과**

	![35](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/35.png)

	- BERT, GPT-2 보다 성능 좋음 (fact completion tasks)
	- ERNIE 보다 성능 좋음 (downstream task - entity typing)
	- Ablation experiments (특정 요소 없애서 영향 확인)
		- MLM loss가 downstream task performance에 필수
		- WKLM은 MLM loss만으로 좋은 성능
- **masking**

	masking을 통해 inductive biases를 배울 수 있을까?


	* inductive biases → 귀납 편향


	즉, 학습 시 만나보지 않았던 상황에 대해 예측하기 위해 사용하는 추가적 가정


	**⇒ ERNIE**



#### ERNIE (Enhanced Representation through Knowledge Integration)


![36](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/36.png)

- **phrase-level, entity-level masking 기법**으로 downstream Chinese NLP tasks에서 좋은 성능


#### salient span masking


![37](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/37.png)

- closed domain q&a tasks에서 좋은 성능

![38](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/38.png)



## Evaluating knowledge in LMs



#### LAMA


![39](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/39.png)

- LM에 얼마나 많은 knowledge가 이미 encode 되었는지? (commonsense / factual)

	⇒ **pre-trained lm의 knowledge를 평가할 수 있는 일종의 도구**

- 추가적인 **training이나 fine-tuning이 필요 x**
- 빈칸이 존재하는 문장 set 준비

	→ missing token을 예측하는 model의 성능을 평가

- KG triples와 question-answer pairs로부터 빈칸이 있는 문장을 생성
- LM을 supervised relation extraction (knowledge triple 추출)과 QA systems을 비교하는 방식으로 평가 수행

![40](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/40.png)

- **P@1**

	만약 MASK 토큰을 정확하게 예측했다 → 1 , 틀렸다 → 0

- 즉 값이 높다?

	→ 생성된 cloze stmts에서 정확하게 예측한 경우의 평균이 높았다! 


	→ 정확하게 답변할 수 있는 확률이 높다


![41](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/41.png)_github에서 사용 가능 ; The cat is on the [MASK]_

- **한계**
	- 왜 model이 그런 결과가 나왔는지 알기 힘들다
		- BERT-large는 이해보다는 co-occurence patterns을 외워서 높을 수도 있음
		- LM은 그저 surface form (ex. <u>Pope Clement</u> VII has the position of <u>pope</u>) 사이의 similarities를 인식할 수도 있음
	- 문장의 phrasing에 sensitive하다
		- 각 relation에 대해 하나의 manually defined template 밖에 없음
		- probe results가 lower bound라는 것을 뜻함


#### LAMA-UnHelpful Names

- **relational knowledge 없이 대답될 수 있는 examples**를 **지우는 것**
- BERT가 이해가 아닌 surface form에 의존하는 것 확인

	ex. String match between subject / object 


	ex. Revealing person name 

		- 이름이 incorrect prior 일수도 있음

			![42](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/42.png)

- string match setting and revealing person name setting 없앰

	**⇒ harder subset**

- 이를 통해 BERT는 8% 하락, 지식 기반 모델 E-BERT는 1%만 하락


#### Developing better prompts to query knowledge in LMs

- LM이 fact를 알지만, LAMA같은 completion task에 실패

	→ because of query itself 


	![43](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/43.png)


	⇒ 구조가 상당히 다르기 때문에, 모델이 답변하기 힘들 수 있음

- 많은 prompts를 생성 by mining templates from Wikipedia

	( dependency parsing / generate paraphrase prompts )


	⇒ prompt의 변화가 엄청난 향상을 이끌 수 있다는 것 확인


	![44](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/44.png)

- prompts의 Ensemble 기법 이용

	⇒ context의 diversity를 증가시켜 다양한 형태의 fact를 볼 수 있도록

- Performance on LAMA for BERT 7% 상승, Ensembling을 통해 4% 추가 상승
undefined- **Knowledge-driven downstream tasks**

	downstream tasks에 얼마나 잘 transfered 되느냐?

	- Relation extraction

		example: [Bill Gates] was born in [Seattle]; label : city of birth

	- Entity typing

		example: [Alice] robbed the bank; label: criminal

	- Question Answering

		example: “What kind of forest is the Amazon?”; label: “moist broadleaf forest”

- **Relation extraction performance on TACRED (관계 추출 데이터셋)**

	![45](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/45.png)

- **Entity performance on Open Entity**
	- 6,000개의 문장
	- 각 문장에는 세밀한 entity type이 주석으로 표시

	![46](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/46.png)

undefined- **Summary**

	![47](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/47.png)

- **Other exciting progress & what’s next?**

	![48](/assets/img/2023-08-13-CS224n---Lecture-15-(Add-Knowledge-to-Language-Models).md/48.png)

