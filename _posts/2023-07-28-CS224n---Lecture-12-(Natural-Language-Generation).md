---
layout: single
date: 2023-07-28
title: "CS224n - Lecture 12 (Natural Language Generation)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---


## Natural Language Generation



#### **NLG (Natural Language Generation)?**


→ 자연어처리(NLP) 분야의 sub-field로, 자연어가 아닌 정보로부터 **이해 가능하고, 일관되며 유용한  text (written / spoken)를 생성하는 system**을 만드는 것에 중점을 두는 분야이다.

- **Examples**
	- **Machine Translation**

		![0](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/0.png)

	- **Dialogue Systems**

		![1](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/1.png)

	- **Summarization**

		![2](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/2.png)

	- **Data-To-Text Generation**

		![3](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/3.png)

	- **Visual Description**

		![4](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/4.png)

	- **Creative Generation**

		![5](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/5.png)

- **최근 chatGPT, BARD 등의 생성 모델들이 인기를 끌며 폭발적으로 성장하고 있는 분야**


#### Basic of NLG


![6](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/6.png)

- 다양한 NLG 모델들이 **autoregressive한 방식**을 채택

	(한 step의 output이 다음 step의 input으로 들어감)

- 정리

	![7](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/7.png)

	- $S\in\mathbb{R}^V ~~~V$: vocabulary,  $t$ : time step,  $\theta$ : 모델의 parameters
	- $t$마다 each token에 대한 scores를 계산
	- $\{y\}_{<t}$의 일련의 text의 token을 받아, 새 token인 $\hat y_t$를 output으로 내놓음

	![8](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/8.png)


	→ 그 후, 이 score를 활용해서, **probability distribution인** $P$**를 계산**


![9](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/9.png)

- 이렇게 나온 score $S$를 기반으로 해서, softmax를 거쳐 **확률 분포 생성**

	![10](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/10.png)


	이렇게 **decoding algorithm**을 통해 나온 분포를 바탕으로 **최종 output**을 내놓음

- **train time**

	![11](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/11.png)


	train time에는, **negative log likelihood를 최소화하는 방향**으로 model을 훈련시킨다.

	- ***은 실제 정답 토큰임**을 **의미**함
	- 이 알고리즘을 “teacher forcing”이라고도 부름
	- 여기에 사용되는 토큰을 “gold”, “ground truth” token이라고도 함

	![12](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/12.png)


	![13](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/13.png)


	이렇게 첫 항을 거쳐 끝까지 진행되어 train이 끝나게 된다.


	⇒ 이 과정을 train



#### Decoding from NLG models


![14](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/14.png)

- 정리

	![15](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/15.png)

	- Encoding과 다른 점이, $\theta$가 없는 것 (decoding엔 직접 조정하지 않음)

	![16](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/16.png)

	- softmax function을 통해 scores에 대한 probability distribution $P$를 계산

		![17](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/17.png)

	- 이 확률분포로부터, token을 생성하는 decoding algorithm을 최종적으로 거쳐 token $\hat y_t$가 생성되는 것

		![18](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/18.png)


		→ 첫 생성 토큰인 $\hat y_1$ 이후로는, $\hat y_t$도 관여를 함


		이전 생성된 y, 지금 실제 y 

	- **Methods**
		- **Greedy Methods**
			- **Argmax Decoding**

				![19](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/19.png)


				⇒ 단순하게 **가장 큰 likelihood 가진 단어 token을 생성 token으로 선정**

			- **Beam Search**

				![20](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/20.png)


				⇒ 매 스텝 마다 **log likelihood가 높은 k개의 후보 중 선택 반복해 argmax보다 자연스러운 문장을 생성**함


			⇒ 이런 **Greedy Methods**는 고질적 문제가 있는데, 바로 **비슷한 단어를 반복 생성**한다는 것이다.


			![21](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/21.png)


			![22](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/22.png)


			![23](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/23.png)_openai : GPT_

			- 그림처럼, 반복이 거듭될수록, negative log likelihood가 점점 낮아져, **모델이 확신을 갖는(confident) 것**
			- 일반 LSTM 단위에선 문제가 되지 않지만, **Transformer 레벨로 올라가면 문제**가 되는 것을 알 수 있음

				→ 일반 RNN은 bottleneck이 일어나기 때문


				→ **bottleneck을 해결한 transformer에선 반복을 거듭하는 경향**이 발생함 


				→ 문제가 되는 것이, 어떤 문장을 15번 반복 말했을 때, 다음 문장 또한 같은 것을 말할 확률 높아짐 **⇒ big problem**


				⇒ repetition을 어떻게 해결할 수 있을까?


					**inference time**

					- **Don’t repeat n-grams**

						→ **같은 단어 반복 x** (simple)


					**train time**

					- **loss function**

						→ **다른 time step** 에서의 **hidden activation의 similarity를 최소화**

					- **coverage loss**

						→ 같은 토큰에 대해 penalty를 부과해, model이 강제로 다른 text를 생성하도록 하는 방법

					- **unlikelihood objective**

						⇒ 이미 생성된 token에 대해 penalty 부여


			하지만, 기본적으로 문제는 greedy algorithm에 존재


			⇒ **사람은 likelihood를 maximize하는 그런 방식으로 대화하지 않는다..!**


			![24](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/24.png)


			“사람이 쓴 text” / “Beam Search 방식으로 decoded된 text”의 per time step probability를 비교한 그래프


			확실히 **beam search로 decoded된 text의 각 선택은 probability가 높고, variance가 적은 경향**을 보인다.


			⇒ 이는, 그럴듯하지만 **사람의 방식과는 차이**가 있었고, 이런 방식을 반복한다면 문제


			⇒ 우리의 궁극적 목표는 **human language pattern의 uncertainty**와 **text의 decoding과 matching**


			⇒ sampling 방식을 다양화하는 방법을 도입

		- **Random Sampling**

			![25](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/25.png)


			![26](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/26.png)_가장 확률이 높은 restroom이 아니라 bathroom이 선택됨_

			- **higher variability sampling**

				다양한 출력을 얻기 위해 **확률적 요소를 도입**해, 가장 확률값이 높은 요소만 선택되는 것이 아닌, **확률에 기반한 sampling을 사용**


				→ 이를 통해 조금 더 의미 있는 stochasticity를 얻을 수 있음

			- **Problem**

				하지만, 이 분포는 굉장히 큰 vocabulary에 대한 분포이기 때문에, 랜덤하게 **현재의 문맥과 전혀 상관없는 이상한 token이 선택될 가능성**도 있음 (각각에 대해 선택 시엔 선택되지 않았겠지만, group으로써 선택했기 때문에 가능성이 있음)


				⇒ inference time에 pruning (위와 같은 엉뚱한 case 방지)

		- **Decoding: Top-k Sampling**

			![27](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/27.png)

			- 선택될 **확률이 높은 token k개 중에서 sampling** 하는 방법
			- parameter k
				- 5, 10, 20, ~ , 100
				- k가 높다

					→ diverse output


					k가 낮다


					→ safe하지만, 지루하고 generic 해짐


					(제일 위의 예시처럼 greedy 해짐)

			- **Problem**

				하지만 이렇게 고정된 k개를 고르는 것도 문제가 있는데, 아래의 예시와 같은 상황이 있을 수 있다.


				![28](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/28.png)

				- 분포가 **1) 위처럼 flat한 경우**와 **2) 아래처럼 특정 token들만 suitable한 경우**, 다른 방법이 필요해 보임

					1) 다양한 가능한 option들이 생성 방지됨


					2) 너무 가능성이 낮은 option들이 생성 방지 되지 않음


				⇒ 고정된 k개가 아니라 token의 분포 확률로 sampling

		- **Decoding: Top-p Sampling**

			![29](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/29.png)

			- 각 token의 probability를 따라 선정하기 때문에, 위와 같은 상황에dynamic한 처리가 가능해짐

				→ 위 상황에서 **flatness는 몇 개의 token을 sampling할지를 결정**


				→ **distribution을 rescaling**해 flatness를 조절해 decoding algorithm에 더 fit하게 만들기 위해 **temperature scaling**이라는 방법으로 구현


				**Temperature Annealing 방법도 있다.**

		- **Scaling randomness: Softmax temperature**

			![30](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/30.png)

			- 너무 flat하기도, 너무 peaky하기도 한 distribution을 우리가 원하는 형태의 distribution으로 rescale
			- **temperature hyperparameter (**$\tau$**)**
				- 각 token의 모든 score에 linear coefficient를 적용한 뒤에 softmax를 통과시킨다.
				- 이 linear coefficient를 결정하는 것이 temperature hyperparameter
					- $\tau $ > 1 : $P_t$ 를 더 uniform하게, flat 하게

						⇒ more diverse output

					- $\tau $ < 1 : $P_t$ 를 더 spiky하게

						⇒ less diverse output (concentrated on top words)

			- **Problem**
				- 하지만, 만약 **model의 distribution**이 **충분히 보정되지 않았다면?**

					→ 위의 방법은 model의 distribution을 바탕으로 scaling하기 때문에, 기존 distribution이 중요함


					→ solution - model의 distribution 외의 **outside info를 활용**할 수 있음 (decoding 시에)

		- **Improving decoding: re-balancing distributions**

			![31](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/31.png)

			- distribution을 어떤 식으로 바꿔야 할까?
				- relative한 magnitude로 바꾸기

					(해당 분포의 값이나 측정치의 크기나 진폭)

				- token들이 더 relevant하도록 rank를 바꾸기

					(embedded token 간 벡터 유사도 ex. cos similarity 등을 높이는 것으로 보임)


				**⇒ KNN language model**


					large corpus (db; cache)에서 phrase statistics를 사용해서 output probability distribution을 recalibrate


					(phrase statistics: 문장 내에서 구, 구문 또는 구문 단위의 통계 정보 나타내는 것; ex. 구의 등장 빈도, 길이, 위치, 구성 요소 등을 분석해 언어 데이터의 특성을 파악하고 이해함)

					1. db에 존재하는 phrase들의 vector representation을 initialize해 놓음
					2. decoding time에서 생성 중인 context의 vector representation과 1의 distance를 계산해 KNN 알고리즘 적용
					3. 이 정보를 바탕으로 normalize, aggregate 해서 outside info인 distribution 정보를 얻고
					4. 이 정보를 decoding time의 distribution과 interpolate 해서 최종적인 output을 계산한다.

					이 과정을 반복하는 방법


				Q. Cache?


				training corpus의 모든 phrase를 cache에 저장한 뒤, 매우 효율적인 알고리즘을 사용해 계산된 similarity에 대해 search해 가장 확률이 높은 것을 가져옴

			- 그렇다면, 다른 corpus에서 trained된 LM model → 다른 domain에서 사용할 수 있도록 rebalancing 할 수 있을까?

				**⇒ backpropagation-based distribution re-balancing**

		- **Backpropagation-based distribution re-balancing**

			![32](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/32.png)

			- **external objective를 define하는 classifier** (Attribute Model)

				→ 어떤 text에 대해 decode 했을 때, exhibit 되길 원하는 어떤 property를 approximate


				즉, 디코딩된 text가 특정한 특성 또는 속성을 근사하도록 하는 것


				**example** - dialogue model & sentiment classifier


				→ “dialogue를 positive하게 바꾸고 싶다.”

				1. dialogue model은 대화를 생성
				2. sentiment classifier의 평가를 통해 sequence의 positive score를 출력
				3. 이 property(여기선 positive score)를 계산한 후, dialogue model에서 back propagation
				4. 이를 통해, parameter의 update가 아닌 intermediate activations를 update한다.

					(즉, 모델 내부의 중간 단계에서 생성되는 활성화 값을 업데이트한다.)


					(왜? attribute model은 주어진 property에 대한 변형을 수행하기 위한 목적으로 사용되어서)


					(input을 변형하고, 중간 단계의 활성화값을 생성해 전달함)

				5. input으로 들어온 특정 단어를 변형하고, 중간 단계의 활성화값을 생성해 전달함 → forward propagation

			(GAN과 유사하다고 생각이 들었습니다.)

			- **Problem**
				- 잘못된 token을 생성했는지 여부를, rebalancing한 이후 알 수 있음 (nn search, discriminator 방식 둘 다)

					⇒ 이런 sequence output을 개선하기 위해 **re-ranker**

		- **Re-Ranker**
			- Multiple sequence를 decode

				→ 우리가 만든 sequence의 평가를 위해 **score를 정의**하고, 다음 이 점수에 따라 re-rank를 진행


				ex. perplexity


					하지만 Perplexity는 위에서의 문제점이었던 repetitive sequences에 대해 very low하다


				ex. attribute model

					- set of fixed sequence (multiple sequence?)를 평가하게 해서 back propagation을 진행
					- multiple reranker (여러 개의 attribute model)를 parallel하게 사용할 수도 있음

		![33](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/33.png)

		- 여전히 challenging한 문제 (2021년 기준)
		- 가장 좋은 접근법은 모델에 의해 생성되는 probability distribution해서 calibrate하여 사람과 비슷하게 하는 것

	### Training NLG models


	초장에서 언급했듯이, likelihood를 maximize하는 방식 만으로 training 시켰을 때 많은 문제점이 있었다. 


	⇒ 이를 해결하기 위해 **unlikelihood training**이라는 새로운 접근법 등장했다.

	- **unlikelihood training**

		![34](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/34.png)

		- 특정 context에서 particular token들에 대한 production을 discourage

			→ model은 training corpus로부터 distribution은 학습하는 동시에, repetition을 제한하고 더 다양한 text를 생성

		- negative log likelihood 최소화 (teacher forcing)

			![35](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/35.png)_teacher forcing_


			![36](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/36.png)_unlikelihood training_

			- 두 식의 차이를 보면, unlikelihood training은 $y_{neg}$ (아마 negative sample?)의 repetition을 제한할 수 있도록 probability를 낮추는 방식으로 학습하는 것으로 보임
	- **Exposure Bias**

		![37](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/37.png)

		- maximize likelihood로 부터 야기되는 문제로 **exposure bias**
			- train time에 model의 학습을 위해 input으로 준 “gold context” (from real, human-generated texts)
			- generation time에 model이 token을 생성하기 위해, input으로 받는 previously-decoded tokens

			→ 이 둘은 다르다.

		- 위에서 언급했듯이, model이 생성하는 text의 type과 human language의 양상은 매우 다름
		- 이 문제의 해결을 위한 여러 방법이 존재
			- Scheduled sampling
			- Dataset Aggregation
			- **Sequence re-writing**

				주어진 human-written prototype의 sequence가 있을 때, 이 sequence를 add, remove, modify


				ex. 입력 문장의 단어 순서 변경, 문법적인 구조 수정, 동의어나 유의어 사용해 출력 문장 생성


				→ 이런 data augmentation을 통해 데이터의 다양성을 확보하고 더 많은 sample을 생성함

			- **Reinforcement Learning**
	- **Reinforcement Learning**

		![38](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/38.png)

		- Markov decision process로 text generation model을 형상화
		- 즉, generate하는 token마다 어떠한 reward를 받아 이 reward를 통해 loss function을 scaling하는 것
		- 용어
			- State $s$

				→ 앞선 context를 거쳐온 모델의 현재 상태

			- Actions $a$

				→ 생성될 가능성이 있는 words를 의미

			- Policy $\pi$

				→ decoder를 의미

			- Rewards $r$

				→ external score에 의해 제공 받음


				→ 이를 통해 loss function scaling


		![39](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/39.png)


		![40](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/40.png)


		→ reward가 높으면 유사한 context에 대해 동일한 sequence를 generate할 확률이 증가

		- Reward Estimation

			![41](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/41.png)

			- 특정 behavior(action)을 encourage하기 위한 reward로 어떤 것이 사용될 수 있을까?

				→ generation pipeline에 따라 달라지지만 최근의 흐름은, 자신의 final evaluation에 대해 reward를 set하는 것

			- 하지만, text generation에 대한 metric은 단지 approximation이었고, 이 근사치에 optimize하는 것이 좋은 방법은 아니었음

				(실제로 Google Machine Translation 논문에선 RL with BLUE score가 translation quality를 향상하지 못했다고 나옴)


			그래서 다양한 reward에 대한 연구가 나왔다.


			![42](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/42.png)


			심지어 reward 자체에 neural network을 사용하는 방법들도 많이 나옴

		- dark side

			![43](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/43.png)

			1. RL은 매우 unstable

				1) 모델이 환경과 상호작용하면서 학습을 진행하기 때문에, noise, 변동성, 비정상적 상태로 인해 학습이 불안정해짐


				2) 학습 과정에서 얻는 보상 신호가 지연된 형태로 주어짐


				→ 늦은 feedback


				→ 신뢰할 수 있는 학습 신호의 제공이 어려움

			2. setup에 매우 정확한 tuning이 필요함

				→ 초기 학습 단계에 데이터의 품질과 다양성에 대한 영향 받기 때문에, 초기 학습의 안정성과 효율성을 높여야 함

			3. 처음부터 RL로 모델을 학습하긴 어렵기 때문에, 보통 teacher forcing이나 baseline reward와 같은 것을 통해 모델을 학습함
undefined		- 결국…

			![44](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/44.png)

			- teacher forcing이 coherent text를 얻기 위한 mainstream
			- 여전히 diversity, exposure bias에 대한 문제 존재

	### Evaluating NLG Systems


	![45](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/45.png)


	세가지 대분류 가능

	1. content overlap metrics → automatic evaluation metric
	2. model-based의 automatic evaluation metric
	3. human evaluation
	- **content overlap metric**

		![46](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/46.png)

		- 두 가지 sequence에 대한 explicit similarity를 계산함

			1) N-gram overlap metric


			2) Semantic overlap metric


		**N-gram overlap metric**


		![47](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/47.png)

		- 매우 빠르고 효율적인 방법으로, 대부분의 N-gram overlap은 좋은 sequence quality의 approximation을 제공하지 못함
		- Failure example

			![48](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/48.png)


			![49](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/49.png)


			박스 안을 보면, 오른쪽의 것들과는 다르게 human의 judgement 양상과 매우 다름


			또한, story generation과 같은 open-ended task가 있을 때, stopwords와 많이 matching 되었고, 실제 content와 상관 없는 story가 만들어지기도 했다.


		**Semantic overlap metrics**


		![50](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/50.png)


		이 중 SPICE는 generated text에 대한 graph를 만들어, reference caption과 비교하는 방법을 사용한다고 함

	- **Model-based metrics**

		![51](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/51.png)

		- Vector Similarity

			Composition function을 정의해, generated text와 거리를 계산하는 방식

		- Word Mover’s distance

			각각에 대응되는 word와의 거리를 계산

		- BERT-SCORE

		![52](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/52.png)

		- Sentence Movers Similarity

			Word mover의 확장버전으로 single sentence가 아니라, 긴 여러 문장에 대해서 유사도를 계산함

		- BLUERT

			BERT 기반 regression model. grammatical, meaning에 대한 점수 반환

	- **Human evaluations**

		![53](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/53.png)

		- 하지만, 가장 중요한 것은 human에게 valuable한지 여부이다.

			→ 이를 통해 새로운 machine learning model을 만들기도 함


		![54](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/54.png)

		- 다음과 같은 특정 criteria를 정의하고, 사람들에게 평가를 내리는 것
		- 하지만, 다른 연구와 비교하면 안 된다고 한다.

			→ 인간 평가는 주관적인 요소가 포함


			→ 평가자의 선호나 의견 등에 영향을 받음


			…


			이런 이유로 다른 연구와의 비교 보다는, 동일한 실험 설정이나 평가 방식을 통해 일관된 비교를 수행하는 것이 좋음


		![55](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/55.png)


		![56](/assets/img/2023-07-28-CS224n---Lecture-12-(Natural-Language-Generation).md/56.png)

