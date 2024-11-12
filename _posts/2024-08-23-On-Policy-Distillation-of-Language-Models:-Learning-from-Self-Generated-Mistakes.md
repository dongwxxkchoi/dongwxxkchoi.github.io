---
layout: single
date: 2024-08-23
title: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
use_math: true
author_profile: false
tags: [논문 정리, ]
categories: [AI, ]
---


## 논문 소개 

1. 제목 : On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
2. keyword : Knowledge Distillation, Auto-regressive model, On-policy generated data, Imitation Learning, RLHF/RLAIF
3. 학회/저널 : ICLR
4. 출판년도 : 2024
5. 논문

[bookmark](https://arxiv.org/abs/2306.13649)



## 본문 소개



### 논문 선정 이유


LLM의 Knowledge Distillation 관련 최신 논문. On-policy generated data를 이용한다는 점과, RLHF와의 쉬운 통합이 가능하다는 것이 인상 깊어서 선택했습니다.



### 논문 3문장 요약

- Issue: teacher-student의 distribution mismatch
- Limitation of previous works

	1. Fixed dataset의 이용 


	2. KL의 선택

- Contribution
	1. on-policy student generated outputs 이용 distillation 고안
	2. RLHF/RLAIF 등과 연결될 수 있는 GKD 제시
	3. Distillation에서의 on-policy student generated output sequence의 중요성 확인
	4. student-teacher 간의 최적의 divergence가 task-dependent 하다는 통찰 제공


### 3. Introduction


LLM과 같은 auto-regressive sequence models는 parameter 수를 증가 시킴으로써, 성능을 증대시켰고, 이는 배포의 어려움을 초래한다. 따라서, 이런 **큰 모델들의 성능을 유지하면서 parameter 수를 줄이는 것이 목표**가 되고 있다.


Model Compression 기법 중 하나가 **knowledge distillation [1]**이다. 하지만, auto-regressive sequence models에서 teacher model의 output sequence 또는 token-level의 **fixed set을 만들어 사용하는 방식**은 **1) expensive**하고, 2) **teacher-student 간 output의 distribution mismatch**(Imitation Learning에서의 문제점과 동일. DAgger[2] 등장)로 이끌 수 있다. **forward KL divergence를 최소화하는 현재 방식**은, student의 expression ability 부족으로 **성능이 좋지 못하다**.


이 논문에서는, **Generalized KD (GKD)**를 통해 문제를 해결했다. **auto-regressive sequence models에 대한 KD를 Imitation learning으로 간주**했고, **student의 self-generated sequence를 통해 학습**할 수 있도록 했다. 또한, **reverse KL과 generalized JSD** 등을 통해 student의 expression ability 부족 문제를 해결했다.


GKD는 autoregressive LMs에 대한 KD methods를 통합했다. 서로 다른 크기의 T5 모델에 on-policy methods를 도입해, 성능 향상을 이끌었다. 


**Key contributions**는 다음과 같다.

1. on-policy student-generated ouputs를 활용하는 GKD를 통해, auto-regressive LMs의 **training-inference 시의 mismatch를 해결**했다.
2. **RLAIF같은 LM에서의 RL fine-tuning과 on-policy GKD를 혼합**할 수 있음을 증명했다.
3. Distillation 도중 **student-generated on-policy output sequences을 활용하는 것의 중요성**과 작업 특성에 따라, **teacher-student 간 최적의 Divergence가 달라져야 한다는 점에 대한 insight를 제공**했다.


### 4. Related works



#### **Knowledge Distillation**

- **supervisedKD,  seqKD[3]**

	기존의 KD 방식들임. 

- **Hidden states / Attention scores**

	Jiao et al. (2020)[4], Wang et al. (2020)[5]의 연구에서 teacher model의 hidden states / attention scores를 모방하도록 teacher model을 훈련시켰음. 


	하지만, KD - imitation learning 간의 연결성을 충분히 탐구하지 못했으며, distribution mismatch는 해결하지 못했다.

- **ImitKD**

	ImitKd(Lin et al., 2020)[6]은 Imitation Learning 개념을 도입했음.


	하지만, on-policy data 수집과 RL fine-tuning 과의 통합에 대한 고려는 없었음.

- **F-distill**

	F-distill(Wen et al., 2023)[7]은 다양한 divergence를 도입했음.


	하지만, GKD 보다 성능이 낮았다.

- **MiniLLM**

	MiniLLM(Gu et al., 2023)[8]은 Imitation Learning 개념을 도입했고, KD 자체를 RL 문제로 framing 했음. policy gradient를 통해 optimization을 수행했다.


	하지만, 굉장히 복잡했고, GKD에 비해 성능이 낮았다.



#### **Divergence**


![0](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/0.png)

- forward KL

	![1](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/1.png)


	forward KL 사용 시, P가 0이 아닌 곳에서, Q도 0이 아닌 값을 가지도록 학습된다. 따라서, 여러 모드들을 전체적으로 넓게 커버링하도록 학습된다.

- reverse KL

	![2](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/2.png)


	reverse KL 사용 시, P가 0인 곳에서, Q가 0이 되도록 학습된다. 주로 P의 모드 중 하나에 집중된다.

- JSD

	![3](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/3.png)


	$\beta=\frac{1}{2}$을 사용시, symmetric하며 bounded한 특성을 갖는다. 또한, $\beta$를 통해 forward, reverse KL 중 어디를 더 많이 반영할 지를 정할 수 있습니다.


**Speculate Decoding**


LLM에서 텍스트 생성 시의 inference 속도를 향상시키는 기술로, small draft model과 big target model을 이용해 과정을 최적화한다. 최근의 두 연구 Zhou et al. (2023)[9]과 Liu et al. (2023)[10]에서 GKD를 이용해 두 모델 간의 alignment를 개선했다.



### 5. Method


**Problem Setup**


두 개의 auto-regressive sequence models,  student $p_S$, teacher $p_T$


student가 learnable parameters $\theta$를 갖고, $p_s^{\theta}$가 $\theta$에 대해 미분가능하다.


inputs $X$가 주어졌고, 그에 대한 dataset으로 input-output sequence pairs $(X,Y)$가 있다.


(또는 Teacher model의 generated sequence를 $Y$로 활용할 수 있다.)


divergence $\mathcal{D}$로 $p_T$와 $p_S$의 token-level distributions 사이의 차이를 정의한다.


![4](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/4.png)


**Supervised FT**


fixed-dataset이 있고 teacher의 feedback이 없다고 할 때, 가장 간단한 방식은 student policy에서 negative log-likelihood를 최소화하는 것이다.


![5](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/5.png)


**Sequence-Level KD**


SeqKD(Kim & Rush, 2016)은 teacher에 의해 생성된 high probability sequences의 likelihood을 maximize하는 방법으로 이뤄진다. 이는 teacher-generated outputs에 대한 supervised FT로 볼 수 있다.


**Supervised KD**


Student가 teacher의 token-level probability distributions을 모방하는 방법으로 이뤄진다. $p_S$는 supervised objective $L_{SD}$로 훈련된다.


![6](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/6.png)



#### 5.1 Generalized Knowledge Distillation (GKD)


Imitation Learning(IL) 중 on-policy imitation approaches(DAgger)을 통해 **student policy를 통해 sequence를 반복적으로 수집**하고, 해당 **sequence에 대한 expert labels**을 얻고, **해당 dataset에서 student를 train**시키는 방법을 제안한다.


**Student의** **self-generated output sequences에 대한 erroneous tokens**에 대해 **teacher의 logit으로 부터 token-specific feedback을 받는 이 방법**을 **on-policy KD** 라고 한다. student는 $y_{<n}$ **상태에서, teacher의 token-level distributions인**  $p_T(y_n\|x)$**을 모방**한다. on-policy loss인 $\mathcal{L}_{OD}$는 아래와 같이 정의된다.


![7](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/7.png)


해당 방법 수행 시, teacher의 feedback을 받기 때문에, student’s sampling distribution인 $p_S(\cdot\|x)$에서의 backpropagation이 일어나지 않는다. 이는 training을 안정적으로 만들고, computationally efficient하다. training은 temperature $\gamma=1$로 다양한 sequence를 generate하도록 한다. 또한, student를 이용해 sequence를 generate하는 것은 teacher를 이용하는 것에 비해 cost도 적게 든다.


on-policy KD에 더해 supervised approach와 on-policy approach를 통합했고, 이를 Generalized KD (GKD)라고 한다. 따라서, GKD는 output-sequence로 fixed dataset과 on-policy student-generated sequences를 둘 다 사용한다. 따라서, GKD는 다음 objective를 최소화한다.


![8](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/8.png)


$\mathcal{D}(p_T,p_S)(y\|x)$은 teacher-student distributions간의 divergence이고, $\lambda\in[0,1]$은 student data fraction을 조절하는 hyper-parameter이다. $\lambda$ 값에 따른 GKD의 결과는 앞으로 확인해 볼 것이다.


![9](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/9.png)


**Remark**


**해당 과정은 RLHF와 상당히 유사하다. 특히,** Student model이 Supervised Fine-Tuning을 통해 어느정도 학습된 상태 가정한다면, **1) SFT**로 시작해서, **2) Expert의 feedback**을 받는 일련의 과정이 RLHF와 유사하다. GKD는 RLHF에서의 hyperparameter tuning insight를 활용해, 추가적인 hyperparameter 없이 적은 overhead 만으로 **쉽게 RLHF와 통합**될 수 있다.



#### 5.2 RL Fine-tuning + On-policy GKD


distillation은 주요 objective를 직접적으로 최적화하는 것이 아닌, 대리적인 방법으로 이용될 수 있으며, 미분불가능한 경우도 있다. 우리는 이 objective를 RL을 통해 optimize할 수 있다. 특히, on-policy KD의 경우는 student의 output만을 요구하므로, RLHF 등과 쉽게 결합될 수 있다. student policy에 대한 reward $r$을 teacher policy에 가깝도록 유지하게 optimize 하면 된다. 


![10](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/10.png)


$\alpha\in[0,1]$을 통해 RL objective에 대한 distillation 강도를 조절할 수 있다. 이를 통해, human preference에 model alignment하는 alignment tax를 줄일 수 있다. 실험을 통해 RLAIF와의 통합으로 hallucination을 줄일 수 있었고, 동시에 distillation을 통한 downstream performance를 증대할 수 있었다.


![11](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/11.png)


**Remark**


RLHF[11], RLAIF[12]을 사용할 때, reverseKL을 이용해 정책이 너무 변화하지 않도록 제약을 걸어주는 역할을 한다. ReverseKL, JSD(0.9)을 사용해 세밀한 조정을 할 수 있다.



#### 6. Result


**Setup**

- Teacher Models: T5-XL (~ 3B)
- Student Models: T5-small (77M), T5-base (250M), T5-large (800M)
- Divergence: forward KL, reverse KL, JSD(0.1), JSD(0.5), JSD(0.9)
- Data fraction 𝜆: 𝜆=1 (On-Policy), 𝜆=0.5 (Mixed), 𝜆=0 (Supervised)
- Baseline models: SeqKD, SupervisedKD, ImitKD, f-distill

**Abstractive Summarization**


Dataset: Xsum dataset
Metric: ROUGE-2, ROUGE-L, ROUGE-1


![12](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/12.png)


GKD가 기존 baseline 보다 성능 증대했으며, 적은 학습 데이터셋에서도 좋은 성능을 보였고, 데이터 증가에 따른 성능의 향상도 좋았음.


Dataset: Xsum dataset
Metric: Self-BLEU


![13](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/13.png)


JSD(0.9), ReverseKL의 경우는 특히, Self-BLEU가 다른 divergence에 비해 높아, diversity가 감소하는 것을 알 수 있음.


Dataset: Xsum dataset
Metric: ROUGE-2, Entailment


![14](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/14.png)


GKD + RLHF 에서 α가 커질수록, **요약의 품질(ROUGE-2)은 올라가나**, **사실적 일관성(Entailment)은 감소**


**Machine Translation**


Dataset: WMT14 en-de dataset 
Metric: BLEU
Method: Beam search를 통해 얻은 3개의 결과를 평균냄


![15](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/15.png)


![16](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/16.png)


GKD (On-policy) 방법의 성능이 제일 좋았다.


Dataset: WMT14 en-de dataset
Metric: BLEU


![17](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/17.png)


Machine translation의 경우는 Forward KL, Reverse KL에 비해 **JSD의 경우가 성능 향상 폭이 컸음**


**Arithmetic Reasoning**


Dataset: GSM8K dataset
Metric: Accuracy
Setup: Wei et al. (2022)에서의 CoT 예제 4개 추가해, few-shot prompting으로 문제를 풀도록 함
Teacher: Flan T5-XL   Student: Flan T5-Base, Small


![18](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/18.png)


Arithmetic Reasoning의 경우는 **Forward KL, Reverse KL이** JSD에 비해 **성능이 좋았음**


**Task Agnostic Distillation**


Train Dataset: FLAN2021    
Test Dataset: MMLU, BBHMetric: Few-Shot Prompted Accuracy
Teacher: Flan T5-XL   
Student: FLAN T5-Base
Train: Instruction – Answer pair을 이용해 학습 수행


![19](/assets/img/2024-08-23-On-Policy-Distillation-of-Language-Models:-Learning-from-Self-Generated-Mistakes.md/19.png)


|            | **MMLU** | **BBH** |
| ---------- | -------- | ------- |
| FLAN-T5 XL | 52.4%    | 41%     |
| T5-Large   | 35.6%    | 31.25%  |

undefined
On Policy with Reverse KL이 가장 성능이 좋았음



#### 8. Conclusion


Auto-regressive model에서의 train-test distribution mismatch를 위해 on-policy GKD을 고안했다.


1) teacher의 token-level guide를 이용한 on-policy student-generated outputs 이용 distillation 고안했고, 2) LM의 RL 학습(ex. RLAIF)과 연결될 수 있는 GKD를 제시했으며, 3) Distillation에서 student generated on-policy output sequence의 중요성을 확인했으며, 4) student-teacher 간의 최적의 divergence가 task-dependent 하다는 통찰을 제공했다. 


논문에선 다른 auto-regressive models (audio, video, text-to-image)로의 연구에 적용해보는 것을 제안했다. 



#### 9. Reference


[1] _Distilling the Knowledge in a Neural Network_ (Hinton et al., 2014)


[2] _A reduction of imitation learning and structured prediction to no-regret online learning_ (Ross et al., 2011)


[3] _Sequence-level knowledge distillation_ (Kim & Rush, 2016)


[4] _Tinybert: Distilling bert for natural language understanding_ (Jiao et al., 2020)


[5] _Minillm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers_ (Wang et al., 2020)


[6] _Autoregressive knowledge distillation through imitation learning_ (Lin et al., 2020)


[7] _f-divergence minimization for sequence-level knowledge distillation._(Wen et al., 2023)


[8] _Knowledge distillation of large language models_ (Gu et al., 2023)


[9] _Distillspec: Improving speculative decoding via knowledge distillation_ (Zhou et al., 2023)


[10] _Online speculative decoding_ (Liu et al., 2023)


[11] _Training language models to follow instructions with human feedback_ (Ouyang et al., 2022)


[12] _RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback_ (Lee et al., 2023)

