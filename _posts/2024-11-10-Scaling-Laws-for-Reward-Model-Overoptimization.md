---
layout: single
date: 2024-11-10
title: "Scaling Laws for Reward Model Overoptimization"
use_math: true
author_profile: false
tags: [논문 정리, ]
categories: [AI, ]
---


## 논문 소개 

1. 제목 : Scaling Laws for Reward Model Overoptimization
2. keyword

	`RLHF`, `Overoptimization`, `Reward Hacking`, `Goodhart’s Law`, `Scaling Laws`

3. 학회/저널 : ICML
4. 출판년도 : 2023
5. 논문

	[https://arxiv.org/abs/2210.10760](https://arxiv.org/abs/2210.10760)



## 본문 소개


어렵다고 느낀 논문이었는데, 설명이 조금 불친절한 감이 없지 않아 있습니다. 정말 수많은 graph 들이 나오기 때문에, 처음에 experiment set을 정확히 하고 넘어가지 않으면 나중에 헷갈리는 경우가 발생합니다. 글 작성에 해당 유튜브 영상의 도움을 많이 받았습니다. 글 만으로 헷갈리시는 분들은 해당 영상을 참고해주시기 바랍니다.


출처: [https://www.youtube.com/watch?v=0UzLOIhfN94](https://www.youtube.com/watch?v=0UzLOIhfN94)



#### 논문 3문장 요약

- Issue: RLHF에서 **reward model의 optimization을 지나치게** 적용할 시, 모델이 proxy reward model의 한계를 악용해 **reward hacking 문제**를 야기
- Limitation of previous works: 기존 연구들은 reward model overoptimization의 구체적인 scaling laws나 optimization에 따른 성능 저하를 체계적으로 분석하지 못함
- Contribution: **KL budget과 gold reward score 간의 functional form**을 empirical method를 통해 규명하고, optimization method, dataset size, model parameter size 등이 **성능에 미치는 영향을 정성적으로 분석**함


#### 1. Introduction


**`Goodhart’s Law`****는** “**measure가 target이 되는 순간, 그것은 더 이상 좋은 measure가 아니다**.”라는 격언을 의미합니다. ML에선, reward model이나 discriminator(ex. GAN)와 같은 **static learned models의 proxy objective**에서 발생합니다. **너무 과도하게 optimizing 해서 모델이 true objective를 저해**하게 되는 현상을 **`Overoptimization`** 이라고 합니다. 이 효과의 크기와 scaling 방식, 실증적 연구를 통한 neural network에서의 Goodhart’s Law의 이론적 모델 개발을 통해 **미래 AI system의 misalignment 문제를 방지하는 데에 핵심적인 역할**을 할 수 있습니다.


이를 알기 위해, **RLHF 등에서 reward model을 이용한 LLM fine-tuning**의 맥락에서 설명하겠습니다. **optimization**은 주로 **policy-gradient 기반 RL**과 **best-of-**$n$ 방식을 사용합니다. Overoptimization은 두 방법 모두에서 발생할 수 있기 때문에, 두 방법 모두를 탐구하고 각 방법에서 overoptimization이 어떻게 발생하는지를 살펴보겠습니다.


overoptimization 연구에 드는 human preference labels의 cost를 해결하기 위해, **synthetic setup**을 사용합니다. Human preference labels로 train된 gold-standard RM이 제공하는 label을 사용하는 방법입니다.


논문의 주요 결과는 **empirical한 방법으로 검증**하여, **KL divergence(**$D_\text{KL}(\pi ||\pi_\text{init}$**))에 따른 reward model scores** $R$**의 변화**를 **functional form**으로 나타내었습니다. $D_\text{KL}(\pi ||\pi_\text{init}$)은 RL training 중 단조 증가하고, n의 function으로 계산될 수 있습니다. Anthropic의 논문[1]에 따라, $d:=\sqrt{D_\text{KL}(\pi||\pi_\text{init})}$**을 정의해** $d$**에 대해 functional form을 정리**합니다.


![0](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/0.png)


translation invariant property에 의거해 $R(0):=0$으로 설정했습니다. parameters인 $\alpha_\text{RL},\beta_\text{RL},\alpha_{\text{bo}n},\alpha_{\text{bo}n}$은 1) proxy reward model parametes, 2) size of the proxy reward model dataset, 등에 영향을 받습니다. 


또한, 이런 quantitive results에 더해 qualitative trends 까지 발견했습니다.


1) **RL vs best-of-**$n$

- **RL이 optimization, overoptimization에서 bo**$n$ **보다 느림**. 따라서, **KL** **divergence** **만을** **이용한 비교는 부적절**함
- **proxy reward score와 gold reward score의 관계는 유사**함

2) **Smooth coefficient scaling**

- $\alpha,\beta$는 proxy reward model parameters의 수에 따라 smooth하게 변화함. (approximate logarithmic trends) 이를 통해 어느 정도 **gold reward model score의 예측이 가능**함

3) **Weak dependence on policy size**

- **Larger policies**는 전반적으로 **더 좋은 성능**을 보였고, 그로 인해 optimizing 시, **optimization으로 인한 추가적 성능 향상은 적었음**.
- 하지만, **overoptimization의 정도는 policy size에 큰 영향을 받지 않았음**

4) **KL penalty ineffectiveness**

- **KL penalty의 사용**은 **KL divergence 대비 proxy rm score를 증가**시킴
- 하지만, **gold RM score-KL frontier**에서의 **measurable improvement는 없었음**

이를 바탕으로 RLHF, Goodhart’s Law, AI Alignment에 적용될 영향을 논의해보겠습니다


* **translation invariant property**: 어떤 system이나 function이 input value의 translation (평행 이동)에 영향을 받지 않는 성질을 의미함. (ex. $f(x+c)=f(x)$)  reward score는 절대적인 값이 중요한 것이 아닌, 상대적 차이가 중요하기 때문에?



#### 2. Methodology


**InstructGPT[2]와 동일한 setting**을 사용했습니다. 기본 setting은 다음과 같습니다.

- observations: text prompts
- policy: prompt에 대한 response 생성
- prompts: 다양한 language model tasks를 묘사하는 자연어 instructions

학습된 RM이 해당 response에 reward signal을 제공해, RL / BoN optimization에 사용합니다.


Policy model과 Reward model의 setting은 다음과 같습니다.

- 모든 Initial policies는 2 epochs 동안 human-generated InstructGPT demonstrations로 SFT
- 모든 RM은 GPT-3 architecture 사용하지만 scalar head 추가해 reward output

RL의 setting은 다음과 같습니다.

- RL은 PPO[3] 사용
- KL penalty를 test하는 Section 3.6 제외, KL penalty는 0으로 setting

	(default PPO hyperparameters 사용했기 때문에, 다른 hyperparameter setting에선 다른 경향 나타날 수 있음)


BoN의 setting은 다음과 같습니다.

- BoN에선, policy에 대한 n trajectories 생성해 highest proxy RM score 갖는 것을 선택
- Unbiased estimator[4] 사용해 BoN gold, proxy scores를 계산함

	(n개의 samples의 반복적 생성 후 가장 높은 gold, proxy RM score의 평균을 구하는 naive estimator 보다 효율적이고, 분산이 낮다고 함)

- BoN의 KL divergence는 $KL_{\text{bo}n}=\log n-\frac{n-1}{n}$으로 계산될 수 있음

![1](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/1.png)

- gold RM의 output을 ground truth로 정의하는 synthetic setup을 사용
- gold RM은 6B reward model 사용
- proxy RM은 3M to 3B parameters model 사용
- 100,000개의 synthetic comparisons dataset 구축. 높은 gold reward score를 가진 response가 선호되도록 함
- 10,000개의 data는 validation loss 계산 위해 남김

**2.2 Recalibration**


RM score는 translation-invariant property을 갖기 때문에, 실험 후 비교를 위해 조정했습니다. 

- initial policy average reward를 0으로 하도록 RM을 recenter
- normalize the variance of gold RM scores
- recalibration of proxy RMs

	hard thresholding(즉, 이진 레이블 생성 방식)으로 인해 proxy RMs이 miscalibrated 됐을 수 있다. 따라서, validation set의 soft labels를 이용해 cross-entropy loss를 최소화하도록 logit을 rescaling 했음.


(이는 BoN과 RL에 영향을 미치지 않는다. BoN은 상대적 순위에, RL의 Adam optimizer는 loss의 상대적 변화에 반응하기 때문이다.)


---


이 setting을 잘 보는 것이 중요하기 때문에, 한번 더 정리하고 넘어가겠습니다. 이 논문에선 기본적으로 **3개의 모델이 등장**합니다. **1) Gold RM, 2) Proxy RM, 3) Policy (LLM)** 


![2](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/2.png)


(아래 정리에 쓰이는 용어들은 설명을 위해 제가 임의로 정의했습니다.) 


1) **Gold RM**은 **Gold standard의 역할을 하는 Reward Model**로 human preference data($\mathcal{D}_\text{h}=\{(x_i,y_{il},y_{iw})|i=1,2,...,32\text k\}$)를 통해 직접 학습됩니다. 논문에선 **6B 크기의 GPT-3 model**을 사용했습니다. 이 Gold RM은 100K의 synthetic dataset 구축에 사용되어, 마찬가지로 gold RM preference dataset($\mathcal{D}_\text{g}=\{(x_i,y_{il},y_{iw})|i=1,2,...,100\text k$)을 구축합니다.


2) **Proxy RM**은 $\mathcal{D}_g$를 이용해 human preference를 학습합니다. **3M to 3B의 다양한 크기의 GPT-3 model**을 사용하며, $\mathcal{D}_g$의 약 10% 가량은 validation set을 위해 남깁니다. 


3) **Policy**는 **Proxy RM을 통해 직접적으로 학습되는 LLM**을 의미합니다. **1.2B, 6B의 GPT-3 model을 사용**하는 것으로 나와있고, optimization 방식으론 RL(PPO), BoN(unbiased estimator) 방법을 사용합니다. 


Anthropic 논문[1]에서 사용한 square root of KL divergence $\sqrt{D_\text{KL}(\pi||\pi_\text{init})}$을 일종의 optimization ratio로서 사용합니다. 이를 KL budget이라고도 표현하는데, KL budget에 따른 RM score의 변화 등을 살펴보고 이를 위에서 언급한 functional form으로 나타냅니다.


![3](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/3.png)


이 과정에서, KL divergence, RM Score (Gold / Proxy), RM Data Size, RM validation loss, coefficients parameters 등등 다양한 변수들 간의 correlation들을 살펴보는 것이 목적인 논문입니다. 


이렇게 한번 더 정리하고 넘어가면 훨씬 논문 이해가 쉬우실 것이라 생각합니다.


---



#### 3. Results


6가지 측면에서 실험 결과를 정리합니다. 하나하나 정리해보겠습니다.


**3.1 Fitting and validating functional forms**


희미한 선이 regression을 통해 예측된 rm score입니다. 


![4](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/4.png)


BoN의 경우, n≤1000, KL divergence = 6 nats 까지의 data를 기반으로 가설을 세웠습니다. 가설을 n≤60000, KL divergence = 10 nats의 data를 통해 예측 검증을 수행했습니다.


![5](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/5.png)


RL의 경우는 전체 데이터를 보고 모델링을 수행했으며, 대신 parameter의 선택은 일부 데이터만(40 nats 이전)을 보고 선택했다고 합니다. 40 nats 이후에 대해선 extraplolation test를 수행했습니다. 


![6](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/6.png)


**Proxy scores에 대해서도 modeling**을 시도했습니다. 하지만, 두 경우 모두 쉽지 않았고, 특히 BoN의 경우 보기엔 regression이 잘 된 것처럼 보이지만 **성능이 좋지 않다**고 했습니다. proxy RM score에 대한 이해는 future work로 남겼습니다.


**3.2 Scaling with RM Parameter Count**


RM의 Parameter size에 대해 coefficient의 변화를 살펴봤습니다.


policy는 1.2B, data size는 90,000으로 설정하고 실험을 진행했습니다.


![7](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/7.png)


gold RM scores에 대한 functional form에서 **RM size를 scaling 할수록 coefficient** $\alpha_{\text{bo}n},\beta_{\text{bo}n}$**은 smoothly 변화**했습니다.


![8](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/8.png)


반면, **RL의 경우 coefficient** $\alpha_{RL}$**는 RM sizes 상관없이 constant**했으며, $\beta_{RL}$**은 마찬가지로 smooth하게 변화**했습니다. (clean scaling curve)


![9](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/9.png)


이를 통해 우리는 **다양한 RM Size에서의 peak gold RM scores를 예측할 수 있습니다**. 


proxy score에서의 경우, BoN의 coefficients도 비슷한 양상을 보였다고 합니다. 특히 $\beta_{\text{bo}n}$은 더 낮은 값을 갖는다고 하는데, 3.1에서 언급했듯이, less confident라고 합니다. 높은 KL에선 proxy reward model이 값을 underestimate하는 경향이 있고, 두 방법 모두에서 linearly grow하는 양상을 보인다고 합니다.


**3.3 Scaling with RM Data Size**


RM Data Size가 미치는 영향을 살펴봅니다.


RM size는 12M으로 고정한 채, RM data size를 바꿔가면서 실험을 진행합니다. 전반적으로, **data가 많을수록 더 좋은 gold scores와 더 적은 overoptimization의 결과를 나타냈습니다**. 


![10](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/10.png)


coefficients $\alpha,\beta$는 3.2와는 달리 상관관계를 보이지 않았습니다.


![11](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/11.png)


**모든 Size의 RM의 경우에 near-chance loss(Data Size ≤ 2,000)는 유의미한 변화가 없었습니다**. 하지만, 해당 **threshold를 넘어서면 RM의 크기가 클수록 Data Size가 커짐에 따라 RM val loss가 빠르게 더 많이 감소**했습니다. 이 **threshold는 RM size에 상관 없이 유지**되었습니다.


![12](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/12.png)


이 경향성은 **Data Size와 Final Gold RM Score 간의 관계에서도 유효**했습니다.


epoch이 미치는 영향도 확인했는데, data size를 유지한 채, epoch만 늘리는 경우는 큰 영향을 미치지 못했다고 합니다. 반면, epoch을 유지한 채 data size를 늘리는 경우는 상당한 영향을 미쳤다고 합니다.


![13](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/13.png)


또한, **같은 validation loss의 두 RM은 RM size / RM data size에 상관 없이 optimization에 대한 같은 robustness를 갖는다고 가설**을 세우고 실험을 진행했습니다. 결과는 가설에 대한 **weak evidence**를 제시했습니다. 다른 setting엔 상관 없이 어느 정도 경향성은 보입니다.


**3.4 Scaling with Policy Size**


policy size의 impact는 어떻게 볼 것인가에 관한 실험입니다.


RM size는 12M으로 constant하게 유지하고, 두 가지의 다른 size의 policy를 비교했습니다.


$\sqrt{KL}$에 따른 RM Score의 변화를 두 Policy에 대해 살펴보았습니다.


![14](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/14.png)


**Policy가 클수록, optimization benefit**(Initial Score ↔ 최종 Score의 Gap)**도 적었고, Overoptimization**(Gold Score의 감소)**도 적었습니다**. 


**Policy와 무관하게, overoptimization이 일어나는 KL**(peak gold score 지점)**은 유사**했습니다. (저자들은 larger policy가 더 빠르게 overoptimize 할 것이라 예측)


![15](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/15.png)


두 Policy sizes에서 **proxy ↔ gold scores gap은 균일하게 유지**됐습니다. 이 gap을 proxy RM의 exploitation 정도로 해석할 수 있습니다. **즉, policy size와 proxy RM의 exploitation 정도는 균일**함


![16](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/16.png)


 또한 다른 크기의 RM(3B)에 대해서도 진행했는데, 비슷한 결과를 얻을 수 있었습니다.


**3.5 RL vs BoN**


policy optimization 방법을 비교합니다.

	- policy training 방법
		- RL → PPO
		- best-on-n

	→ 다른 optimization 방법이 다른 overoptimization?


**RL이 BoN**에 비해 **less KL-efficient** 합니다. optimization, overoptimization 모두 RL에서 KL이 더 커졌을때 나타납니다. 직관적으로 BoN은 초기 정책 주변에서 매우 국소적으로 탐색하고, KL은 $\log(n)$에 비례해 증가합니다. 반면, RL은 이전 정책을 매 step 수정해 사용하고, KL penalty가 없는 경우 RL은 2차 함수[Figure 16]에 가깝게 증가합니다.


![17](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/17.png)


![18](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/18.png)


BoN의 Step - KL Divergence는 graph는 제시되어 있지 않네요.


이로써, **두 방법에서의 KL-efficiency가 다르기 때문에, square root KL은 정확한 metric이라고 할 수 없다는 결론**에 이릅니다. (4.1에서 더 살펴봄)


반면, **proxy RM score와 gold RM score 간의 관계성은 두 방법에서 유사**했습니다.


![19](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/19.png)


다만, 전체적으로 RL이 처음에 더 큰 proxy-gold gap이 있고, 더 높은 peak gold RM scores를 보였습니다. 따라서, 저자들은 **proxy RM scores를 진척도로 보는 것이 더 합당하다고 할 수 있다고 주장**합니다.


KL Penalty가 미치는 영향을 $D_\text{KL}$과 gold RM score의 관계를 통해 살펴보았습니다.


![20](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/20.png)


두가지를 발견합니다.

1. **KL penalty는 gold RM score가 빨리 수렴**하도록 한다.
2. **KL penalty는** $d_\text{RL}$**-gold reward frontier에는 영향을 미치지 않는다**.

![21](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/21.png)


두가지 결과와 위 그래프로 미뤄보았을 때, **KL penalty는 단지 early stopping의 효과**만 제공한다고 할 수 있습니다.


하지만, 저자는 이 결과가 **PPO의 surrogate objective의 영향일 수도 있다**고 하비다. PPO는 학습의 안정성을 위해 $D_\text{KL}(\pi_\text{old}||\pi)$**을 사용**하는데, 이 term이 $D_\text{KL}(\pi||\pi_\text{init})$에도 **간접적 영향**을 미쳐 더 느리게 증가한다고 합니다. 하지만, 저자들도 왜 이런 효과가 나타나는지는 모른다고 하네요.


---


다음과 같이 rough하게 정리할 수 있습니다.

1. **Functional Forms**
	- BoN과 RL 두 경우 모두 Gold RM Score는 functional forms으로 나타냄
	- 하지만, proxy scores는 모델링이 어려웠음
2. **RM Parameter Scaling**
	- RM 크기가 증가할수록 BoN에선 $\alpha$(증가), $\beta$(감소) 두 계수 부드럽게 변화했고, RL에선 $\alpha$는 일정, $\beta$(감소)는 부드럽게 변화
	- BoN의 경우 proxy RM에서도 유사한 경향 있었으나, KL이 클수록 reward를 underestimate하는 경향
3. **RM Data Size Scaling**
	- Data Size의 증가가 더 나은 gold scores와 적은 overoptimization
	- 다만 Data가 특정 threshold(2,000)를 넘기 전까진 RM Size 무관 효과가 없었음
	- Epoch을 늘리는 것은 효과 없고, Data Size의 증가가 중요한 영향을 미침
4. **Policy Size Scaling**
	- Policy size가 증가하면 Overoptimization 감소하고, optimization benefit도 감소함
	- Proxy ↔ Gold gap은 policy 크기와는 무관했음
5. **RL vs BoN**
	- BoN: KL 증가가 log(n) 비례로 KL-efficiency가 더 높음
	- RL: KL이 2차에 가깝게 증가하며 KL-efficiency가 낮음
	- RL이 더 높은 초기 gap과 peak gold score를 가짐
6. **KL penalty**
	- Gold score 수렴을 가속할 뿐, KL-reward 관계에는 영향 없음
	- Early stopping의 역할을 수행함

---



#### 4. Discussion


**4.1 KL as a measure of amount of optimization**


**KL**은 Section 3.2 (RM parameter scaling)에서 **일관된 스케일링 경향**을 보이며, Section 3.4 (Policy Size Scaling)에서 **Gold RM Score의 KL peak 지점도 일관성**을 보입니다. 하지만, Section 3.5 (RL vs BoN)에서 **RL과 BoN에서 step에 따른 KL이 다름**을 확인했습니다. 따라서, **KL은 optimization amount를 위한 metric으로 부적절**했습니다.  


**KL 변화가 아닌 reward에 큰 영향을 주는 어떤 요인이 있을 것이라고 추측**합니다.


**4.2 Relation to Goodhart Taxonomy**


4가지의 goodhart’s law 분류[5] 중 regressional goodhart’s law는 proxy RM이 noise가 있는 features에 의존할 때 발생합니다. 


proxy reward $\hat X$ = gold reward $X$ + independent noise $Z$ 로 나타낼 수 있습니다. optimization power가 일부의 noise choice에 사용되어, gold reward가 proxy reward 예측치보다 작아집니다.


수학적 해석을 해보면, independent한 continuous random variables $X$와 $Z$가 있을 때, $X$는 gaussian distribution, $Z$는 두 경우 중 하나를 따릅니다.


(a) Z도 gaussian distribution을 따름


(b) $Z$는 평균 $\mathbb{E}|Z|$에서 $\delta$ 이내로 제한됨 ($|Z-\mathbb{E}|Z||<\delta,\delta>0$)


이 때, model은 Gold reward를 아래와 같이 예측합니다.


![22](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/22.png)


증명은 생략하겠습니다.


직관적으로 보면, (1)은 **optimization power가 gold reward** $X$**와 noise** $\epsilon$ **사이에 분산됨**을 알 수 있습니다. 따라서, **두 reward 모두 optimization의 대상이지만, proxy reward는 noise로 인해 gold reward와 다르게 변할 수 있습니다**.


![23](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/23.png)


만약 위와 같은 관계라면 **gold reward는 proxy reward에 대해 항상 단조 증가해야 하지만, Figure 8에 의하면, 그렇지 않습니다**. 이는 noise의 분포가 가정을 위반하거나, 다른 종류의 Goodhart’s Law가 작용함을 알 수 있습니다.


![24](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/24.png)


![25](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/25.png)


![26](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/26.png)


**KL에 따른 proxy RM score의 기울기와 gold score scaling에서 선형 성분의 기울기를 나타내는** $\alpha$**는 regressional goodhart’s law의 크기**라고 할 수 있습니다.


**4.3 Implications for iterated RLHF**


RLHF는 주로 **online setup**을 사용합니다. **새 정보를 즉각 반영하기 때문에, Overoptimization을 막을 수 있습니다**. 


실험을 통해 얻은 **scaling law를 통해 iterative approach의 효과 분석을 수행**해보겠습니다. 우선, $\alpha,\beta$가 iteration 마다 constant로 유지되고, $d$는 iteration 마다 더해진다고 가정하면, k번의 반복 후 최종 gold RM score를 다음과 같이 나타낼 수 있습니다. 


$$


R_{\text{RL}}(d) = d \left( \alpha_{\text{RL}} - \beta_{\text{RL}} \log(d) + \beta_{\text{RL}} \log(k) \right)
$$


$\alpha_\text{RL}$는 Goodhart’s Law의 효과를 나타내고, $\beta_\text{RL}$와 $k$는 iterative approach가 reward model의 최종 성능에 미치는 효과를 나타낸다고 할 수 잇습니다.


식을 해석해보면, $\alpha_\text{RL}$**은 반복 학습에도 불구하고 변하지 않습니다**. 즉, iterative approach 만으로는 Goodhart’s Law를 해결할 수 없다는 뜻이겠죠. 하지만, **iterative approach는 최종 Gold RM Score를 증가**시킵니다. 그리고 해당 증가량은 $β_\text{RL}\cdot d\cdot\log(k)$에 비례합니다. 


하지만, 이 가정은 $k$가 너무 커지거나, $d/k$가 작아지는 경우 작동하지 않을 수 있습니다. 따라서, 이에 대한 추가 검증과 연구가 필요합니다.


**4.4 Policy size independence**


**overoptimization의 정도는 Policy size와 무관하게 동일**했습니다. 즉, 더 큰 policy가 더 큰 optimization power를 발휘하거나 학습 속도가 더 빠르다고 할 수 없었습니다. 다만, 초기에는 더 큰 policy가 gold score에서 높은 성능으로 시작했습니다.


![27](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/27.png)


![28](/assets/img/2024-11-10-Scaling-Laws-for-Reward-Model-Overoptimization.md/28.png)


한가지 가능한 가설은 RLHF가 initial policy의 **prior**에 대한 bayesian inference로 해석하는 방법입니다. 즉, policy size가 커질수록, human demonstrations 분포를 **더 정확히 모델링**할 수 있지만, **최적화의 효율 자체엔 영향을 미치지 못한다**는 것입니다. prior로써의 영향뿐, 학습 과정엔 영향을 못 미친다는 것 같네요.


**4.5 Limitations and Future Work**


**outer alignment**


추가적인 Overoptimization의 영향이 있을 수도 있습니다. 본 논문에선 reward model과 gold labels 간의 misalignment만을 다뤘고, ground truth label과 실제 인간의 의도 간의 misalignment는 반영하지 못했습니다. inner alignment만을 다뤘지, outer alignment에 대한 언급은 빠졌습니다. 


**Validating these results on other environments and experimental setups**


InstructGPT의 setting을 사용했기 때문에, 다른 setting에서도의 검증을 통한 generalization이 필요합니다.


**Validating the synthetic setting**


proxy RM과 gold RM의 scale이 비슷해질수록, overoptimization은 더 underestimate 될 수 있습니다. 따라서, 실제 환경에서의 검증이 필요합니다.


**Investigating methods for making RMs more robust to optimization**


Reward Model은 optimization에 취약합니다. 따라서, 이를 더 체계적으로 강화해 robust하게 하는 연구가 필요합니다.


**Exploring other forms of optimization and categorizing their differences**


해당 연구는 BoN, RL optimization만 다뤘습니다. 따라서, 다른 optimization methods에 대한 적용도 필요합니다. ex. GeDi, Decision Transformers, beam search, other RL …


**Better understanding the functional form of proxy RM scores**


proxy RM scores의 예측은 어려웠습니다. proxy RM scores를 위한 더 나은 functional form modeling 연구가 필요합니다.


**Exploring adversarial Goodhart empirically**


해당 연구는 Adversarial Goodhart 효과를 다룰 만큼 강력한 시스템이 아닙니다. Adversarial Goodhart와 관련된 현상 및 phase change 연구가 필요합니다.


**Exploring scaling with policy size in more detail**


policy size를 좀 더 다변화한 실험 필요합니다.


**Exploring multi-iteration RLHF**


Section 4.3에서의 iterated RLHF에서의 functional form 가정은 깨질 수 있습니다. 



#### 5. Related Works

- Goodhart’s Law
- RM Overoptimization / Specification Gaming
- Overfitting / Goodhart’s Law
- Adversarial Attacks / Robustness
- Scaling Laws
- RLHF
- AI Alignment


#### 6. Conclusion


본 논문은 **RLHF에서의 overoptimization**(misalignment) 문제를 다루고 있습니다.


$d$**와 gold RM score의 관계를 functional form**으로 나타낼 수 있었고, 다양한 요인들의 효과를 확인할 수 있었습니다. **proxy RM score와 gold RM score의 관계성**을 통해 **Regressional Goodhart’s Law 이외의 다른 요인이 관여함**을 알 수 있었습니다. 


비록, **KL**을 이용해 gold RM score의 식을 나타냈지만, **optimization method 마다 그 양상이 다르기 때문에 적합한 지표가 아니었습니다**. 


**Iterative RLHF**은 **Overoptimization을 완화**하는 효과가 있을 수 있지만, **Goodhart’s Law를 근본적으로 해결하진 못합니다**.


**RM size와 RM Data size**가 gold score에 **중요한 영향**을 미친 반면에 **policy size는 큰 영향을 미치지 못했습니다**. Policy Size는 bayesian 관점에서 **prior**에 그치기 때문일 수 있었습니다.  


**이를 해결하기 위해 다양한 부분에서의 추가 연구가 필요한 바입니다.**



#### 7. Reference


[1] Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (Bai et al, 2022)


[2] Training language models to follow instructions with human feedback (Ouyang et al, 2022)


[3] Proximal policy optimization algorithms (Schulman et al, 2017)


[4] WebGPT: Browser-assisted question-answering with human feedback (Nakano et al, 2022)


[5] Categorizing variants of goodhart’s law (Manheim & Garrabrant, 2018)


[4] WebGPT: Browser-assisted question-answering with human feedback (Nakano et al, 2022)


[5] Categorizing variants of goodhart’s law (Manheim & Garrabrant, 2018)

