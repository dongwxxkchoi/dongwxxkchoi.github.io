---
layout: single
date: 2024-10-21
title: "David Silver - RL Lecture 5 (Model-free control)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

대부분의 real-world problem은 model에 대해 알 수 없는 경우가 많습니다. Model에 대한 정보가 없다는 것은, 각 state 간의 transition probability와 reward function을 알 수 없는 상황을 말합니다. 지난 Lecture를 통해, model을 모르는 상황에서, value function을 구하는 방법(Model-Free Prediction)을 배웠습니다. **Model-Free Control**은 이런 **model을 모르는 상황에서, optimal policy를 학습하는 방법**을 의미합니다.



#### Model-Free Control


Model-Free Control은 2가지 방법으로 나눌 수 있습니다.

1. **On-policy Learning**
2. **Off-policy Learning**

**On-policy Learning**이란 **target policy와 behavior policy가 같은 경우**를 의미하고, **Off-policy Learning**은 **target policy와 behavior policy가 다른 경우**를 의미합니다. 


**target policy**란 말 그대로 **우리가 학습해야 하는 target이 되는 policy**를 의미하고, **behavior policy**는 agent가 iteration 과정에서 **실제 행동을 하는 policy**를 의미합니다. 



## On-policy Learning


![0](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/0.png)


우리는 지금까지 배운 MC나 TD같은 방법들에서, agent가 직접 이동하며 얻은 sample을 사용해 value function evaluation에 사용했고, 이를 바탕으로 policy improvement를 수행했습니다. 이렇게 **improve된 policy를 통해 또 action을 sampling하고 과정을 반복**합니다. 즉, **target policy와 behavior policy가 같은 경우**라고 할 수 있습니다. 자세하게 알아보기 전, 복습을 해보겠습니다.


3장에서 **policy iteration**에 대해 잠깐 배웠었습니다. Optimal policy를 찾기 위해서  **policy evaluation과 policy improvement을 반복 수행해 policy를 개선**했습니다.


![1](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/1.png)


이 아이디어를 살려서, 4장에서 배운 **model-free policy evaluation 방법인 Monte-carlo policy evaluation을 사용**하고, **greedy policy improvement** 방법을 사용하면 되지 않을까요? 정답은 **No**입니다.


지금 우리는 model을 모르는 상황이라는 것을 유의하셔야 합니다. greedy policy improvement 방법은 각 state에서 reward를 가장 많이 받을 action을 선택하는 방법으로 policy를 수정하는 방법입니다. **state value function** $V(s)$ **만을 알고 있는 현 상황에선 불가능**합니다.


![2](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/2.png)


만약, **action-value function** $Q(s,a)$**을 아는 상황**이라면 위에서의 문제점은 해결할 수 있습니다. 단순히 $argmax_aQ(s,a)$**를 선택하면 되므로 가능**하다고 할 수 있습니다. **Policy improvement 방법엔 문제가 없을까요?** 


Greedy improvement 방법도 문제가 존재합니다.


![3](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/3.png)


예시를 보시면, 처음 left를 열었을 땐 0, 그 다음 right를 열었을 땐 1의 reward를 얻었으므로, greedy하게 선택한다면 계속 right를 고를 것입니다. 하지만, 이 방식은 best라고 보기 어려운 것이, **left에 대한 정보가 너무 부족**합니다. 최적의 policy를 구하기 위해선, **충분한 데이터를 쌓기 위한 exploration**과 **경험한 데이터를 사용하는 exploitation** 사이의 **균형을 맞춰야 합니다**. 따라서, greedy improvement를 변형한 $\epsilon$**-greedy improvement** 방법이 등장했습니다.



#### $\epsilon$-greedy improvement


![4](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/4.png)


생각할 수 있는 가장 간단한 아이디어는 **확률을 이용해 항상 greedy하게만 선택하지 않도록 하는 것**입니다. $\epsilon$-greedy improvement는 이 아이디어를 사용한 방법입니다. $\epsilon$ 이라는 term을 두어, $1-\epsilon$ **의 확률로 greedy action**, $\epsilon$**의 확률로 random action을 선택**하도록 합니다.


m개의 actions이 있을 때, greedy한 action을 제외하곤 균등하게 sampling 되도록 random actions의 확률은 $\frac{\epsilon}{m}$을 할당합니다. greedy action은 계산 상 편의를 위해, $\frac{\epsilon}{m}+1-\epsilon$을 할당합니다.


![5](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/5.png)


($\pi$와 $\pi'$가 모두 $\epsilon$-greedy라고 표기되어 있는데, 아마 slide에 오류가 있는 것 같습니다. 제가 듣고 있는 팡요랩 강의에 의하면, $\pi$는 일반적인 policy, $\pi'$는 $\epsilon$-greedy policy라고 보는 것이 맞는 것 같습니다.)


이 수식은 $\epsilon$-greedy policy를 따르는 것이 일반적인 policy를 따를 때 보다 더 좋은 value function을 얻을 수 있는 것 같습니다.


위 식은, $\epsilon$-soft policy를 따를 때보다, $\epsilon$-greedy policy를 따를 때, 더 큰 기대 value를 가질 수 있다는 증명입니다. 식은 $\epsilon$-soft policy를 통해, greedy한 행동을 제외하곤 $\epsilon$/m을 따르고, greedy한 행동은 $1-$$\epsilon$를 따른다고 해보자. 이때, greedy한 행동은 $\epsilon$-soft policy를 따랐을 땐, 1-$\epsilon$+$\epsilon$/m이고, 식을 정리하면 $\epsilon$-greedy policy를 따랐을때가 더 큰 value를 가져온다는 것을 알 수 있습니다.


⇒ 모르겠으니 나중에 다시 해보자


![6](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/6.png)


따라서, $\epsilon$-greedy policy improvement를 사용하면, 적절한 학습이 가능합니다.



#### Monte-Carlo Control


Monte-Carlo policy evaluation 방법을 통해 value function을 estimate할 때, 수렴을 보일 때 까지 episode를 반복하며 평가를 수행합니다. **수렴을 보이기 전 improvement 단계로 넘어갈 순 없을까요?**


![7](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/7.png)


즉, $Q=q_\pi$를 찾지 않고, $Q\simeq q_\pi$ 정도로 근사하는 $Q$만 찾는 방법입니다. 이 방법이 가능하기 위한 조건들이 존재하는데, 그 중 하나가 GLIE라는 조건입니다.



#### GLIE (Greedy in the Limit with Infinite Exploration)


![8](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/8.png)


GLIE property라고 부르는데, 두 가지 조건으로 구성되어 있습니다.

1. **Infinite exploration (Exploration)**

	**모든 state-action pairs는 무한히 자주 반복**되어야 한다. ($\epsilon$-greedy를 사용한다면 만족할 수 있습니다)

2. **Greedy in the limit (Exploitation)**

	policy는 결국엔 greedy policy로 수렴해야 한다. ($\epsilon$**-greedy**를 예로 들면, 충분한 탐색이 이뤄졌음에도, $\epsilon$ **만큼의 random action**을 할 수 밖에 없습니다. 두번째 조건의 만족을 위해서라면, **충분한 탐색 후엔 greedy policy로 전환되어야 합니다**)


두 조건을 만족하는 $\epsilon$-greedy policy를 설계하기 위해선, $\epsilon$**이 충분한 time step 이후엔 값이 감소해서, greedy한 action을 더 많이 선택해야 합니다**. 따라서, $\epsilon$을 time step에 dependent하게 $\frac{1}{t}$와 같은 값으로 설정하면 됩니다.


![9](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/9.png)


따라서, **GLIE property를 만족하는** $\epsilon$**-greedy의 적용을 통해, 수렴을 보이기 전 improvement를 진행할 수 있다**고 할 수 있습니다.


지금까지 Monte-Carlo 방법을 적용해 보았습니다. Temporal Difference 방법을 사용할 순 없을까요? 물론 가능합니다. **TD를 이용한 Control 방법**으로 유명한 알고리즘이 있는데, 바로 **SARSA**라는 알고리즘입니다.



#### SARSA 


![10](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/10.png)


현 step의 $Q(S,A)$의 evaluation을 위해서, 한 step 이후 즉각적으로 받은 reward $R$과 그 때의 value function의 추정치 $Q(S’, A’)$을 이용하기 때문에, 등장하는 variable들을 순서대로 나열한 **SARSA**라는 이름으로 부르게 됐습니다.


![11](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/11.png)


SARSA를 이용한 iteration을 생각해 보겠습니다. TD 방식을 사용하기 때문에, Episode 단위였던 Monte carlo와는 다르게, **훨씬 짧은 주기로 iteration이 이뤄집니다**.


![12](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/12.png)


pseudo code로 표현하면 위와 같습니다. 각 step마다 sampling과 update가 이뤄짐을 알 수 있습니다.


![13](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/13.png)


SARSA는 두가지 조건 하에 optimal action-value function으로 수렴합니다. 먼저, GLIE property는 $\epsilon$-greedy 방법을 사용함으로 만족하는 것을 알 수 있습니다. 


두번째의 Robbins-Monro sequence step-sizes는 **SARSA의 step-size** $\alpha_t$**가 수렴성을 보장하기 위해 만족해야 하는 수학적 condition**입니다. 만약, $\alpha_t=\frac{1}{t}$와 같은 값으로 설정한다면 만족합니다.


example - SARSA


![14](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/14.png)


![15](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/15.png)


SARSA를 이용해 problem을 풀었을 때의 timestep-episodes의 관계를 나타낸 그래프입니다. 초반 2000번 정도 동안은, 제대로 된 정보가 존재하지 않기 때문에, agent가 굉장히 헤매는 모습을 보이지만, 우연히 한 번 도달하고 나선 정보를 얻기 때문에, timestep이 증가할수록 한 episode에 도달하는 속도가 굉장히 빨라지는 것을 볼 수 있습니다.



#### n-step Sarsa


한 step 만 이용하는 것이 아닌, n step의 정보를 이용했던 TD(n)과 유사하게, n-step SARSA도 생각해 볼 수 있습니다.


![16](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/16.png)


n-step SARSA의 update 수식입니다. 


TD($\lambda$)의 경우 forward view와 backward view가 존재했는데, 역시 forward view SARSA($\lambda$)와 backward view SARSA($\lambda$)가 존재합니다.



#### SARSA($\lambda$)


**Forward view SARSA(**$\lambda$**)**


![17](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/17.png)


Forward view TD($\lambda$)는 현재 step의 value function update에 한 episode 전체의 reward를 사용하기 위해, 완전한 episode가 필요하다는 단점이 존재했습니다. Forward View SARSA($\lambda$)의 경우도 동일합니다.


**Backward view SARSA(**$\lambda$**)**


Backward view TD($\lambda$)는 eligibility traces를 이용해, Forward view TD($\lambda$)와는 다르게 실시간 update가 가능하다는 것이 특징이었습니다. Backward view SARSA($\lambda$)도 동일합니다.


![18](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/18.png)


![19](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/19.png)


pseudo-code로 나타낸 Backward view SARSA($\lambda$)입니다. **현재의 TD-error가 과거의 모든 state-action pair에 영향을 주기 때문에, memory 속 모든** $s\in\mathcal{S},a\in\mathcal{A}(s)$**에 대한 value function과 eligibility도 업데이트 됩니다.**


![20](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/20.png)


그림으로 이해하면, 다음과 같습니다. one-step SARSA의 경운 reward가 발생한 순간, 해당 state-action value만 update하는 반면, SARSA($\lambda$)는 전반적인 path에 대한 update가 이뤄지는 것을 알 수 있습니다. recent state 일수록 그 eligibility update가 큰 것을 볼 수 있습니다.



#### Off-Policy Learning


**Off-Policy Learning**이란, **target policy와 behavior policy**를 다르게 설정하는 방법이라고 했습니다. 따라서, target policy $\pi$와 behavior policy $\mu$의 2개의 다른 policy를 두는 것이 특징입니다. **target policy** $\pi(a|s)$**를 따랐을 때의** $v_\pi(s)$ **또는** $q_\pi(s,a)$**를 구하는 것이 목적**인데, **정작 agent의 behavior를 위한 sampling은** $\mu(a|s)$**를 따릅니다.** 선뜻 이해가 쉽진 않은데, 한번 살펴보겠습니다.


![21](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/21.png)


생각해보면, Off-policy learning은 많은 장점을 갖습니다. 


1) 본인이 아닌 **다른 agent의 행동을 참고해, target policy의 update에 참고.**


2) policy가 계속 update되어 지속적인 sampling이 필요한 on-policy 방법과는 다르게, **old policy의 활용이 가능.**


3) RL에서 exploration ↔ exploitation은 trade off 관계에 있었음. 하지만, off-policy를 사용하면 **exploratory policy를 수행하면서 optimal policy를 학습할 수 있음.**


4) 한 policy를 따르면서 **multiple policies를 학습할 수 있음.**


그렇다면, 어떻게 Off-policy Learning이 가능한지 알아보겠습니다. 



#### Importance Sampling


Off-policy Learning은 target policy와 behavior policy가 다르기 때문에, **behavior policy를 통한 경험을 target policy에 맞게 조정해 학습할 수 있어야** 합니다. 이를 위해 importance sampling 방법이 사용됩니다.


![22](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/22.png)


**importance ratio**를 사용해 조정하는데, **behavior policy가 특정 action을 선택할 확률과 target policy가 그 action을 선택할 확률의 비율로 정의**할 수 있습니다. 


$$
\text{importance ratio}=\frac{\pi(a,s)}{\mu(a,s)} 
$$


![23](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/23.png)


MC의 return에 importance ratio를 곱한 $G_t^{\pi/\mu}$로 대체해 사용이 가능합니다. 하지만, 이 방법으로 얻은 해당 term은 문제가 존재하는데, **굉장히 많은 term이 곱해지기 때문에, variance가 굉장히 큽니다.** 


![24](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/24.png)


TD에도 적용할 수 있습니다. MC와는 달리, 한 step의 정보만을 활용하기 때문에, variance가 크기 않습니다. 



#### Q-Learning


반면, importance sampling을 사용하지 않는 방법도 존재합니다. 대표적인 방법이 **Q-Learning**입니다. **behavior policy의 data를 직접적으로 target policy에 반영하지 않는 방법입니다.** 


**behavior policy를 통해선 실제 agent의 action을 결정하지만, value update 시엔 target policy를 통해서 추정이 이뤄지게 됩니다**. 즉, value update에 behavior policy는 개입하지 않습니다. 


![25](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/25.png)


그렇다면 behavior policy를 따로 두는 이유는 무엇일까요? 늘 얘기하듯, RL에선 **exploration과 exploitation의 적절한 균형이 중요**합니다. 다양한 $(s,a)$에 대한 data를 쌓는 것은 중요합니다. **Behavior policy를 두면, target policy의 update는 목적대로 진행하되, agent가 수행할 action 자체는 target policy의 영향을 받지 않고, 별도의 policy를 통해 explorate 하게 할 수 있습니다**. 


![26](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/26.png)


behavior policy와 target policy는 서로 다른 목적을 갖고 있는 만큼, improvemenet 방법도 주로 다르게 설정합니다. **behavior policy는 exploration 적인 측면이 중요하므로,** $\epsilon$**-greedy improvement를 사용**합니다. 반면, **target policy는 exploitation의 의도를 갖고 있으므로, greedy improvement를 사용**합니다. (이렇게 설정하면, GLIE property도 만족할 수 있습니다.)


![27](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/27.png)


Figure로 표현하면 위와 같습니다.


![28](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/28.png)


Pseudo code로 나타낸 것입니다.



### 정리


지금까지 배운 방법들을 정리해보겠습니다.


![29](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/29.png)


![30](/assets/img/2024-10-21-David-Silver---RL-Lecture-5-(Model-free-control).md/30.png)


이렇게 모든 가능한 결과를 고려했던 full backup 방식의 iterative policy evaluation, q-policy iteration, q-value iteration을 살펴보았고, 더 효율적인 방법인 sample back 방식의 TD learning, SARSA, Q-Learning을 차례대로 살펴 보았습니다.


이렇게 model을 모르는 상태에서 **value function을 evaluate**하고 **해당 정보를 바탕으로 policy를 improve 하는 방법**을 알아 보았습니다. 하지만 이 모든 방식들은 **look-up table을 두고 value를 update하는 방법**이었으며, 만약 **state-action pairs의 수가 증가한다면, 또는 continuous한 space라면 해당 방법은 사용할 수 없습니다**. 


따라서, 다음 lecture에선 **value function을 정확히 구하는 것이 아닌, neural network 등을 통해 approximate하는 방법들을 알아보겠습니다**.

