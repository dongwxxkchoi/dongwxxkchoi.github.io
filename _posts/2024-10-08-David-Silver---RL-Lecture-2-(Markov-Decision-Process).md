---
layout: single
date: 2024-10-08
title: "David Silver - RL Lecture 2 (Markov Decision Process)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---


### **Introduction**


**Reinforcement Learning**은 **agent가 environment와 상호작용하며, 최적의 policy를 학습하는 과정**입니다. Environment는 각 state와 해당 state에서 취할 수 있는 action, 각 action을 할 probability, 그 때의 reward 등으로 구성되어 있습니다. 따라서, **RL의 목표는 agent가 주어진 state에서 어떤 action을 할 지를 결정해 reward의 총합을 최대화하는 것**이라고 할 수 있습니다.


이를 알기 위해, 먼저 **Environment를 정의하는 Markov Decision Process를 알아야 합니다**. Markov Decision Process를 알기위해, Markov → Markov Process → Markov Reward Process → Markov Decision Process의 순서로 개념과 그 구성요소들을 알아보겠습니다.



#### Markov Property


![0](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/0.png)


Markov Property는 미래 state는 현재 state에만 영향을 받는다는 성질입니다. 


![1](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/1.png)


따라서, 특정 state로 이동할 확률일 state transition probability는 다음과 같이 정의됩니다. 


![2](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/2.png)


State transition matrix $\mathcal{P}$는 state transition probabilities를 matrix의 형태로 정리한 matrix입니다. 특정 state $i$에서 다른 states로 이동할 수 있는 확률의 총합은 1이기 때문에, $\sum_{j=1}^n\mathcal{P_{ij}}=1$인 성질을 갖고 있습니다.



#### Markov Process


Markov Property를 적용한 probabilistic state transition model입니다. random states $S_1,S_2,...$의 sequence가 markov property를 만족하는 memoryless random process입니다. 


![3](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/3.png)


Markov Process는 set of states $\mathcal{S}$와 state transition probability matrix $\mathcal{P}$로 구성됩니다. ($\langle\mathcal{S,P}\rangle$)

- $\mathcal{S}$: set of states
- $\mathcal{P}$: state transition probability matrix

**Example - Markov Process**


![4](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/4.png)


![5](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/5.png)


![6](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/6.png)


왼쪽의 graph가 markov process를 시각적으로 정의한 것이고, 오른쪽은 해당 process에서 sampling된 sequence입니다.


![7](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/7.png)


state transition matrix로 표현하면 위 그림과 같습니다. 



#### **Markov Reward Process**


Markov Reward Process는 Markov Process를 확장하여 각 state에서의 reward 까지 고려하는 system 입니다. ($\langle\mathcal{S,P,R},\gamma\rangle$) 여기서 조금 헷갈리는 것이 Reward와 Reward Function의 구분입니다.


![8](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/8.png)


reward $R$은 각 state마다 할당된 reward value를 의미합니다. 


reward function $\mathcal{R}$은 현재 state에서 다음 time step에 받을 reward의 기댓값을 의미합니다. 따라서, state $s$에서의 reward function을 나타내어 보면, $\mathcal{R}_s=\mathbb{E}[R_{t+1}|S_t=s]$로 나타낼 수 있습니다. RL에선 agent가 특정 state에서 평균적으로 얼마나의 보상을 받는 지, 즉 특정 state가 얼마나 좋은 지를 예측하기 위해 사용됩니다.


![9](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/9.png)


이들을 통해 return $G_t$를 정의합니다. return은 시간에 대해 표현되는 total discounted reward라고 할 수 있습니다. 미래에 받을 reward들에 discount factor $\gamma$를 곱해 더하는 방법을 사용합니다.


![10](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/10.png)


discount factor는 미래의 reward를 현재 가치로 조정하기 위해 사용되며, 이외에도 사용되는 다양한 이유들이 존재합니다. (영상에서 언급된 가장 큰 이유는 **수학적 편의성**이라고 하네요.)


1) Mathematical Convenience (수렴성과 관련)


2) Infinite Returns 방지 (agent가 같은 상태를 무한히 반복하는 것)


3) 미래에 대한 불확실성 반영


4) 즉각적 보상 선호


5) 재정적 관점 이자 효과 (금융 문제와 관련 있어 보임)


다만, 모든 episode가 종료되는 경우엔 discount factor를 곱하지 않는 것도 가능하다고 합니다.


이 return을 이용하면, 각 state의 가치를 나타내는 value function을 정의할 수 있습니다.


![11](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/11.png)


특정 state s에서, 앞으로 받을 return들의 기댓값을 의미합니다. 이 value function을 정의하기 위해, bellman equation이라는 방법을 사용합니다. 


![12](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/12.png)


Return을 다음 time step에 받는 즉각적인 reward $R_{t+1}$과 그 이후에 받을 rewards sequence로 구분합니다. rewards sequence $\gamma(R_{t+2}+\gamma R_{t+3}+...)=\gamma G_{t+1}$로 나타낼 수 있고 정의에 의해 $v(S_{t+1})$로 변환할 수 있습니다.


![13](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/13.png)


이렇게 bellman equation을 통해 현재 상태의 value function을 1) 즉각적 reward와 2) 다음 상태의 value function으로 변환할 수 있습니다.


![14](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/14.png)


matrices의 형태로 표현하면 다음과 같습니다. 


![15](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/15.png)


![16](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/16.png)


이렇게 linear equation 형태로 표현한 후 계산할 수 있습니다..! 


![17](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/17.png)


물론 해당 방법이 만능인 것은 아닙니다. **계산복잡도가 무려** $O(n^3)$이기 때문에, 사실상 불가능하다고 할 수 있고, **small MRPs에만 적용 가능**한 방법입니다. 따라서 대부분의 경우인 large MRPs의 경우는 closed form이 아닌 **Iterative methods를 통해 해결**합니다.


**Example - MRPs**


![18](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/18.png)


![19](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/19.png)


![20](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/20.png)


![21](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/21.png)


Markov process의 예시에서, reward가 추가된 모습을 볼 수 있습니다. 그리고, MRPs에서 sampling된 sequence는 reward를 계산할 수 있습니다.



#### **Markov Decision Process**


Markov Decision Process는 Markov Reward Process에 decision 요소를 추가한 모델입니다. ($\langle\mathcal{S,A, P,R},\gamma\rangle$) MDPs에서 agent는 특정 state에서 action을 통해 다른 state로 이동합니다. 이 때, action에 대한 transition은 deterministic하지 않고, probabilistic한 것이 특징입니다.


![22](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/22.png)


이 때, action에 대한 transition은 deterministic하지 않고, probabilistic한 것이 특징입니다. 따라서, transition probability는 아래와 같이 정의할 수 있습니다.


$$
\mathcal{P^a_{ss'}}=\mathbb{P}[S_{t+1}=s'|s_t=s,A_t=a]
$$


이 때, Agent가 action을 수행하는 전략을 policy라고 합니다.


![23](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/23.png)


Agent의 action $A_t\sim\pi(\cdot|S_t)$으로 부터 sampling 합니다. 


즉, 완성된 MDPs 즉 environment는 $\mathcal{M=\langle S,A,P,R,\gamma\rangle}$, 그리고 policy $\pi$로 구성됩니다. 그때, Probability와 Reward는 아래와 같이 정의할 수 있습니다.


![24](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/24.png)


MDPs에선 value function을 두가지 정의할 수 있습니다. 기존 state-value function에 더해, (state, action) pairs에 대한 action-value function도 정의할 수 있습니다.


**1) state-value function**


![25](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/25.png)


특정 state에서 policy $\pi$를 따랐을 때의 value function을 의미합니다.


**2) action-value function**


![26](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/26.png)


특정 state s에서 policy $\pi$를 따라 특정 action a를 수행했을 때의 value function을 의미합니다.


**Bellman equation 적용**


이렇게 정의한 value functions를 bellman equation을 통해, 구할 수 있는 term으로 변화시킬 수 있습니다.


![27](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/27.png)


![28](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/28.png)


이렇게 두 term을 정의하였습니다. (계산과정 생략)


사실 state-value function과 action-value function은 밀접하게 연관되어 있기 때문에, 약간의 변환을 통해 하나의 variable로 변환할 수 있습니다.


![29](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/29.png)


위 식처럼 state-value function은 특정 state에서 $\pi(a|s)$(policy에 따라 action을 택할 확률)과 $q_\pi(s,a)$(그때의 action-value function)의 곱의 합으로 나타낼 수 있습니다. 


action-value function은 (s, a)에 대한 reward와 그 후 transition probability와 해당 확률에 따른 states에서의 value function의 곱의 합으로 나타낼 수 있습니다.


이런 식으로, 단계를 반복하다 보면, 하나의 variable로 정리할 수 있습니다.


![30](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/30.png)


value function에 대해 정리하게 되면,


$v_\pi=\mathcal{R}^\pi + \gamma\mathcal{P}^\pi v_\pi$


$v_\pi=(I-\gamma\mathcal{P}^\pi )^{-1}\mathcal{R}^\pi$


의 방법으로 closed form 방식으로 구할 수 있게 됩니다.


**Example - MDPs**


![31](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/31.png)


![32](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/32.png)


![33](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/33.png)



#### **Optimal value function & Optimal policy**


주어진 상태 s 또는 (s,a)에서의 value function을 가장 높게 만드는 policy를 선택했을 때의 value function output 그 자체입니다. 


![34](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/34.png)


optimal value function은 주어진 MDP에서의 best possible performance를 나타냅니다. 따라서, 우리가 optimal value function을 안다면 그 MDP는 solved 라고 할 수 있습니다.


그렇다면 optimal value function을 유도하는 optimal policy는 어떻게 구할수 있을까요? 구하기에 앞서 어떤 것이 좋은 policy 인지 기준이 필요합니다.


policy에 대한 partial ordering은 아래와 같이 정의할 수 있습니다.


![35](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/35.png)


![36](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/36.png)


따라서, 제일 simple 하게 생각해보면, optimal policy는 각 (state, action) pair가 주어졌을 때, value function을 maximize하는 action을 수행하는 방법일 겁니다.


![37](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/37.png)


 이렇게 greedy하게 수행하는 방법을 생각해볼 수 있겠습니다.



#### Bellman Optimality Equation


위에서의 value function을 구하는 방법으로, optimal한 value function과 policy를 구할 수 있습니다.


![38](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/38.png)


![39](/assets/img/2024-10-08-David-Silver---RL-Lecture-2-(Markov-Decision-Process).md/39.png)


위와 동일한 방법으로 value function을 정의할 수 있습니다. 다만, max term이 포함되어 있기 때문에, closed form을 통해 해결할 순 없어, 다른 solution이 필요합니다.


ex. value iteration, policy iteration, q-learning, SARSA 등


따라서, optimal한 value function과 policy를 찾기 위해, 여러 chapter에 걸쳐 방법들을 배울 것입니다!

