---
layout: single
date: 2024-11-05
title: "David Silver - RL Lecture 3 (Planning by Dynamic Programming)"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
author_profile: false
---

Bellman equation을 통해 recursive하게 value function을 나타내는 데에 성공했다. 하지만, optimal value function을 얻기 위해선, max term 해결을 위해, closed-form 방법이 아닌 Iterative method가 필요했었다.어떻게 푸는지 확인해 보겠습니다.



### **Planning by Dynamic Programming**


Dynamic Programming은 큰 문제들을 sub-problem으로 나누고, sub-problem들의 solutions를 선형적으로 활용해 큰 문제를 푸는 방법입니다. 두가지 특성을 만족하는 문제에 적용 가능합니다.


![0](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/0.png)

1. **optimal substructure** (principle of optimality)

	**최적의 solution을 여러개의 sub-problems로 나눌 수 있어야** 한다.

2. **overlapping subproblems**

	**sub-problems가 반복**되어야 하고 **이전의 solution을 cache처럼 재사용**할 수 있어야 한다.


두 가지 특성을 만족하는 대표적인 예시가 **Markov Decision Process**입니다. 따라서, MDP의 Planning에 DP를 적용해보자는 시도가 있습니다.


**Planning 이란?**


![1](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/1.png)


Planning이란 **MDP를** **model을 아는 경우**(**MDP에 대한 full knowledge 존재**)**에 문제를 푸는 것**입니다. MDP 모델을 안다는 것은 즉, $(S,A,P,R,\gamma)$ 등의 구성 요소들을 아는 것이라고 할 수 있습니다. 이 때, 문제는 2가지 종류가 있는데, **Prediction**과 **Control** 문제입니다.

1. **Prediction**

	주어진 policy를 따랐을 때의 value function을 찾는 문제입니다. 대표적으로 game에서 특정 policy를 따랐을 때의 승률/점수를 구하는 문제 등이 해당합니다.

	- input

		1) MDP $\langle\mathcal{S,A,P,R,\gamma}\rangle$ + policy $\pi$


		2) MRP $\langle\mathcal{S,P^\pi,R^\pi,\gamma}\rangle$

	- output

		value function $v_\pi$

2. **Control**

	최적의 policy를 찾는 문제입니다. 자율주행 자동차의 운전 policy, 로봇의 제어 policy 등이 있겠습니다.

	- Input

		MDP $\langle\mathcal{S,A,P,R,\gamma}\rangle$

	- Output

		1) optimal value function $v_*$


		2) optimal policy $\pi_*$



### **Synchronous DP Algorithms**



#### Policy Iteration


![2](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/2.png)


Policy Evaluation과 Improvement를 반복적으로 수행해나가면서, optimal policy를 찾는 방법입니다. 


**Policy evaluation**


![3](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/3.png)


주어진 Policy $\pi$를 평가하는 문제입니다. $\pi$를 따라갔을 때, 얼마만큼의 return, 즉 value function을 찾는 문제입니다. bellman expectation을 통해 recursive하게 value function을 정의했는데, 해당 수식을 바탕으로 iterative method를 통해 구할 수 있습니다. 


![4](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/4.png)


random한 $v_1$으로 initialize 한 후, 계속해서 $v_2,v_3,...,v_\pi$를 찾는 방법입니다. 방법으론 **synchronous backups 방법**을 사용합니다. 모든 states에 대한 value table을 두어, 매 Iteration 마다 해당 table을 update 합니다. $v_k(s') \rightarrow v_{k+1}(s)$의 update를 통해 수렴할 때 까지 반복합니다.


**Policy improvement**


![5](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/5.png)


policy evaluation을 통해 $v_\pi$를 구했습니다. optimal policy는 해당 value function에 대해 greedy하게, 즉 가장 높은 value를 갖는 방법으로 policy를 수정하면 됩니다.  ($\pi'=greedy(v_\pi)$)


![6](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/6.png)


![7](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/7.png)


![8](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/8.png)


이렇게 improvement를 반복하다가, 더이상 improvement가 이뤄지지 않으면 수렴한 것이라고 볼 수 있습니다.



#### **Modified Policy Iteration**


하지만, 꼭 $v_\pi$가 수렴할 때 까지 policy evalution을 해야 할까요? 만약, 수렴이 아닌 다른 다른 stopping 조건이 있다거나, 매 iteration policy를 update하는 방법이 없을까요? (ex. $\epsilon$-convergence, stop after k iteration)


![9](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/9.png)


이처럼, policy iteration은 꼭 고정된 방법을 통해서 수행하지 않아도 됩니다.



#### **Principle of Optimality**


![10](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/10.png)


각 상태에서의 최적의 선택을 통해 최적의 policy를 구성할 수 있다는 내용입니다. 2가지 components로 분리 가능합니다. 

1. optimal first action $A_*$
2. optimal policy from successor state $S'$

이를 value iteration algorithm에 적용할 수 있습니다.



#### Value Iteration


**Deterministic Value Iteration**


![11](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/11.png)


policy improvement 방법에선 policy evaluation으로 얻은 value function을 통해, 더 나은 policy $\pi'$을 선택했다면, value iteration은 value function 자체를 직접 optimize 해서 optimal policy를 찾는 algorithm입니다. 즉, 명확하게 evaluation, improvement로 나누지 않고, 반복적으로 value function 자체를 갱신해 최적의 value function에 수렴하도록 하는 방식입니다.


![12](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/12.png)


principle of optimality에 따라, 각 상태에서의 최적의 value function을 선택하고, 그 다음 상태에서도 최적의 policy를 따를 때, 최적의 결과를 보장합니다. 이렇게 optimal policy를 찾는 방법입니다.


![13](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/13.png)


간략하게 정리하면 아래와 같습니다.


![14](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/14.png)



### Asynchronous Dynamic Programming


지금까지 synchronous 한 방법을 알아보았습니다. asynchronous 한 방법도 존재합니다. 각 상태를 개별적으로 갱신하기 때문에, computation 을 줄일 수 있고, convergence가 보장된다는 성질이 있습니다. 하지만, 여러 states를 골고루 sampling 되어야 합니다.


이는 상태 공간이 매우 크지만, agent의 action이 sparse하게 일어나는 경우 유용합니다. 종류로는 In-place DP, Prioritised sweeping, Real-time DP 등이 있습니다. 강의에선 각 방법에 대해 간단하게만 정리합니다.


**In-place DP**


기존엔 $v_{old},v_{new}$ 두가지 table을 모두 memory에 갖고 있지만, 오직 하나의 table $v$만 갖도록 하는 방법입니다.


**Prioritised sweeping**


![15](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/15.png)


Bellman error가 큰 state 위주로 backup 수행하는 방법입니다.


**Real-time DP**


![16](/assets/img/2024-11-05-David-Silver---RL-Lecture-3-(Planning-by-Dynamic-Programming).md/16.png)


Agent가 방문할 가능성이 높은 state 들을 우선적으로 backup하는 방식입니다.



#### Full-width Backups & Sample Backups


**Full-width Backups**


지금까지의 DP 방법은 모든 테이블을 update하는 방법이었습니다. curse of dimensionality가 발생하는, 큰 문제에선 불가능한 방법입니다. 


**Sample backups**


반면, sample backup 방식은 전체 백업 대신, agent가 실제로 action을 수행하면서 얻은 sample을 이용하는 방식입니다. 해당 방법을 사용하면, environment에 대한 정보가 없는 model-free 문제에도 대처할 수 있고, curse of dimensionality도 발생하지 않으며, backup cost도 크지 않습니다. 


reward function과 transition probability가 아닌, sample rewards와 sample transitions을 이용합니다. ($\langle{S,A,R,S'}\rangle$)


Sample backup 방식의 알고리즘으론 Q-Learning, SARSA 등이 있습니다. 대부분의 RL 문제가 model-free 문제인 만큼 중요하다고 할 수 있습니다.

