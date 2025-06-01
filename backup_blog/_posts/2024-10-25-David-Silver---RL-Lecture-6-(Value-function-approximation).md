---
layout: single
date: 2024-10-25
title: "David Silver - RL Lecture 6 (Value function approximation)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

현실의 **대규모의 문제에 table lookup 방법을 적용하는 것은 불가능**합니다. 최근의 real-world problems(ex. Computer Go: $10^{170}$)의 경우 state의 크기가 굉장히 크다고 할 수 있습니다. lookup table 방식을 사용하기 위해선, 모든 state $s\in\mathcal{S}$ 당 $V(s)$이 존재하고, 모든 $(s,a),a\in\mathcal{A}$ 당 $Q(s,a)$이 존재하기 때문에, 필요한 table의 크기를 메모리로 마련하는 것은 불가능하다고 볼 수 있습니다. 


따라서, **neural network 등을 통해 approximate 하려는 시도가 등장**했습니다.



### Value Function Approximation


기존의 value function을 근사하는 **function approximation 방법을 사용**합니다. 

- $\hat v(s,\mathbf{w})\approx v_\pi(s)$
- $\hat q(s,a,\mathbf{w})\approx q_\pi(s,a)$

이 방법은 **1) large MDPs에 적용 가능하다는 장점** 뿐 아니라, **2) seen states로부터 unseen states를 generalize 할 수 있다는 장점**도 있습니다. learnable parameter $\mathbf{w}$는 앞에서 배웠던 MC, TD learning을 통해 update 할 수 있습니다.


![0](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/0.png)


**Function Approximator**는 parameter $\mathbf{w}$**로 구성된** **일종의 mode**l로, s**tate 혹은 (state, action) pair를 받아 value function을 내보내는 역할**을 합니다. model로는 **linear combination** 또는 **neural network** 등의 **differentiable function approximators**를 고려해볼 수 있습니다.


**RL 문제의 특징**이 **1) 모분포가 계속 바뀌면서**, **2) 각 상태들이 independent 하지 않다**는 것인데, 이런 **non-stationary, non-iid data에 적합한 training method가 필요**합니다.



#### Gradient Descent


Neural network를 통한 deep learning에 많이 사용되는 방법으로, 미분계수를 통한 최적해를 한 번에 찾기 어렵기 때문에, gradient를 이용해 local minimum을 찾으려는 시도입니다.


cost function $J(\mathbf{w})$은 $v_\pi(S)$와 $\hat v(S,\mathbf{w})$의 MSE를 통해 정의합니다.


$$
J(\mathbf{w})=\mathbb{E}_\pi[(v_\pi(S)-\hat v(S,\mathbf{w}))^2]
$$


cost function을 $\mathbf{w}$에 대해 미분하면, 다음과 같이 나타낼 수 있습니다.


$$
\Delta\mathbf{w}=\alpha(v_\pi(S)-\hat v(S,\mathbf{w}))\nabla_\mathbf{w}\hat v(S,\mathbf{w})
$$


value function을 linear combination으로 approximate한다고 가정한다면, value function의 gradient를 다음과 같이 나타낼 수 있습니다.


$$
\hat v(S,\mathbf{w})=\mathbf{x}(S)^T\mathbf{w}=\sum^n_{j=1}\mathbf{x}_j(S)\mathbf{w}_j\\ \nabla_\mathbf{w}\hat v(S,\mathbf{w})=\mathbf{x}(S),~~~\mathbf{x}(S)=\begin{pmatrix}\mathbf{x}_1(S)\\...\\\mathbf{x}_n(S)\end{pmatrix}
$$


따라서, gradient descent를 위한 update는 $\Delta\mathbf{w}=\alpha(v_\pi(S)-\hat v(S,\mathbf{w}))\mathbf{x}(S)$라고 정리할 수 있습니다.


강의에선, table lookup 방법이 크게 다르지 않다는 점도 얘기합니다. table lookup features를 다음과 같이 볼 수 있고, 각 state에 parameter vector $\mathbf{w}$가 table의 각 값으로 주어졌다고 볼 수 있는 것입니다. 


$$
\mathbf{x}^\text{table}(S)=\begin{pmatrix}\mathbf{1}(S=s_1)\\...\\\mathbf{1}(S=s_n)\end{pmatrix}
$$



#### Approximation at Incremental Prediction Algorithms


![1](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/1.png)


MC와 TD 모두에 approximation 방법을 적용할 수 있습니다. 기존 table lookup 방식으로 값을 기록하고 참고하던 것을, neural network에 input, output하는 것으로 대체할 수 있습니다. 


기존 방법들 모두 그대로 사용했을 때, 수렴 함을 확인할 수 있었습니다. **approximation을 통해서도 수렴하는 지**를 확인해야 합니다.


**Monte Carlo**


![2](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/2.png)


Monte Carlo 방식의 경우, return $G_t$를 구했습니다. $G_t$는 직접 episode를 돌면서 받은 Reward에 기반하기 때문에, **true value** $v_\pi(S_t)$**의 unbiased, noisy estimate**입니다. 특히 **전체 episode 기반 추정치를 사용하기 때문에, 수렴한다고 밝혀져 있다 합니다.**


**Temporal Difference**


![3](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/3.png)


TD의 경우는 TD-target인 $R_{t+1}+\gamma\hat v(S_{t+1},\mathbf{w})$을 구했습니다. 해당 값은 **biased sample**이라고 할 수 있고, 특히 **Linear** **TD(0)은 global optimum에 가깝게 수렴한다고 증명**돼있다 합니다. 


(MC 방법과는 다르게, TD-target은 neural network estimate을 사용합니다. 하지만, 상수로 여겨지는 것을 볼 수 있는데, 이는 time reversal 문제 때문이라고 하며, 강의에서도 조금 복잡하다 하니 일단은 넘어가도록 하겠습니다…)


![4](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/4.png)


TD($\lambda$)의 경우는 forward-view의 경우는 return $G_t^\lambda$를 구해서 하는 방식이고, backward-view는 마찬가지로 상수취급해서 update 값을 구하는 모습입니다. 



#### Control with Value Function Approximation


![5](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/5.png)


이런 approximation 방법을 **control 문제에도 적용**할 수 있습니다.


![6](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/6.png)


cost function $J(\mathbf{w})$를 true action-value function $q_\pi(S,A)$와 approximate action-value function $\hat q(S,A,\mathbf{w})$의 MSE로 정의합니다.


![7](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/7.png)


$(s,a)$ pairs을 linear combination의 feature로 사용해 $\hat q$를 나타내어, gradient descent update를 수행합니다.


![8](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/8.png)


이렇게 얻은 $\hat q$를 사용하는 방법을 **Linear SARSA**라고 합니다.


**example**


Linear SARSA를 통해 문제를 어떻게 푸는지 결과를 살펴보겠습니다.


![9](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/9.png)


Linear SARSA를 이용해 mountain car problem을 푸는 과정입니다.


문제정의> 자동차가 골짜기에 빠져 있습니다. 중력의 작용으로 액셀을 밟는 것 만으로는 Goal에 도달할 수 없습니다. 따라서, 가속도를 이용하기 위해, 앞뒤로 왔다갔다 하면서 가속도를 붙여야 한다고 합니다. state는 (차의 위치, 속도)로 정의되고, action은 (전진, 후진, 아무것도 안하기) 3가지가 가능하고, action 시마다 reward로 -1, 도달할 때 끝납니다.


이런 문제가 있을 때, Linear SARSA를 이용하면 위와 같은 과정을 거쳐 아래와 같은 최종 graph를 갖게 된다고 합니다. (그냥 보고 넘어가면 될 것 같습니다.)


![10](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/10.png)


이런 문제 풀이 결과를 통해, bootstrapping을 통한 TD 방법론들, 특히 TD($\lambda$)가 왜 중요한지를 알 수 있습니다.


![11](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/11.png)


$\lambda=0$인 경우는 TD(0), $\lambda=1$인 경우는 MC라고 했습니다. 그래프를 보면, MC 방법을 사용하는 경우는 눈에 띄게 error, cost 등이 높고, TD(0) > TD($\lambda$) 순으로 낮아지는 것을 볼 수 있습니다. 그냥 그렇구나 하고 넘어가면 될 듯 합니다.


**Counterexample (TD(0))**


![12](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/12.png)


![13](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/13.png)


TD(0)가 수렴하지 않는 예시라고 합니다. 역시 그렇구나 하고 넘어갑시다.


**Convergence of Prediction Algorithms**


지금까지 배운 방법들의 convergence를 비교하는 graph입니다.


![14](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/14.png)


On-policy와 Off-policy로 구분할 수 있고, 두 경우에 MC, TD(0), TD($\lambda$)을 적용해봤습니다. On-policy가 Off-policy에 비해 더 잘 수렴하는 것을 볼 수 있고, Table Lookup 방식보다는 Non-Linear 방식으로 갈수록 수렴이 어려웠습니다.


Silver 교수님이 만드신 Gradient TD에 대한 얘기도 하나 등장하는데요


![15](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/15.png)_prediction_


![16](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/16.png)_control_


강의에선 그냥 그렇구나 하고 넘어가니, 따로 정리하진 않겠습니다.



### Batch Methods



#### Batch Reinforcement Learning


![17](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/17.png)


지금까지 배운 방법은 agent가 online으로 획득한 sample을 통해 gradient를 구해 학습하는 방법입니다. 한 번 사용된 sample 들은 다시 사용되지 않습니다. 이 방법은 sample efficient 한 방법은 아닙니다. 


Batch reinforcement learning은 수집한 과거의 samples를 training data로써 학습에 활용해보자는 아이디어에서 시작됐습니다. 


![18](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/18.png)


$\mathcal{D}=\{\langle s_1,v_1^\pi\rangle\,\langle s_2,v_2^\pi\rangle\,...,\langle s_T,v_T^\pi\rangle\}$는 수집한 samples, experience를 의미합니다. Objective function으로 $\mathcal{D}$에 대한 MSE을 적용해 Least squares algorithm을 적용해, 최적의 $\hat v$를 찾겠다는 방법입니다.


![19](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/19.png)


![20](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/20.png)


항상 모든 $\mathcal{D}$의 samples를 사용하지 않습니다. stochastic한 방법으로 $\mathcal{D}$로 부터 samples를 sampling해서 사용하는 stochastic gradient descent 방법을 사용합니다.



#### Deep Q-Networks (DQN)


Atari Game에 RL을 적용한 것으로 유명한 DQN 알고리즘을 배워보겠습니다. Nature 지에 실릴 정도로 유명한 알고리즘입니다. 강의에선 맛보기 정도로 간단하게 설명합니다.


![21](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/21.png)


approximator로 non-linear 함수를 사용했을 때, 수렴성이 보장되지 않았었습니다. DQN은 이 점을 해결하기 위해 2가지 trick을 사용합니다. **1)** **experience replay**와 **2) fixed Q-targets**입니다.


**1) Experience replay**


**transition 정보인** $(s_t, a_t, r_{t+1}, s_{t+1})$**을 replay memory** $\mathcal{D}$에 저장합니다. $\mathcal{D}$로 부터 random하게 **mini-batch sample** $(s,a,r,s')$**을 sampling** 해서 학습을 수행합니다. (MSE로 error 정의)


**2) Fixed Q-targets**


**두 가지 parameter를 이용**합니다. **old, fixed parameters** $w^-$와 **지속적으로 update하는 parameters** $w_i$입니다.


non-linear approximator를 사용하면 특히 gradient의 방향이 쉽게 바뀌기 때문에 수렴이 어렵다고 합니다. 따라서, **일정 기간 parameter를** $w^-$**로 fix 시켜 학습에 활용하는 방법을 도입**했다고 합니다.


DQN을 Atari에 어떻게 적용했는지 알아보겠습니다.


![22](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/22.png)


approximator로 CNN을 이용합니다. 이전 4장의 frame을 input state $s$로 넣어주었고, output인 $Q(s,a)$는 18개의 수행 가능한 행동의 확률을 의미합니다. 이렇게 구한 value function에 대해 $\epsilon$-greedy improvement를 사용했습니다.


![23](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/23.png)


Replay memory와 Fixed-Q 방법의 효과를 확인할 수 있습니다.


![24](/assets/img/2024-10-25-David-Silver---RL-Lecture-6-(Value-function-approximation).md/24.png)


가운데 직선이 사람의 능력 수준을 나타내는 선입니다. 직선 기준 왼쪽은 사람의 능력을 넘어 섰다는 것을 의미합니다. DQN을 통해 대부분의 Atari games에서 사람을 능가하는 능력을 얻을 수 있었습니다.

