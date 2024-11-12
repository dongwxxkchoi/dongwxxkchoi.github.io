---
layout: single
date: 2024-10-17
title: "David Silver - RL Lecture 4 (Model-free prediction)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

지금까지, RL 문제를 정의하는 Markov Decision Process가 뭔지, 그리고 Dynamic Programming 방식을 사용하는 Policy Iteration, Value Iteration 방법에 대해 알아 보았습니다. Lecture 3의 후반부에, full-width backup 방법 대신 sample을 이용하는 sample backup 방법에 대해 잠깐 맛보기를 했습니다. Lecture 4에선 sample backup 방법을 이용해, environments를 모르는 Model-free 환경에서 value function을 estimate 하는 방법을 배울 것입니다.



### Model-free prediction


**주어진 environment (Transition probability / Reward function)를 알 수 없는 상태**에서, **value function을 어떻게 estimate** 할 것인가에 관한 문제입니다. 해당 문제를 푸는 대표적인 방법으로 **Monte-Carlo RL**과 **Temporal Difference RL**이 있습니다.



#### **Monte-Carlo Reinforcement Learning**


![0](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/0.png)


environment를 모르기 때문에, agent가 policy를 따라 직접 한 episodes를 탐색하고, 그 때 얻은 data를 활용하는 매우 empirical한 방법입니다. 실제 sample 들을 이용하기 때문에, model-free 한 상황에 적용할 수 있습니다.


MC의 가장 큰 특징은 complete episodes로부터 학습한다는 점입니다. 따라서, 미래의 예측된 값으로 현재의 예측을 업데이트하는 bootstrapping이 일어나지 않고, 실제 관찰한 값 만을 사용합니다.


방문했던 state의 그때그때의 return을 기록해두었다가, 해당 값들의 mean을 value function으로 업데이트 합니다. 정리하면 아래와 같습니다.


![1](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/1.png)


goal: learn $v_\pi$ from **episodes** of **experience under policy** $\pi$ ($S_1,A_1,R_2,...,S_k\sim\pi$)


return: $G_t=R_{t+1}+\gamma R_{t+2}+...+\gamma^{T-1}R_T$


value function: $v_\pi(s)=\mathbb{E}_\pi[G_t|S_t=s]$


mean을 구하기 위해선 몇 번 방문했는지를 나타내는 counter와 total return variable이 필요합니다. 해당 정책을 어떻게 할지에 따라 두 type으로 분류할 수 있습니다.


**First-Visit Monte-Carlo**


![2](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/2.png)


first-visit의 경우는 **한 episode에서 첫 방문시에만 counter와 total return을 증가**시킵니다. 방문횟수가 증가할수록, 큰 수의 법칙에 의해 $V(s)$는 $v_\pi(s)$로 수렴한다고 합니다.


**Every-Visit Monte-Carlo**


![3](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/3.png)


반면 every-visit의 경우는 **한 episode에서 모든 방문 시 counter와 total return을 증가**시킵니다.


단, 타입에 상관 없이 Monte-carlo 방법은 모든 state의 균일하게 방문한다는 가정이 있어야 가능합니다.


**Example - Blackjack**


![4](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/4.png)


![5](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/5.png)



#### Temporal Difference Reinforcement Learning


![6](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/6.png)


![7](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/7.png)


Monte Carlo 방법과는 다르게, episode가 끝나기 전 매 step 마다 value function을 update하는 방법입니다. 역시 model-free 문제에 적용 가능하고, MC와는 다르게 bootstrapping을 이용한다는 특징이 있습니다.


TD를 이용하면, $v_\pi$를 실시간으로 개선할 수 있습니다. (online) 따라서, 이를 정리해보면 아래와 같습니다.


![8](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/8.png)


Goal: learn $v_\pi $ online from experience under policy $\pi$


Return: $G_t=R_{t+1}+\gamma V(S_{t+1})$


TD에서의 return은 estimated value를 사용합니다. 그 estimated value를 몇 단계 받을지에 따라 TD(n)으로 분류할 수 있습니다. 


TD(0)을 예시로 들어 보겠습니다. TD(0)에서의 estimated return인 $R_{t+1}+\gamma V(S_{t+1})$을 TD target, value function update의 term $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$을 TD error로 정의합니다.


이 뜻은, 한 step 더 나아가서 측정한 temporal한 difference을 통해 현재의 state를 update하는 것이라고 할 수 있습니다. 즉, 다음 step에 대한 즉각적인 보상과 자체 예측한 보상값을 조합하는 시도입니다.


**Example - TD(0)**


![9](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/9.png)


![10](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/10.png)



#### **Bias / Variance Trade-off**


![11](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/11.png)


Monte Carlo에서의 Return $G_t$는 실제 episode를 돌면서 수집한 sample로부터 구하기 때문에, $v_\pi(S_t)$의 unbiased estimate입니다. 


True TD target인 $R_{t+1}+\gamma v_\pi(S_{t+1})$ 역시 다음 상태에서의 실제 value function을 사용하기 때문에, $v_\pi(S_t)$의 unbiased estimation입니다. 하지만, 실제 환경에선 true value function을 알 수 없는 경우가 많습니다.


따라서, 실제 사용되는 TD target은 state value function $V(S_{t+1})$을 사용합니다. TD target $R_{t+1}+\gamma V(S_{t+1})$은 $v_\pi(S_t)$의 biased estimation이라고 할 수 있습니다. 


variance의 경우 return은 많은 random한 action, transition, reward에 의존하므로 크다고 할 수 있으나, TD target은 하나의 sample에 의존하므로 variance가 작다고 할 수 있습니다.



#### MC vs TD 정리


![12](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/12.png)

1. **MC와 TD의 장단점을 비교**

**MC 장점**

- bias가 없습니다.
- 수렴성이 좋습니다. (deep learning과 같은 function approximation 방법에서도)
- initial value에 sensitive하지 않습니다.
- simple한 방법입니다.

**MC 단점**

- 전체 episode 단위의 value function을 추산하다 보니, 분산이 높습니다.

**TD 장점**

- temporal difference를 이용하기 때문에 variance가 낮습니다.
- MC에 비해 효과적입니다.
- online learning이 가능합니다.
- TD(0)를 통해 $v_\pi(s)$에 수렴합니다. (하지만, function approximation 방법 사용 시, 수렴 보장이 안됩니다.)

**TD 단점**

- initial value에 sensitive 합니다.
- bias가 있습니다.
1. **MC와 TD figure 비교**
- Monte-Carlo Backup

	![13](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/13.png)

- TD Backup

	![14](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/14.png)

- DP backup

	![15](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/15.png)

- Comparison

	![16](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/16.png)



#### **Batch MC and TD**


![17](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/17.png)


MC와 TD는 일반적으로 경험이 무한대로 수렴할 때 true value function $v_\pi(s)$로 수렴하는 것으로 알려져 있습니다. 그러나 유한한 batch 만으로 true value function을 근사해야 할 땐, 다음과 같은 방식으로 접근할 수 있습니다.


K개의 episode sample이 존재한다면, K개의 episode 에서 반복적으로 episode를 sampling 해서 학습에 사용합니다. 무한한 experience가 아니기 때문에, 완전한 수렴은 아니지만 어느정도 근사적인 결과를 얻을 수 있다고 합니다.



#### n-Step Prediction


TD의 변형된 방법으로, n-step 만큼 sampling을 진행해 return을 얻고, 그 이후의 값을 estimate 하는 방법입니다. TD(n)이라고 할 수 있습니다.


n-step return은 아래와 같이 정의됩니다. 


![18](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/18.png)


따라서 n에 따라, n=1이면 TD(0), n=$\infty$면, MC라고 할 수 있습니다.


![19](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/19.png)


이때, 여러 n-step returns를 averaging 하는 것도 가능합니다.


![20](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/20.png)



#### **TD(**$\lambda$**)**


![21](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/21.png)


모든 $\lambda$ - Return $G_t^\lambda$를 조합해 가중 평균을 구하는 방식입니다. ($G_t^\lambda=(1-\lambda)\sum^\infty_{n=1}\lambda^{n-1}G_t^{(n)}$)


TD($\lambda$)를 구하려면 여러 timestep에서의 return을 알아야 하는데, 구하는 방법으로 forward / backward view가 있습니다.


cf) TD($\lambda$) Weighting Function


![22](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/22.png)


Geometric mean을 쓰는 이유는 memory less하게 쓸 수 있으므로, computational 효과적이기 때문이라고 합니다. (TD(0)와 같은 비용으로 TD($\lambda$) 계산이 가능하다고 합니다.)


**Forward-view TD(**$\lambda$**)**


![23](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/23.png)


$s_t$에서 부터 모든 미래의 return을 가중 평균으로 계산해 value function을 업데이트 하는 방법입니다. 미래의 모든 정보를 알아야 한다는 점 때문에, episode 단위로 수행하는 경우가 많아, 실시간엔 부적합한 방법이라 할 수 있습니다.


**Backward-view TD(**$\lambda$**)**


![24](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/24.png)


Backward-view TD($\lambda$)는 과거에 대한 정보를 추적해, 실시간으로 업데이트를 수행할 수 있는 방법입니다. 과거의 states에 eligibility trace를 설정해, 최근에 방문했거나, 더 자주 있었던 state에는 가중치를 더 많이 부여하는 heuristic한 방법을 사용합니다.


![25](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/25.png)


Elibility Trace는 다음과 같이 정의합니다. $
E_t(s) = \lambda \gamma E_{t-1}(s) + \mathbf{1}(s = S_t)$


Backward-view를 사용하면, 매 step 마다 미래를 예측하는 것이 아닌, 과거의 step들을 참고해 가져오기 때문에 실시간 업데이트가 가능하기 때문에 주로 사용하는 방법이라고 합니다.


따라서, 이 TD($\lambda$)의 $\lambda$를 0으로 설정하면, TD(0)와 같고, $\lambda$를 1로 설정하면, MC와 같다고 합니다.


**총 정리**


![26](/assets/img/2024-10-17-David-Silver---RL-Lecture-4-(Model-free-prediction).md/26.png)

