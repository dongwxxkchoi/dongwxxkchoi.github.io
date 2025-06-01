---
layout: single
date: 2024-11-01
title: "David Silver - RL Lecture 7 (Policy Gradient)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

Lecture 6까지 **estimate한 value function에 따라 action을 어떻게 sampling 할지의 idea를 통해 policy를 정의**했습니다. 이런 방법은 **value function의 변화에 따라 급격하게 policy가 변화할 수 있다는 단점**이 있습니다. **value function에 independent하게 policy를 직접 구할 순 없을까요?**


Policy를 직접적으로 optimize 하려면, **해당 policy가 얼마나 좋은지 평가**할 수 있어야 합니다. 또한, 해당 **Policy를 gradient descent 방식으로 optimize하기 위해, gradient를 구하는 방법**을 알아야 합니다. 그 function을 policy objective function이라고 하고, **policy를 학습하기 위해, objective function의 gradient를 구하는 방법**을 알아볼 것입니다.



## Policy Objective Functions


Policy objective functions를 정의하는 방법은 3가지가 존재합니다. 


![0](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/0.png)


**1) episodic environments의 경우**


즉, **initial state와 terminal state가 명확하게 구분되는 환경을 의미**합니다. 이 때는, **start value를 사용**할 수 있습니다. 격투 게임에서, 두 캐릭터가 서로를 마주보고 특정 거리에 고정되어 시작하는 것을 예시로 들 수 있습니다. start state $s_1$은 고정된 state이거나, 또는 고정된 distribution(ex. 50%확률로 왼쪽에 서기, 50%확률로 오른쪽에 서기 등)이 가능합니다.


**2) continuing environments의 경우 - average value**


episodic environments와는 다르게, **terminal state가 존재하지 않습니다**. agent는 **해당 environments에서 끝없이 동작해야 합니다**. 따라서, 1)과 같이 명시적으로 정해진 값은 존재하지 않기 때문에, **average value를 사용**합니다. 


이 때, **stationary distribution**이라는 개념이 등장합니다. stationary distribution이란, **agent가 오랜 시간 동안 특정 policy를 따랐을 때, 각 state에 도달할 장기적인 확률을 의미**합니다. $d^{\pi_\theta}(s)$로 나타냅니다.


stationary distribution과 policy $\pi_\theta$를 따랐을 때의 해당 state $s$에서의 value function을 $V^{\pi_\theta}(s)$ 곱한 값을 average value로 사용할 수 있습니다.


**3) continuing environments의 경우 - average reward per time-step**


average reward per time-step을 목표로 정의하는 경우입니다. 장기적인 state의 value function이 아닌, 해당 timestep에서 받을 수 있는 즉각적인 보상들을 의미하는 것이 차이입니다.



## Policy Gradient



### Finite Difference Policy Gradient


![1](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/1.png)


$J(\theta)$를 policy objective function이라고 할 때, gradient descent method를 사용하기 위해, poligy gradient를 구해야 합니다. $J(\theta)$에 근거해 update를 수행하기 위해 $\Delta\theta$를 구하면, $\Delta\theta=\alpha\nabla_\theta J(\theta)$라고 나타낼 수 있습니다.


$\nabla_\theta J(\theta)$를 구하기 위한 **가장 simple** 한 방법으론, $\theta_1$ 부터 $\theta_n$ 까지 각 $\theta_i$를 하나씩 값을 미묘하게 변화시킨 후의 값과 변화시키기 전 값과의 차이를 이용하는 방법이 있을 수 있겠습니다. differentiable 하지 않은 경우에도 사용할 수 있지만, 값을 얻기 위해 여러 번의 policy evaluation이 필요하기 때문에, simple 하지만 비효율적인 방법입니다. 이 방법을 **Finite Difference Policy Gradient** 방법이라고 합니다.



### Monte Carlo Policy Gradient


$\pi_\theta$가 differentiable하고, $\nabla_\theta \pi_\theta(s,a)$를 안다는 가정하에, 해당 방법을 사용할 수 있습니다. 이 방법을 위해, **Likelihood ratios trick**이 무엇인지 알아야 합니다.



#### **Likelihood ratios trick**


likelihood ratios trick은 아래와 같은 수식으로, **policy gradient 값을 변환하는 방법**을 말합니다. 


![2](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/2.png)_여기서의 \nabla_\theta~log~\pi_\theta(s,a)를 score function이라고 합니다._


이것이 왜 쓰이는 지를 알기 위해, **initial state에서 한 step 이동 후, reward를 받고 끝나는 상황을 가정**해 보겠습니다. (**one-step MDPs**)


![3](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/3.png)


initial state $s\sim d(s)$이고, 해당 state로부터 한 step 진행 후, reward $r=\mathcal{R}_{s,a}$를 받고 끝나는 case를 생각해봅시다. objective function $J(\theta)$을 average reward per time-step을 사용해 변환하겠습니다.


![4](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/4.png)


$J(\theta)$를 미분하면, $\nabla_\theta~\pi_\theta(s,a)$ term이 생기는데, likelihood ratios trick을 통해 아래와 같이 변환할 수 있고, 정리하면, $\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta~log~\pi_\theta(s,a)r]$로 나타낼 수 있게 됩니다.


![5](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/5.png)


이 방법을 사용하면, **기댓값으로서** **policy gradient를 정의**할 수 있어 **agent가 탐색하면서 얻는 값들을 통해 objective function**을 정의할 수 있게 되었습니다. 


그렇다면 어떤 policy들이 존재하는지 알아봅시다.

1. Softmax Policy

![6](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/6.png)


neural network에서 주로 output activation 용으로, output을 확률로 표현하고 싶을 때 많이 사용하는 softmax를 policy로 사용할 수 있습니다.

1. Gaussian Policy

![7](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/7.png)


만약, continuous한 action space에 있을 땐, gaussian policy를 더 자주 사용한다고 합니다.


두 방법 모두 score function을 구할 수 있고, 따라서 해당 방법으로 policy의 학습이 가능합니다. **Policy gradient theorem을 이용하면, one-step MDPs에 적용했던 방법을 multi-step MDPs에 적용**할 수 있습니다.



#### **Policy gradient theorem**


likelihood ratio approach를 이용한 방법을 multi-step MDPs에 적용하는 theorem 입니다.


![8](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/8.png)


multi-step MDPs로 확장하기 위해선, **one-step MDPs에서의 sample reward** $r$**을 long-term value인** $Q^\pi(s,a)$**로 대체**하면 됩니다. 


![9](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/9.png)


$Q^\pi(s,a)$**는 return** $v_t$ **(cumulative discounted reward)을 통해 근사**합니다. gradient를 value function을 통해서가 아닌 **return**을 통해 구하고 있으며, **monte-carlo estimate를 사용**하기 때문에, 이 방법을 **Monte-carlo Policy Gradient**라고 합니다.


![10](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/10.png)


pseudo code로 나타내면 위와 같습니다.


**Example**


![11](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/11.png)


Policy gradient 방법으로 이뤄지는 예시입니다. 몇가지 특징을 언급합니다. 정리하면 다음과 같습니다.

1. value function을 배울때처럼 들쭉날쭉하지 않고 **꾸준히 개선**
2. **속도가 굉장히 느립니다. variance가 크기 때문**인데, **return을 사용하는 것의 단점**이라고 볼 수 있습니다.

1번이라는 장점이 있지만, 2번의 **속도가 느리다는 치명적인 단점을 개선하기 위한 알고리즘**이 **Actor-Critic algorithm**입니다. 



### Actor-Critic Algorithm


![12](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/12.png)


monte carlo policy gradient 방법에서는 $Q^{\pi_\theta}(s,a)$를 return으로 대체했었습니다. **Actor-Critic algorithm에선** $Q_w(s,a)\simeq Q^{\pi_\theta}(s,a)$**로 Q를 학습해 근사**하는 방법입니다. 따라서, $Q_w$**와** $\pi_\theta$ **모두 학습의 대상**이 됩니다. 여기서 $\pi_\theta$**를 actor,** $Q_w$**를 critic으로 표현**하기 때문에, **actor-critic algorithm**이라고 합니다.


하지만 $Q$**와** $\pi$**는 서로 상호의존적인 관계**이기 때문에, 영향을 미칩니다. 따라서, actor-critic algorithms에선 $w$**와** $\theta$**가 순차적으로 update** 됩니다. Critic은 policy evaluation, Actor는 policy improvement를 담당하기 때문에, **policy iteration과 유사**하다고도 볼 수 있겠습니다.


![13](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/13.png)


Actor-Critic Algorithm의 pseudo code입니다. **TD-error** $\delta$**를 정의해, critic의 update (**$w$**)에 사용**합니다. **Actor의 parameter** $\theta$**는 policy gradient를 이용해 update** 되는 모습입니다. 


Monte Carlo 방법의 높은 variance 문제는 TD 방법론을 사용하는 Actor-Critic Algorithm을 통해 어느정도 해결할 수 있었습니다. 하지만 **여전히 높은 variance**는 문제가 될 수 있고, 그 때 **baseline function** $B(s)$**를 policy gradient에서 빼주는 방법을 사용**할 수 있습니다.


policy gradient가 높은 값들을 가지고 있다고 가정해 봅시다. 


$\text{data}_A=1,000,000~~~\text{data}_B=990,000$


A와 B는 값의 차이가 크지 않지만, 그 값 자체가 크기 때문에, high variance를 갖고 있다고 볼 수 있습니다. 여기서 $B(s)$로 둘의 평균값인 995,000을 빼준다면, 


$\text{data}_A=5,000~~~\text{data}_B=-5,000$


으로 **variance를 감소**시킬 수 있습니다. 이렇게 Baseline을 빼주면, **sample들이 평균 reward를 중심으로 분포하게 되어, 더 variance를 줄이면서, 효율적인 수렴이 가능**하다고 합니다.


따라서, $B(s)=V^{\pi_\theta}(s)$로 설정을 하고, $\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta~log~\pi_\theta(s,a)\cdot(Q^{\pi_\theta}(s,a)-B(s))]$로 계산을 수행하게 됩니다. 


하지만, **baseline을 빼주는 것이 기댓값에 영향을 준다면, 학습 자체가 요동치며 수렴하지 못하는 현상이 발생**할 수 있습니다. **Baseline을 빼주는 방법은 다행히 expectation 엔 영향을 주지 않는다**고 합니다.


![14](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/14.png)


1) likelihood ratio trick을 역으로 적용해 식을 다시 표현해주고, 
2) $B(s)$를 Sigma 밖으로 빼주었습니다. 
3) 여기서 남는 항 $\sum_{a\in\mathcal{A}}\pi_\theta(s,a)=1$이기 때문에
4) $B(s)$가 곱해진 policy gradient term은 0이 됩니다.


⇒ 따라서, **expectation에 영향을 주지 않으므로, high variance를 줄일 수 있는 좋은 방법**이라 할 수 있습니다.


![15](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/15.png)


좋은 Baseline은 Optimal Value function입니다. $A^{\pi_\theta}(s,a)=Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s)$**로 advantage function을 정의**해, **policy gradient를 재정의**합니다.


$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}[\nabla_\theta~log~\pi_\theta(s,a)\cdot A^{\pi_\theta}(s,a)]
$$


![16](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/16.png)


그런데 이렇게 정의한 **advantage function을 사용하려면,** $V^{\pi_\theta}(s)$ **역시 알아야** 합니다. 마찬가지로 **function approximator를 이용해** $V_v(s)\simeq V^{\pi_\theta}(s)$**를 근사**합니다.


$Q_w(s,a)$를 update하던 것 처럼 동일하게 **TD Learning 등을 통해 학습을 수행**합니다.


![17](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/17.png)


$V^{\pi_\theta}(s)$의 TD error $\delta^{\pi_\theta}$는 위와 같이 표현할 수 있습니다. 


해당 수식을 통해 **advantage function이 TD-error의 기댓값임**을 알 수 있습니다.


변환되는 부분인 $\mathbb{E}_{\pi_\theta}[r+\gamma V^{\pi_\theta}(s')|s,a]$은 정확히 $Q^{\pi_\theta}(s,a)$의 정의이기 때문에, 결과적으로 **advantage function은 V에 대한 TD-error의 기댓값**이라 할 수 있는 것입니다.


![18](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/18.png)


따라서, **advantage function**을 추정을 통해 수행하는 것이 아닌, **TD error의 sample로 구할 수 있게 됩니다**. 또한, **기존의 w로 Q를 근사해 사용**했지만, 이 방법을 사용하면 **V의 parameter set** $v$**만 학습**한다면, policy improvement를 위한 advantage function을 계산할 수 있으므로, 학습을 간소화할 수 있다는 장점도 있습니다.


이렇게 Actor-Critic algorithm의 전반적인 logic을 알아 보았습니다.


**Critics at Different Time-Scales**


![19](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/19.png)


Critic 학습엔 MC, TD(0), TD($\lambda$) 사용 가능하다는 slide 입니다.


**Actors at Different Time-Scales**


![20](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/20.png)


![21](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/21.png)


Actors 학습에도 역시, Monte-carlo (complete return), one-step TD error, TD($\lambda$) (mix over time-scales)등이 가능하다는 slide입니다.


![22](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/22.png)_기존의 방법 - (s,a) 별 방문 횟수, 최근에 방문했는 지가 중요했음_


참고로 backward-view TD($\lambda$)의 경우는 eligibility traces를 적용할 때 **기존의 value function을 이용했던 방법과는 다르게,** **policy의 parameter 자체를 최적화하는 것이 목표**이기 때문에, $\nabla_\theta~\text{log }\pi_\theta(s,a)$를 이용하는 모습입니다. **policy가 특정 action을 선택할 확률에 대한 변화를 추적하는 것이 중요하기 때문**입니다.


**정리**


![23](/assets/img/2024-11-01-David-Silver---RL-Lecture-7-(Policy-Gradient).md/23.png)


이렇게 **policy gradient 방법**에 대해 알아보았습니다. **value function을 이용하지 않고, policy 자체를 최적화하기 때문에, policy 자체의 gradient를 이용하는 것이 중요**했습니다. 가장 simple한 finitie 방법부터, **MC**를 이용한 방법, 여러 **Actor-Critic 알고리즘들** 까지 살펴보았습니다.


강의에 따르면, policy gradient까지 완벽하게 익히면 대부분의 논문을 읽는 데는 문제가 없다고 합니다. 하지만, 저는 완강을 목표로 하고 있기 때문에 다음은 integrating learning and planning이란 주제의 강의 정리로 돌아오겠습니다!

