---
layout: single
date: 2023-07-03
title: "CS224n - Lecture 3 (Backprop and Neural Networks)"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
---


### Named Entity Recognition (NER)


한글로 **개체명 인식**이라고 합니다.


어떤 이름을 의미하는 단어를 보고는 그 **단어가 어떤 유형인지를 인식**하는 것


(neural network의 예시를 위해 나온 task로 보임)


![0](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/0.png)_분홍색 글씨로 쓰인 것은 그 단어의 유형을 나타낸 것_


PER 사람이름, LOC 위치, DATE 시간 등으로 분류가 되어 있는데,


눈에 띄는 것은 **Paris**입니다. 


위의 예시에서 Paris는 두 가지로 분류되는데,


Paris Hilton의 **Paris(PER)**와, Hilton Hotel이 있는 **Paris(LOC)**입니다.


→ 이렇게 같은 이름을 다른 유형으로 인식하기 위해선 **context를 이용**해야 합니다.


이 작업을 위해 다음 과정을 수행합니다.


> 🧮 **과정**  
> 1. word vector로 이뤄진 **context window** 만들기  
>   
> 2. **neural network layer**에 투입  
>   
> 3. **logistic classifier**로 **분류** (일단은 simple 하게)

	1. word vector로 이뤄진 **context window** 만들기
	2. **neural network layer**에 투입
	3. **logistic classifier**로 **분류** (일단은 simple 하게)

우리는 문맥속에 등장하는 **Paris**라는 단어가 **LOC**인지 알아볼 것입니다. 


**이 주제를 통해 전반적인 neural network 과정을 살펴볼 것입니다.**



#### A. context window


$x_{Paris}$**와 location 과의 연관성**을 찾아보기 위해, $x_{Paris}$ 주위의 **context window**를 구성


![1](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/1.png)


x 하나 하나는 word imbedding을 통해 구성된 **word vector를 의미**하고


그 결과, $X_{window}$는 **길게 늘여진 vector의 형태**가 됩니다.


![2](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/2.png)_예시는 4-dimensional word imbedding 시의 window vector_



#### B. **neural network layer**


context window를 통해 vector가 구성된 후, neural network layer들로 투입되게 됩니다.


![3](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/3.png)



#### C. logic classifier


neural network layer들을 지나, logic classifier에 활용하기 위해 한 probability value를 output함


($s = u^Th$**; u는 (mx1)의 weight vector, 위의 예시에선 (8x1)**)


**s를 확률로 변환**하기 위해 **sigmoid** 함수를 사용해, 최종 output인 $J_t(\theta) = \sigma(s) = \frac{1}{1+e^{-s}}$를 반환함


이렇게 output을 산출해 loss function을 통해 차이를 줄여나가야 합니다.


neural network 방식에서 가장 많이 사용하는 방식이 **Gradient Descent 방식**이고, 이 방식을 사용하기 위해선 **parameter의 지속적인 update**가 필요합니다. cost로 인해 Stochastic Gradient Descent 방식을 사용합니다.


$$
\theta^{new} = \theta^{old}-\alpha\triangledown_\theta J(\theta)
$$


각 parameter 단위로 본다면 $\theta_j^{new} = \theta_j^{old}-\alpha \frac{\partial J(\theta)}{\partial~\theta_j^{old}}$로 나타낼 수 있습니다.


**이 A→B→C의 과정을 통해 진행됨**



### Gradient Descent

1. **n inputs and 1 output**

$f(x) = f(x_1, x_2, ..., x_n)$ 


$\frac{\partial f}{\partial x} = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}]$ 

1. **n inputs and m outputs   (MxN)**

 $f(x) = [f_1(x_1, x_2, ..., x_n),...,f_m(x_1, x_2, ..., x_n)]$ 


![4](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/4.png)


---



#### **What is Jacobian?**


자코비안(Jacobian)은 **벡터 함수(vector-valued function)의 각 원소에 대한 모든 가능한 일차 편미분(First-order partial derivatives)을 구성한 행렬**


자코비안 행렬은 **벡터 함수의 미분에 관한 정보**를 담고 있으며, 매우 중요한 선형 대수학적 도구로 사용됩니다. 예를 들어, 자코비안 행렬은 함수의 기울기 벡터(gradient vector)를 계산하고, 함수의 테일러 전개(Taylor expansion)를 위한 선형 근사(linear approximation)를 수행하는 등의 다양한 분야에서 활용됩니다.


말로 들으니 뭔 소린지 모르겠네요…


수식을 통해 알아보겠습니다.


![5](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/5.png)


다음과 같이, n개의 input을 받는 m개의 함수가 있다고 합시다.


이들을 모두 편미분 하려면 다음과 같은 벡터/행렬로 표시합니다.


![6](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/6.png)_Jacobian Matrix_


이렇게 n개의 변수를 받는 m개의 함수가 있을 때, 이 함수들의 편미분을 구해야 하는 상황입니다.


이 때, 이 **전체 편미분**을 우리가 단순히 **곱해서 더하는 폼으로 만들어 놓을 수 있는 것**을 **Jacobian Matrix**라고 합니다.


예시를 통해 알아보죠


![7](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/7.png)


$F_1(x, y) = x^2y,~~~F_2(x,y)=5x+sin(y)$ 일때,


Jacobian Matrix  $J_F(x, y)$ 는 다음처럼 나타냅니다.


![8](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/8.png)


만약, 2개의 변수에 3개의 함수라면?


**→ 3x2 matrix가 생성됨**


**이처럼 Jacobian matrix의 형태는 mxn matrix입니다. (m: 함수 개수, n: input 개수)**



#### **도함수는 어떻게 구할 수 있을까? - Chain Rule**


우리는 **chain rule을 이용**해, 도함수를 구한다.


![9](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/9.png)


Jacobian의 경우에도 비슷하게 수행 가능


(아래의 z, W, b 등은 모두 행렬값)


![10](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/10.png)


($h=f(z)$를 미분해 $\frac{\partial h}{\partial z}$, $z = Wx + b$를 미분해 $\frac{\partial z}{\partial x}$를 구함)


(1) $x$  (input)    (Mx1)


![11](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/11.png)


(2) $z = Wx +b$    (W: NxM,   b: Nx1)


![12](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/12.png)


![13](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/13.png)


(3) $h = f(z)$    (h: Nx1,    z: Nx1)  


![14](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/14.png)


(4) $s = u^Th$    (s: scalar,    u: Nx1) 


![15](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/15.png)


**(1) → (4)의 방향을 통해 forward propagation**이 이뤄지고


![16](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/16.png)


**gradient descent 방식을 통해 (4) → (1)의 방향으로 backpropagation**이 이뤄진다.


![17](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/17.png)


![18](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/18.png)_상세 과정 1_


![19](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/19.png)_상세 과정 2_


이 과정에서 $\frac{\partial s}{\partial b}, \frac{\partial s}{\partial z} $ 등은 바로 알 수 없다. 이를 간편하게 해주는 방식이 바로 **chain rule**이다.


그렇다면, 각 (1)~(4)에서의 편미분값을 계산하려면 어떻게 chain rule을 적용해야 하는지 알아보자.



#### (4) $s = u^Th$    (s: scalar,    u: Mx1) 


우선, (4)는 s를 구하는 식이기 때문에, backpropagation의 처음 단계이다.


![20](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/20.png)


**scalar를 vector로 편미분**하는 경우에 해당한다.


---


스칼라 함수를 벡터로 미분(Gradient) (y가 scalar, x가 vector)


![21](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/21.png)


---


![22](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/22.png)


> 💡 왜 $u^T$**인가요?**  
> → $s = u^Th$ 에서 s는 $u^Th$ 를 통해서만 h에 연관되기 때문  
> → dot product의 derivate는 계산에서의 다른 vector의 transpose로 주어진다


$$
\frac{\partial s}{\partial h} = u^T
$$



#### (3) $h = f(z)$    (h: Nx1,    z: Nx1)  


![23](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/23.png)


**vector를 vector로 편미분하는 경우**에 해당


---


벡터를 벡터로 미분


![24](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/24.png)


---


![25](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/25.png)


> 💡 **왜 diagonal matrix가 나오나요?**  
> $h_i = f(z_i)$ 처럼, index가 같은 변수끼리만 영향을 받으므로, index가 같지 않은 경우는 partial derivative가 0이 됩니다.


$$
\frac{\partial s}{\partial z} = \frac{\partial s}{\partial h}\frac{\partial h}{\partial z} = u^Tdiag(f\prime(z))
$$



#### (2) $z = Wx +b$    (W: NxM,   b: Nx1)


![26](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/26.png)


![27](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/27.png)



#### (2) - **1**     $\frac{\partial s}{\partial W}$


jacobian matrix의 예시에서 N개의 함수, M개의 input이 있는 경우에 해당


$\frac{\partial s}{\partial W} = \frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial W} = u^Tdiag(f\prime(z))x^T$


![28](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/28.png)



#### (2) - 2     $\frac{\partial s}{\partial b}$


jacobian matrix의 예시에서 N개의 함수, 1개의 input이 있는 경우에 해당


$\frac{\partial s}{\partial b}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial b} = u^Tdiag(f\prime(z))$


![29](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/29.png)


> 💡 **왜** $\frac{\partial z}{\partial b}=I$ **인가요?**  
> $z = Wx + b$ 이고, $z_i = Wx + b_i$ 이기 때문에, $z$와 $b$는 index가 같을 때만 영향을 받아 일단 diagonal matrix입니다. 근데, z의 식을 보면 b가 directly added 되어 있기 때문에, b를 바꾸는 것이 직접적으로 z에 영향을 줍니다. 그렇기 때문에, $\frac{\partial z_i}{\partial b_i} = 1$ 이기 때문에, Identity Matrix가 되는 것입니다



#### (1) $x$  (input)    (Mx1)


![30](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/30.png)


$\frac{\partial s}{\partial x}$ **-** Jacobian matrix의 예시에서 N개의 함수, M개의 Input이 있는 경우에 해당


$\frac{\partial s}{\partial b}=\frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial x} = u^Tdiag(f\prime(z))W$


![31](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/31.png)


> 💡 **왜** $\frac{\partial z}{\partial x} = W$**인가요?**  
> 위에서 얘기했듯이, dot product에서의 한쪽을 대상으로 편미분을 하면 나머지 계산의 대상이 결과물입니다.  
> **”dot product의 derivate는 계산에서의 다른 vector의 transpose로 주어진다”**


---

- **헷갈리는 점?**

$u^Th$ 처럼 u와 h의 dot product는 T가 붙어있지만, $Wx+b$에서 W와 x의 dot product에는 T가 붙어있지 않음. 그렇기 때문에 W는 NxM인지, MxN인지 헷갈림.


---


💡 **계산량을 줄이는 방법**


$\frac{\partial s}{\partial W}$ = $\frac{\partial s}{\partial h} \frac{\partial h}{\partial z}$$\frac{\partial z}{\partial W}$


$\frac{\partial s}{\partial b}$ = $\frac{\partial s}{\partial h} \frac{\partial h}{\partial z}$$\frac{\partial z}{\partial b}$


(→ 공통부분 존재)


**→ avoid duplicated computation**


define $\delta$

<details>
  <summary>delta?</summary>


---


$\delta$ is defined in two methods


1) **small change in a variable or function**


can take the limit as delta approaches zero to obtain the derivative.


2) difference between the actual output and the desired output of a neural network


(local error signal)


This is used in backpropagation algorithms to calculate the gradients of the loss function with respect to the network parameters. The delta is propagated backwards through the layers of the network to update the weights and biases.


---



  </details>

#### $\delta $ = $\frac{\partial s}{\partial h} \frac{\partial h}{\partial z}$



#### $\frac{\partial s}{\partial W}$ = $\delta$ $\frac{\partial z}{\partial W}$



#### $\frac{\partial s}{\partial b} $ = $\delta$ $ \frac{\partial z}{\partial b}$


---



#### disagreement between jacobian form and the shape convention


![32](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/32.png)



#### Let’s Get $\frac{\partial z}{\partial W}$


$W \in \mathbb{R}^{n \times m}$


$\frac{\partial s}{\partial W} = \frac{\partial s}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial W} = u^Tdiag(f\prime(z))\frac{\partial z}{\partial W}$


![33](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/33.png)


⇒ 결과물은 1xM matrix


$\theta^{new} = \theta^{old}-\alpha \triangledown_\theta J(\theta)$


하지만, **W는 NxM matrix**이기 때문에, **shape convention을 해줘야 함**


이처럼, **backpropagation을 하다보면, SGD 방법을 사용하기 편하지 않은 형태가 되기도 함**


![34](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/34.png)


그림에서 볼 수 있다시피, $\frac{\partial z}{\partial W}$는 두가지 형태, NxM 또는 1xM matrix로 나타낼 수 있다.


**(1xM matrix가 바로** $x^T$**에 해당)**


⇒ $\frac{\partial s}{\partial W}$$= $ $\delta$ $\frac{\partial z}{\partial W}$ $ =$ $\delta^T$ $x^T$


($\delta$: local error signal at z)


($x$ is local input signal)

<details>
  <summary>왜 $\frac{\partial z}{\partial W} = x$ 일까 강의 ver.</summary>


**왜** $x$**일까?** 


$\frac{\partial s}{\partial W}$$= $ $\delta$ $\frac{\partial z}{\partial W}$ $ =$ $\delta$ $\frac{\partial}{\partial W}(Wx + b)$


다시 이렇게 표시할 수 있는데,


이제 single weight인 $W_{ij}$에 대한 derivative를 생각해보자


$W_{ij}$ 는 $z_i$에만 영향을 받음


ex. $W_{23}$은 오직, $z_2$에만 영향을 주지, $z_1$에 영향을 주진 않음



### $\frac{\partial z_i}{\partial W_{ij}} $$= \frac{\partial}{\partial W_{ij}}W_i\cdot x + b_i = \frac{\partial}{\partial W_{ij}}\Sigma^d_{k=1}W_{ik}x_k = x_j$


($\because z_i = W_i\cdot x + b_i$)


($\because W_i \cdot x = \Sigma^d_{k=1}W_{ik}x_k$, W는 2차원 matrix)



#### ⇒ $\frac{\partial s}{\partial W}$$= $ $\delta$ $\frac{\partial z}{\partial W}$ $ =$ $\delta^T$ $x^T$



  </details>
**그래서 아래와 같은 식이 가능했던 것이다.**


![35](/assets/img/2023-07-03-CS224n---Lecture-3-(Backprop-and-Neural-Networks).md/35.png)


비슷하게, $\frac{\partial s}{\partial b}$ $= h^T \circ f\prime(z)$ 는 **row vector  (1xN)**


b는 Nx1 column vector이기 때문에, transpose를 통해 shape convention 필요

- Jacobian form → chain rule 적용에 유리
- shape convention → SGD 적용에 유리

어떻게 할까?

1. Use Jacobian forms as much as possible, reshape to follow the shape convention at the end
	- 우리가 한 방식입니다.
	- 하지만, 마지막에 transpose $\frac{\partial s}{\partial b}$ to make the derivative a column vector, resulting in $\delta^T$
2. Always follow the shape convention
	- Look at dimensions to figure out when to transpose and/or reorder terms
	- The error message $\delta$ that arrives at a hidden layer has the same dimensionality as that hidden layer
