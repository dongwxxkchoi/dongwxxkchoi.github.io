---
layout: single
date: 2023-06-09
title: "CS224n - Lecture 1 (Intro & Word Vectors)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

> ☑️ **Lecture 1 : Introduction and Word Vectors**  
> 1. Human language and word meaning  
> 2. Word2vec introduction  
> 3. Word2vec objective function gradients  
> 4. Optimization basics  
> 5. Looking at word vectors



### 말의 의미를 어떻게 표현할까?


**의미**란 무엇일까?

- the idea that is represented by a word, phrase, etc.

	word, phrase에 의해 표현되는 생각

- the idea that a person wants to express by using words, signs, etc.

	words, signs을 사용해 표현하고자 하는 사람의 생각

- the idea that is expressed in a work of writing, art, etc.

	writing, art 작품 등에 의해 표현되는 생각


> 💡 **signifier** (기표; symbol) ↔ **signified** (기의; idea or thing)  
> **⇒ denotational semantics (표시적 의미론)**


**가장 간단한 NLP solution?**


→ 개별적인 word에 대해 dictionary, thesaurus 등을 사용하는 것이다.


(synonym(동의어 관계) sets, hypernyms(상하위 관계)를 포함)


→ 다시 말해 “ISA” relationship이라고 할 수 있다.


![0](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/0.png)



#### 간단한 만큼 단점도 존재

- **nuance**를 살리지 못합니다.

	→ **context의 고려**가 필요함

- word의 새 의미를 놓침

	→ ex. wicked 등등


	→ 항상, up-to-date하기 힘듦


이처럼, 전통적인 NLP는 단어를 그저 discrete symbol로 여겨 왔습니다


→ one-hot vector로 변환해 matrix에서 한 값만 1이고 나머지가 0인 데이터로 취급


→ one-hot vector로 변환된 **두 vector는 유사한 의미**를 갖나, **orthogonal**하기 때문에, **similarity가 없음**


![1](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/1.png)


![2](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/2.png)



#### 문맥을 통해 words 나타내기


> 💡 **Distributional semantics (분포 의미론)**  
> → word의 meaning은 근처에 자주 나타나는 words들에 의해 주어짐


이를 통해, 여러가지의 context 속에서의 w를 사용해, w를 나타내는 것이다.


![3](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/3.png)


→ 이처럼 **각 word에 대한 vector**를 **우리가 설정**해, **context와 연관**되게 함


→ 이 word vectors를 **word embeddings** 또는 **word representations**라고 한다.


→ **실제**로는 훨씬 더 **고차원**일 것


대표적 예시 **Word2vec**


---



## Word2vec


2013년 발표된 word vector 학습 **framework**


(neural network 이용)


![4](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/4.png)


* corpus: 말뭉치


* text의 t라는 위치, c: centor word, o: context words


→ c와 o의 word vector들의 similarity를 사용해, 주어진 c(centor)가 주어졌을 때의 o(해당 context)를 예측/계산 하는 것


⇒  $P(w_{t+j}|w_t)$를 구하는 것 (t가 주어졌을 때(t를 중심으로) 앞뒤의 w의 확률을 예측)


![5](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/5.png)


**(skip-gram 방식)**


![6](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/6.png)

- 각 위치 **t = 1, …, T  /**  즉, 총 단어의 수가 T
- context words **o**를 예측
- window of fixed size **m** (window 안 2m+1의 words)
- given center word $w_j$
- data likelihood: $L(\theta) = \Pi^T_{t=1}\Pi_{-m\leq j \leq m, j\ne0}P(w_{t+j}|w_t;\theta)$
- objective function(cost, loss): $J(\theta) = -\frac{1}{T}logL(\theta)$

	**Minimize** $J(\theta)$ **↔ Maximize** $L(\theta)$


⇒ 즉, centor word 기준 2m개의 확률이 높아지게


그렇다면 어떻게 하면, $P(w_{t+j}|w_t;\theta)$를 계산할 수 있을까?


⇒ word **w**로 **두 벡터값을 이용**

- $v_w$ : w가 centor word(c)일 때
- $u_w$ : w가 context word(o)일 때

이렇게 두개로 나눠 **center word가 주어졌을 때의 그 context word를 예측**할 수 있다. (outputs)


$$
P(o|c) = \frac{exp(u^T_ov_c)}{∑_{w∈V}exp(u^T_wv_c)}
$$


centor일 때 output(context)의 확률


하나씩 살펴보자

- $exp(u^T_ov_c)$에서 $u^T_ov_c$

	dot product를 통해 o와 c의 유사도를 비교함 ( $∑^n_{i=1}u_iv_i$ )


	만약 이 **dot product의 결과값이 크다면 더 유사도가 높은 것**이다.


	(벡터 공간에서 같은 방향에 있는 벡터일수록 내적의 값이 커지기 때문)

- $∑_{w∈V}exp(u^T_wv_c)$

	entire vocabulary에 대해 normalize를 수행해 probability distribution을 제공


사실 이 수식의 꼴은 softmax 함수를 적용한 형태이다.


softmax의 특징이라 하면, classification에서 주로 사용하고,


여러가지 가능한 경우에 대한 확률을 나타낸다


각 확률의 총 합은 1이 된다


그렇기 때문에 다시 적는다면,


$$
softmax(x_i) = \frac{exp(x_i)}{∑^n_{j=1}exp(x_j)} =p_i
$$


이렇게 나타낼 수 있다.


즉, arbitrary values $x_i$를 probability distribution $p_i$로 매핑

- soft → 더 **작은** $x_i$**에도 값을 할당**하기 때문
- max → **가장 비슷한** $x_i$**의 확률을 증폭**시키기 때문

이 식에 log를 씌우는데, 이는 계산을 편리하게 하기 위함임


(Log와 exp는 상쇄되기 때문)


이 모델에선 gradient를 줄이는 방식으로 학습하기 때문에,


미분을 통해 미분계수를 구함


→ 미분식은 아래


word2vec 상세 설명:


[https://ratsgo.github.io/from frequency to semantics/2017/03/30/word2vec/](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/30/word2vec/)


워드 임베딩 과정 그래픽:


[bookmark](https://ronxin.github.io/wevi/)


![7](/assets/img/2023-06-09-CS224n---Lecture-1-(Intro-&-Word-Vectors).md/7.png)



#### 왜 loss function을 ( predicted value - observation value )라고 표현할까?


word2vec의 skip-gram 모델의 학습은 context window 단위로 이뤄짐


→ context window의 centor word를 input으로 받았을 때, 모든 vocab들에 대한 발생 확률을 output


(softmax)


→ 그 확률들을 $L(\theta) = \Pi^T_{t=1}\Pi_{-m\leq j \leq m, j\ne0}P(w_{t+j}|w_t;\theta)$


$$
\mathcal{L} = -\log \sigma(\mathbf{v}{w_o}^T \mathbf{v}{w_i}) - \sum_{j=1}^{k} \log \sigma(-\mathbf{v}_{\tilde{w}j}^T \mathbf{v}{w_i})
$$

