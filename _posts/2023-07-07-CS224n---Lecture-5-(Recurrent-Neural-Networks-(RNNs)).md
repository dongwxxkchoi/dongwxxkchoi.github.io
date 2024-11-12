---
layout: single
date: 2023-07-07
title: "CS224n - Lecture 5 (Recurrent Neural Networks (RNNs))"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
---

> ☑️ **Lecture 5  
>   
> 1. Neural dependency parsing  
> 2. A bit more about neural networks  
> 3. Language modeling + RNNs**



### Dependency parser


![0](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/0.png)


**conventional transition based dependency parser**

- worked with indicator features**(= boolean features)**

	→ represent the presence or absence


	→ specifying some condition if it’s true of a configuration

	- word on top of the stack is good
	- part of speech is adjective
	- next word coming up is a personal pronoun

이런 구성 요건 하나하나가 boolean features로 저장 


그렇기 때문에 **단점이 존재**


단점 1 - features: **sparse**  


단점 2 - features: **incomplete** 


단점 3 - **expensive computation**


	→ 95%의 parsing time이 feature computation에 쓰임


그래서 등장한 것이 **Neural Dependency Parser**


(좀 더 dense하고 compact한 feature representation 보여줌)



#### Neural Dependency Parser


![1](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/1.png)

- use **Stack** and **Buffer**
- **dimensionality** : several million → 1,000

![2](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/2.png)


두 가지 type의 parser 

- **Transition based parser**
	- 종류
		- MaltParser
		- C&M 2014
	- 특징
		- 빠름 - 모든 feature computation 수행하지 않아서
- **Symbolic Graph based parser**
	- 종류
		- MSTParser
		- TurboParser
	- 특징
		- 정확함
		- transition based parser에 비해 느림

**⇒** **C&M Parser → best**



#### Distributed Representations


각 word를 word embedding해서 나타냄


→ Similar words는 close vectors


![3](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/3.png)


**part-of-speech tags (POS)**와 **dependency labels** 또한 d-dimensional vectors로 표현 가능


더 작은 discrete sets가 많은 semantical similarities를 드러낼 수 있음


![4](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/4.png)


---


> 💡 **What is Part-of-speech (POS) tags (품사 태깅)?**  
>   
> syntactic and grammatical context(문맥)에 기반해 문장에서 각 단어의 품사를 할당하는 과정  
>   
> - **단어의 카테고리**  
> - **문장에서의 구문 기능**  
> - **다른 단어들과의 관계**를 나타냄  
>   
> 일반적인 pos:   
> - nouns (명사)  
> - verbs (동사)  
> - adjectives (형용사)  
> - adverbs (부사)  
> - pronouns (대명사)  
> - prepositions (전치사)  
> - conjunctions (접속사)  
> - interjections (감탄사)  
>   
> **The cat sat on the mat**  
> - “The”: DT(determiner)  
> - "cat": NN(noun)  
> - "sat": VBD(verb; past tense)  
> - "on": IN(preposition or subordinating conjuction)  
> - "the": DT(determiner)  
> - “mat”: NN(noun)  
>   
> 개체명 인식, 감정 분석, 텍스트 분류 및 기계 번역과 같은 많은 자연어 처리 작업에 중요한 전처리 단계


[bookmark](https://stackoverflow.com/questions/29332851/what-does-nn-vbd-in-dt-nns-rb-means-in-nltk)


---


**Example - POS tagging & dependency labels**


![5](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/5.png)



### Extracting Tokens & vector representations from configuration


stack, buffer을 이용해, set of tokens를 추출하는 방법


![6](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/6.png)


**→ C&M Parser 읽는 법?**


[bookmark](https://velog.io/@tobigs-text1415/Lecture-5-Linguistic-Structure-Dependency-Parsing)



#### Softmax Classifier


deep learning에서 자주 사용하는 simple한 classifier


![7](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/7.png)

- **inputs** : $x \in \mathbb{R}^d$
- **classes** : $y \in C$
- **Weight** **matrix** : $W \in \mathbb{R}^{C\times d}$

d dimensional vectors인 **x를 입력**으로 받아 **C에 속한 클래스 y에 대한 확률을 반환**


$$
p(y|x) = \frac{exp(W_y \cdot x)}{\Sigma^C_{c=1}exp(W_c\cdot x)}
$$

- **softmax**를 통해, negative log loss인 $\Sigma_i-log~p(y_i|x_i)$ 를 최소화함 **(= cross entropy loss)**
- Traditional ML classifiers에 비해 성능 좋음 **(non-linear classification > linear classification)**

	![8](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/8.png)


하지만 softmax 자체가 non linear한 classification을 제공하는 것은 아님


softmax 아래에 쌓여 있는 **neural network로 부터 non linear한 표현 공간**을 제공 받음


(what neural net can do is warp the space around and move the representation of data points)


softmax는 그 비선형적 표현 공간에서 **linear한 classification 수행**


**⇒ simple feed forward neural network multi-class classifier**


![9](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/9.png)


![10](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/10.png)


![11](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/11.png)


![12](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/12.png)


이렇게 Neural Network가 구성되는데, Dependency parser


![13](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/13.png)


![14](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/14.png)


Graph-based dependency parsers



#### Multiclass Classifier using softmax


Dependency parser model


graph-based dependency parser



#### Regularization



#### L2 Regularization


![15](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/15.png)

- big neural nets, so many parameters → **overfit 가능성 높음**
- $\lambda\Sigma_k\theta^2_k$에서의 $\lambda$**는 strength of regularization을 나타냄**
	- $\lambda$가 크면 클수록, regularization 강도 세짐

![16](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/16.png)



#### Dropout

- Train 과정에서, **랜덤하게 뉴런의 일부분을 0으로 만들어** 학습되지 않도록 하는 것
- prevent **feature co-adaptation**
undefined<details>
  <summary>**feature co-adaptation**</summary>


상호적응 문제는, 신경망의 학습 중, **어느 시점에서 같은 층의 두 개 이상의 노드의 입력 및 출력 연결강도가 같아지면, 아무리 학습이 진행되어도 그 노드들은 같은 일을 수행하게 되어 불필요한 중복이 생기는 문제**를 말한다. 즉 연결강도들이 학습을 통해 업데이트 되더라도 이들은 계속해서 서로 같은 입출력 연결 강도들을 유지하게 되고 이는 결국 하나의 노드로 작동하는 것으로써, 이후 어떠한 학습을 통해서도 이들은 다른 값으로 나눠질 수 없고 상호 적응하는 노드들에는 낭비가 발생하는 것이다. 결국 이것은 컴퓨팅 파워와 메모리의 낭비로 이어진다.


드랍아웃은 이러한 상호적응 문제를 해소한다. 즉, 드랍아웃이 임의로 노드들을 생략할 때 이러한 상호 적응 중인 노드들 중 일부는 생략하고 일부는 생략하지 않게 되므로 학습 중 상호 적응이 발생한 노드들이 분리될 수 있어서 상호 적응 문제를 회피할 수 있게 된다.



  </details>- Naive Bayes 모델과 Logistic Regression Model의 중간 느낌을 제공

	(Naive Bayes는 weights가 모두 독립적인 반면, logistic regression model에선 모든 weights가 연관이 있음)



#### Vectorization


**Vector loop** v.s. **concatenated vector matrix**


![17](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/17.png)


![18](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/18.png)



#### Activation Function → Non-linearities


Multi-Layer Neural Network를 설계하면서, **non-linearity를 갖는 것이 중요**하다고 위에서 언급


Layer 사이에서 그 역할을 해주는 것이 바로 activation function


![19](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/19.png)

- **Sigmoid**

	→ real number를 받아 [0, 1]로 변환


	→ 항상 positive space인 것이 단점

- **tanh**

	→ rescaled and shifted version of sigmoid


	→ [-1, 1] 사이에서 대칭


	→ 느리고, 계산하기 expensive

- **hard tanh**

	→ fast, less expensive


	→ flat lines을 살린 tanh

- **ReLU (Rectified Linear Unit)**

	→ 가장 많이 사용하는 방식 


	→ 가장 simple, negative는 0, positive는 y=x


	→ train very quickly, straightforward gradient back flow

- **Leaky ReLU**

	→ ReLU의 변형


	→ negative 부분에 slope

- **Parametric ReLU**

	→ slope를 parameter화 해서 조절할 수 있음

- **Swish**

	→ 0 부근에서 gradient를 가짐


이 중 어떤 것도 superior한 것은 없음



#### Parameter Initialization

- Uniform distribution [-r, r]
- zero initialization
- **Xavier Initialization**

	sigmoid 계열의 activation function을 사용할 때, 가중치를 초기화하는 방법


	입력 데이터의 분산이 출력 데이터에서 유지되도록 가중치를 초기화


	$$
	Var(W_i) = \frac{2}{n_{in}+n_{out}}
	$$



#### Optimizer

- **SGD**
- Adagrad
- RMSprop
- **Adam → please Use!**
- SparseAdam


#### Learning Rates


model에 따라 설정해야 하는 정도가 다름


$10^{-3} \sim 10^{-4}$ 정도면 좋음


→ **halve the learning rate after every k epochs**



#### Language Modeling


**Language Modeling**


: the task of predicting what word comes next


![20](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/20.png)


![21](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/21.png)


![22](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/22.png)


즉, $x^{(1)},x^{(2)},...,x^{(t)}$의 sequence가 앞에 주어졌을 때, 


다음 word인 $x^{(t+1)}$의 probability distribution을 계산하는 것이라고 할 수 있다.


(단, $x^{(t+1)}$은 vocab 모음인 $V = \{w_1, ..., w_{|V|}\}$에 속하는 word)


(즉, 우리가 이미 알고 있는 vocab 중에 예측을 진행)


![23](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/23.png)


→ _**Language Model로 다음을 계산할 수 있음**_



#### n-gram Language Models


neural network 방식 전 2~30년 동안 지배적인 방식


**n-gram**


: chunk of _**n**_ **consecutive(연이은) words**

- unigrams - 1개 word
- bigrams - 2개 words
- trigrams - 3개 words
- 4-grams - 4개 words

**Markov Assumption**


→ $x^{(t+1)}$**depends only on the preceding** _**n**_**-1 words**


![24](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/24.png)_즉, 앞의 n-1 words에만 영향을 받는다는 가정_


![25](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/25.png)


조건부 확률을 구하려면, 각각 n, n-1 gram의 확률을 다 구해야 함    **how?**


![26](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/26.png)


→ like 큰수의 법칙


![27](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/27.png)


**n-gram** 모델을 사용 → $x^{t+1}$의 예측을 위해 그 **앞의 n개의 words만 고려**하겠다. (그 전은 discard)


![28](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/28.png)


하지만 이렇게 된다면, 앞의 proctor에 대한 내용은 사라지게 되어, 


문맥에 더 맞는 words인 **exams**가 아니라 **books**로 예측할 확률이 높음


**⇒ Problem 1**


(Counting을 이용한다는 점에서 Naive Bayes Models와 비슷, but unigram이기 때문에 neighbors를 고려하지 않는 다는 점은 다름)



#### Sparsity Problem


![29](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/29.png)


**특정 N-gram의 corpus 내 출현 빈도가 낮을 때 발생** 


**Problem 1) “**students opened their w”가 이전에 등장하지 않는다면?


**→ smoothing** (작은 값을 더해 0이 되지 않도록 하는 것)


**Problem 2) “**students opened their” 가 이전에 등장하지 않는다면?


**→ backoff** (use N-1 gram)


**⇒ n을 증가시키는 것은 심각한 Sparsity Problem 초래**



#### Storage Problems


![30](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/30.png)


**N-gram의 모든 정보를 저장하려면, model의 크기가 지나치게 커짐**


---



#### **Example - Generating Text by predicted words**


![31](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/31.png)


![32](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/32.png)


![33](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/33.png)


![34](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/34.png)


![35](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/35.png)


![36](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/36.png)


**grammatical but not incoherent**


**trigram의 예시인데, n을 더 늘리는 것은 문제가 있으므로 좋지 못한 방법**


---



### Neural Language Model


window를 center word 주위가 아닌, center word 직전으로 생성 (fixed window)


![37](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/37.png)


그걸 바탕으로 해서 neural network 설계


![38](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/38.png)


→ softmax를 통해, vocab 속 word들에 대한 확률 반환


→ 학습에 negative sampling도 사용 가능


![39](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/39.png)

- **장점**
1. Softmax → **Sparsity 문제 해소**
2. Counting 값을 저장할 필요가 없음 → **Storage 문제 해소**
- **단점**
1. N-gram과 같이 문맥을 반영하지 못함 (small window size)
2. 단어의 위치에 따라 곱해지는 가중치가 다르기 때문에 **Neural Model이 비슷한 내용을 여러 번 학습하는 비효율성**을 가짐


### RNN(Recurrent Neural Network) Language Model


**순차적인 (Recurrent) 정보를 처리하는 것이 기본 아이디어**


**1) 동일한 태스크를 한 시퀀스의 요소마다 적용**


**2)** **출력 결과가 이전의 계산 결과에 영향 받음**


![40](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/40.png)


![41](/assets/img/2023-07-07-CS224n---Lecture-5-(Recurrent-Neural-Networks-(RNNs)).md/41.png)

