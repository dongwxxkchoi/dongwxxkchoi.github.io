---
layout: single
date: 2023-06-29
title: "CS224n - Lecture 2 (Neural Classifiers)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---

> ☑️ **Lecture 2 : Neural Classifiers**  
> 1. Finish looking at word vectors and word2vec  
> 2. Optimization basics  
> 3. Can we capture the essence of word meaning more effectively by counting?  
> 4. The Glove model of word vectors  
> 5. Evaluating word vectors  
> 6. Word senses  
> 7. Review of classification and how neural nets differ  
> 8. Introducing neural networks



## Word2vec의 parameter와 computations


word2vec의 특징은 corpus 속의 word를 **u와 v 두개의 벡터로 나타내는 것**이었다.


**(context word & center word)**


간단하게 형태를 살펴보면 밑의 형태를 보일 것이다.


![0](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/0.png)


다른 모델인 **“Bag of words”**은 각 position에 대해 같은 예측을 내리는데, 


이는 context에 따라 word의 뜻이 바뀌는 것을 표현할 수 없다.


반면 word2vec의 경우 context에 대해 고려하기 때문에, 아래와 같이 vector들을 2차원에 표시했을 때, 비슷한 단어들 끼리 그룹을 형성하는 것을 볼 수 있다.


![1](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/1.png)_신기하네요… (samsung과 nokia라니 재학습이 필요할듯…)_


지난 Lecture에서 gradient descent 과정에 쓰이는 미분과정을 알아 보았는데


이번 Lecture에선 본격적으로 **Gradient Descent**에 대해 알아봅니다.



### Gradient Descent


지난번에, 우리는 $L(\theta)$와 $J(\theta)$를 다음과 같이 정의했었다

- **data likelihood:** $L(\theta) = \Pi^T_{t=1}\Pi_{-m\leq j \leq m, j\ne0}P(w_{t+j}|w_t;\theta)$
- **objective function(cost, loss):** $J(\theta) = -\frac{1}{T}logL(\theta)$

(T는 총 word 수)


수백가지의 parameter가 존재하기 때문에, **cost function인** $J(\theta)$의 global minimum은 우리가 알 수 없다. 대신, 임의의 점(초기화된 parameter의 값)에서 시작해서 매 순간순간 그래프에서 minimum으로 향하는 방향으로 점을 이동해, **local minimum**은 구할 수 있다.


parameter들은 다음 식처럼 update된다.


![2](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/2.png)


![3](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/3.png)


하지만 이 방식 역시 **단점이 존재**한다.


$J(\theta)$는 모든 window 안의 corpus를 대상으로 하기 때문에, $\theta$의 수는 거의 billions에 가까울 정도로 많고, 그렇기 때문에 $\triangledown_\theta J(\theta)$**를 구하는 것은 cost가 너무 expensive**


그렇기 때문에 나온 개념이 **Stochastic Gradient Descent**



### Stochastic Gradient Descent


전체 data 단위로 update하는 것이 아닌, 


**batch 단위로 나눠 여러번에 걸쳐 update를 진행**


**window 단위 update가 일어남**


![4](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/4.png)


$\triangledown_\theta J_t(\theta)$는 sparse하게 일어남


왜냐하면, word2vec에서 matrix는 unique vocab의 수에 영향을 받는데,


window 단위로 update를 한다면, window 안에 포함된 unique vocab의 수는 matrix 크기 대비 굉장히 적을 것


만약 centor word 양 옆으로 m개의 word를 고려한다 했을 때, $\triangledown_\theta J_t(\theta)$는 오직 2m+1개에서만 일어나기 때문에, 성능이 좋아짐


![5](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/5.png)


![6](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/6.png)



## Word2vec details


앞에서, word2vec은 2가지 vector, centor word와 context(outside) words를 사용한다고 밝혔다.


근데 **왜 2가지일까?**


→ **easier optimization**


**(오히려 1가지 vector만 사용하는 것이 더 복잡함)**


이 2가지 벡터를 사용했을 때, **2가지 모델이 존재**



#### 1. **Skip-grams (SG)**


**→ Predict context(outside) words / given center word**



#### 2. **Continuous Bag of Words (CBOW)**


**→ Predict center word / given context(outside) words**


이 외에도 **추가 training method**가 있는데


**Negative sampling**이라고 한다.


(지금까지 배워온 naive softmax는 간단하나, cost가 많이 드는데, **negative sampling은 좀 더 efficient**)



### Negative sampling


이 개념의 메인 아이디어는 다음과 같다.


> 🧮 **Negative sampling**  
>   
> binary logistic regressions for a true pair (center word and a word in its context window) versus several noise pairs (the center word paired with a random word)


즉, **1) 원래의 center word와 context word의 쌍**과, **2) centor word와 의미 없는 word들의 쌍**을 **binary classification**으로 비교한다는 것


objective function을 알아보자


$$
J_t(θ) = log~σ(u_o^Tv_c) + Σ_{i=1}^k E_{j\simeq P(w)}[log~σ(-u_j^Tv_c)]
$$

- $σ(x) = \frac{1}{1+e^{-x}}$

	0,1 사이 정규화, activation function, binary classification에 fit

- t : time step
- log 연산 이유? 연산시 합산 연산으로 바꿔주기 때문
- k : negative sampling 수
- P(w) : unigram distribution. (3/4 power)
- j : P(w)에서 sampling된 word (negative training example; k개 존재)

쉽게 나타내면 다음과 같다


$$
J_{neg-sample}(u_o, v_c, U)=-log~σ(u^T_ov_c)-\Sigma_{k∈\{K~sampled~indices\}}~log~σ(-u^T_kv_c)
$$

- k : negative sampling 수 (word probability 이용)
- $P(w) = U(w)^{3/4}/Z$, unigram distribution을 3/4 제곱함

	(이렇게 3/4 제곱하는 것은 덜 자주 나타나는 단어가 더 자주 sampled 되게 함)


	(negative sample이 올바르게 선택될 수 있도록)


**수식을 나눠서 살펴보자**

- $-log~σ(u^T_ov_c)$ : **target word와 context word 사이의 유사성을 측정**

	→ model이 target word가 주어졌을 때, context word에 대해 높은 확률을 예측할 수 있도록 함


	(cost function을 minimize해야 하므로, target과 context가 유사하면 cost function 작아)

- $\Sigma_{k∈\{K~sampled~indices\}}~log~σ(-u^T_kv_c)$ : **negative sample과 centor word 간의 유사성 측정**

	→ 주어진 context word에 대해 negative sample에 대해 낮은 확률을 예측할 수 있도록 


	→ negative sample과 유사하다면 cost function이 커져야 한다.


	→ 따라서 negative sample과 유사할수록 $\Sigma_{k∈\{K~sampled~indices\}}~log~σ(-u_k^Tv_c)$는 작아진다. 


		(절댓값이 큰 음수가 됨)


	![7](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/7.png)_logarithmic sigmoid_


	→ 결과적으로 **negative sample과 유사할수록 cost function이 커지기 때문**에, gradient descent 과정에서 더욱 negative sample과 멀어지는 방향으로 학습이 진행된다.


	(negative sample을 잘 선정해야 하므로, 여기서 3/4 제곱 개념이 추가된 것)


In natural language processing, unigram distribution refers to the probability distribution of single words in a corpus. It is often used in language modeling and can be calculated by counting the frequency of each word in the corpus and then normalizing the counts to get probabilities. In some cases, the probabilities may be adjusted to account for the fact that less frequent words are more likely to be sampled as negative examples in models like word2vec.



### Example


![8](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/8.png)


→ 옆의 표는 co-occurence vector (동시 출현 vector) 들을 matrix로 정리한 것


지금의 예시에선, **corpus의 수, vocab의 수가 작아**


→ co-occurence matrix의 크기가 작기 때문에, 학습에 드는 비용이 적다.


하지만 실생활에선, 수많은 vocab이 존재하기 때문


→ co-occurence matrix 너무 **high dimensional해짐**


→ 학습 비용 크고, 오히려 학습 효과 좋지 않음


(too sparse, less robust)


주로, 25-1000 dimension을 유지하는 것이 좋기 때문에, dimension을 줄여야 겠다는 목적이 등장


→ 이를 위해 사용한 개념이 **SVD(Singular Value Decomposition)**


---



#### SVD (Singular Value Decomposition)


> 💡 **SVD**  
> Factorizes X into $U\Sigma V^T$, where U and V are orthonormal


ChatGPT 피셜


$U, \Sigma, V $ 모두 다양한 용도로 사용


![9](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/9.png)


---


다시 돌아와서, co-occurence matrix는 단지, corpus에서 window 안에 centor word에 대해 context word로 몇 번 등장하는지를 check해 matrix에 store 하는, 그저 **raw counts 만을 기록**하는 matrix


효과적이지 않음

- **High dimensionality**

	: sparse, high dimension → poor result

- **Noise**

	: word embedding에 악영향


	(몇번 등장하지 않는 단어 즉, noise 들은 관계 설정에 악영향 가능)

- **Scale**

	: 문제 있대요~


 때문에, 몇몇 fix를 하기도 함

- count 조절
	- log the frequencies
	- min(X, t) with t = 100
	- Ignore the function words
- ramped windows

	가까이 있는 단어를 더 크게 카운트

- Pearson correlations instead of counts

	count의 방식을 아예 바꿈


이렇게 등장한 방식이 **COALS**


![10](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/10.png)


(하지만 여전히 Count based)



### Count based vs. direct prediction


GloVe → word vectors 


linear algebra based methods on co-occurence matrices


→ ex. LSA, COALS

- Fast training
- Efficient usage of statistics
- Primarily used to capture word similarity
- Disproportionate importance given to large counts

models like skip-gram, CBOW


→ iterative neural updating algorithm

- Scales with corpus size
- Inefficient usage of statistics
- generate improved performance on other tasks
- Can capture complex patterns beyond word similarity

GloVe uses encoding

- ratios of co-occurence probabilities can encode meaning components
- weighted co-occurence matrices


### GloVe


GloVe algorithm의 가장 큰 특징은, 비슷한 context 속에서의 words 간의 relationship을 확인할 수 있다는 점


예시를 살펴보자


![11](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/11.png)


words 간 관계성이 크다면, probability는 large, 아니라면 small일 것이다.


GloVe algorithm은 이를 활용한다.


![12](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/12.png)


log-bilinear model을 사용하는데,


$w_i·w_j = log~P(i|j) $로 나타낼 수 있는 것은, dot product를 사용해, P(k|j)를 근사하겠다는 뜻임


→ dot product를 통해 P(k|j)를 represent하려고 한다.


이 핵심 아이디어에서 알고리즘이 시작함


$$
w_i·w_j = log~P(i|j)
$$


$$
J = \Sigma^V_{i, j=1}f(X_{ij})(w_i^T\tilde w_j + b_i + \tilde b_j - log~X_{ij})^2
$$

- fast training
- Scalable to huge corpora
- Good performance even with small corpus and small vectors

b는 bias를 의미


~는 median을 의미


그렇기 때문에, wiwj와 logX_ij를 같게 근사시켜야 하기 때문에, 같을 수록 J는 줄어들고, b는 조정을 위해 필요한 값인 것이다.


![13](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/13.png)


f의 역할은, 더 많이 co-occurence matrix로 count되는 단어에 더 큰 가중치를 부여하겠다는 것이고, 일정 수준 이상으로 넘어가는 너무 많이 나오는 단어들(ex. function words)에 대해선 일정치까지만 count 하겠다는 뜻이다.


learn vector representations for each word in the vocabulary such that the dot product of two word vectors is proportional to the log co-occurrence probability of the corresponding words.


GloVe 방식 훌륭함



### Evaluation


How to evaluate in nlp?


2가지 방법 존재


**intrinsic vs extrinsic**

- **Intrinsic**
	- Evaluation on a specific/intermediate subtask
	- Fast to compute
	- Helps to understand that system
	- Not clear if really helpful unless correlation to real task is established

a:b의 관계성을 c:?에 대해 적용해 ?(d)를 찾는 것


ex. man:woman :: king:? 


$$
d = arg~max_i\frac{(x_b-x_a+x_c)^Tx_i}{||x_b-x_a+x_c||}
$$


sol) ?는 queen일 것


x_a, x_b, x_c는 a, b, c에 대한 word vector임


b와 a의 뺄셈 연산으로 얻은 벡터를 c에 더하는 방식으로 확인


argmax_i는 이 수식을 최대화할 수 있는 i를 선택하겠다는 뜻임


ex1) GloVe Visualizations


![14](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/14.png)


ex2)


![15](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/15.png)

- **Extrinsic**
	- Evaluation on a real task
	- Can take a long time to compute accuracy
	- Unclear if the subsystem is the problem or its interaction or other subsystems
	- If replacing exactly one subsystem with another improves accuracy → Winning!

![16](/assets/img/2023-06-29-CS224n---Lecture-2-(Neural-Classifiers).md/16.png)


word2vec에선 한 word를 여러개의 sense를 가진 word로 나눔

