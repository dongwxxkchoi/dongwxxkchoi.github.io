---
layout: single
date: 2023-07-18
title: "CS224n - Lecture 9 (Self-Attention, Transformer)"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
---

> ✅ **Index**  
> 1. Problem of RNNs (Recurrent models)  
> 2. Review of Attention  
> 3. Self-Attention  
> 4. Structure of Transformer



## Problem of RNNs (Recurrent models)


**RNN의 특징 : 순차적 연산**

- 왼쪽 → 오른쪽
- present time step(t)의 hidden state → future time step(t+1)의 hidden state

![0](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/0.png)_chef와 was 사이 sequence에 대한 연산이 필요_


**가까운 단어들의 의미에 대해 영향**을 많이 주고 받아, word embedding에 결과가 잘 반영된다는 장점이 있지만, 단점 역시 존재하는데,

1. **Long distance dependency problem**

	input sequence의 길이가 길어질수록, 즉 **먼 거리의 단어일수록 상호작용이 어려움** 


	**gradient vanishing**의 문제로 dependency 학습이 어렵다

2. **Lack of Parallelizability**

	순차적 연산 때문에, **GPU를 활용한 병렬 연산이 불가능**하다. 


	특정 hidden state의 **연산을 위해** **이전 hidden state가 필요**


	**O(sequence length)**


	(recurrent model의 일반적인 문제)


![1](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/1.png)_해당 hidden state 연산되기 전 까지의 최소 연산 횟수_



#### → 그렇기 때문에, **Model이나 Data가 커질수록 더 큰 약점**이 됨



### How about word windows?


word window 방식도 local context를 aggregate 할 수 있다.


→ sequence length가 증가된다고 하더라도, 병렬처리가 불가능한 연산 증가 X


(한번에 여러 window에 대한 병렬 연산 가능)


![2](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/2.png)_해당 hidden state 연산되기 전 까지의 최소 연산 횟수_


하지만, long distance에 대한 **lack of dependency 문제는 해결되지 않음**


![3](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/3.png)


예를 들어 h1와 hk의 dependency를 반영하기 위해선 window layer를 높이 쌓아야 함


상호작용할 수 있는 최대 길이는 **전체 sequence의 길이 / window size** 


**분명 한계점이 존재**


→ long distance dependency : O


    lack of parallelizability :  X (not solved)


_“Attention은 이 두 가지 한계점을 모두 해결할 수 있는 solution”_



## Review of Attention


기존 Attention은 **decoder에서 encoder로 query**를 날려, **attention score를 통해 주목해야 할 부분을 파악**하는 mechanism이었습니다.


---


<From Lecture 7>



#### Attention Score


![4](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/4.png)


<u>**하나의 디코더와 각 인코더를 내적하여 스칼라 값을 구하면 그것이 바로 각각의 attention score**</u>이다. 


즉, attention score는 **현재 시점의 디코더의 정보**와 **인코더의 매 시점의 정보 간 유사도**를 의미



#### Attention Distribution


![5](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/5.png)


<u>**attention score를 softmax 함수에 통과시켜 생성한 확률 분포**</u>이다. 


위의 예시에서는 il에 가장 분포가 집중되었으므로 가장 먼저 he를 생성한다.



#### Attention Output


![6](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/6.png)


<u>**attention distribution을 가중치로 하여 인코더의 각 hidden state를 가중합**</u>한 것이다. 


attention output은 **높은 attention을 가진 hidden state의 정보를 포함**하고 있다.


![7](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/7.png)


마지막으로 <u>**attention output과 디코더의 hidden state를 결합하여**</u><u>_**y**_</u><u>**^를 산출**</u>한다. 


그리고 각 디코더에서 위의 과정을 반복한다. 


---


정리하자면,

1. 인코더 RNN을 거치면서 hidden state 생성 (recurrent)
2. 디코더를 거치면서 해당 time step의 decoder input과 이전 인코더 hidden state들 간의 dot product를 통해 attention scores 획득 (recurrent)
3. 해당 time step의 attention score들을 softmax를 거치면서 distribution을 구함
4. 가장 확률이 높은 output을 받아와 다음 단어를 예측

	(timestep 증가시키면서 2~4 반복)


이 과정에서 Attention은 **각 word’s representation을 query**로 여겨 **set of values의 정보에 접근하고 그들을 통합**한다.


여기서 **single sentence에 대한 attention**에 대해 생각해보자


number of unparallelizable operations는 sequence length를 증가시키지 않음


모든 words가 모든 layer에서 interact → O(1)


기존의 attention의 1, 2 과정이 recurrent한 것이 문제


→ self-attention은 이런 encoder, decoder 구조에서 벗어나 **자기 자신 안에서 attention 과정**을 수행하면서 **1, 2 과정을 parallelized하게 전환한 모델**입니다. 



## Self-Attention 


![8](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/8.png)


encoder, decoder로 이루어진 기존 attention과는 다르게, 


self 즉, **자기 자신에 대해서 attention을 수행**한다는 뜻입니다.


위에 사진을 보면, **한 문장인 input sequence 안에서 attention 과정이 수행**되는 것을 볼 수 있음


모든 words가 모든 layer에서 상호작용하므로, **maximum interaction distance**는 **O(1)**


그렇다면 이전 attention에서 수행되었듯이, query 역할을 했던 각 timestep에서의 decoder의 hidden state와 encoder의 hidden state 등등이 필요한데 한 문장 안에서 역할이 구분되어야 함


→ 여기서 나온 개념이 **query, key, value**

<details>
  <summary>chat gpt 설명</summary>


in self-attention, the query, key, and value vectors can be thought of as the elements that are used to compute the attention scores. Specifically, given a query vector, the dot product of the query vector with each key vector generates a set of attention scores, which are then used to weight the value vectors to produce the final output.


In other words, the attention mechanism can be viewed as a process of assigning importance to different parts of the input sequence (the values) based on how relevant they are to the current element being processed (the query), with the relevance being determined by the similarity between the current element and the other elements in the sequence (the keys).


assign importance to **different parts of the input sequence (value)**


based on how relevant they are to the **current element being processed (query)**


relevance being determined by the similarity between the current element and the **other elements in the sequence** **(key)**


The keys are used to retrieve the relevant information from the input sequence by computing a dot product between the keys and the query vector.


The resulting attention scores indicate the importance of each position in the input sequence with respect to the current output position. The values are then weighted by these attention scores and combined to produce the context vector, which contains the information that is relevant to the current output position.



  </details>
self-attention에 대해 이해하려면, query, key, value에 대한 이해를 해야 함.



### Query, Key, Value


이 세 요소는 모두 input sequence로 부터 계산되어 나오는 요소들


이 **세 요소 간의 computation을 통해서 self-attention 과정이 수행**됨


![9](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/9.png)


이 요소들은 attention 구조에서의 

- **Query - Decoder의 hidden state (현재 우리가 유추/처리하는 정보에 대한 hidden state)**
- **Key - Encoder의 hidden states (Query와 얼마나 연관되어있는지 체크해야할 Encoder의 LSTM셀의 hidden vectors)**
- **Value - Encoder의 hidden states (Key와 Query 간의 relevance를 측정해서 input sequence의 각 부분에 importance 부여)**

에 해당합니다.


이 세 요소는 **모두 같은 source로 부터 도출**됩니다.


sequence $\mathbf{x}_{1:n}$ 속 token $\mathbf{x}_i$ 이라 할 때, 

- **Query :** $\mathbf{q}_1, \mathbf{q}_2, ..., \mathbf{q}_T, ~~~\mathbf{q}_i \in \mathbb{R}^d, ~~ \mathbf{q}_i = Q\mathbf{x}_i, ~~~Q\in\mathbb{R}^{d\times d}$

이고, 각 token $\mathbf{x}_j \in \{x_1, ...~, x_n\}$ 에 대해서,

- **Key :** $\mathbf{k}_1, \mathbf{k}_2, ..., \mathbf{k}_T,~~~\mathbf{k}_j \in \mathbb{R }^d, ~~~\mathbf{k}_j = K\mathbf{x}_j,~~~K\in\mathbb{R}^{d\times d}$
- **Value :** $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_T,~~~ \mathbf{v}_j \in \mathbb{R}^d,~~~ \mathbf{v}_j = V\mathbf{v}_j,~~~V\in\mathbb{R}^{d\times d}$

이다.


(d는 hyperparameter) (i가 query, j가 key)


![10](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/10.png)

1. $e_{ij} = q_i^Tk_j$는 **key-query 간 affinities** (**유사도**)
2. $\alpha_{ij}$는 **각** $\mathbf{v}_j$(**value**)의 **기여도**(strength of contribution)를 조절

	어떤 data를 주목해야 할지를 정하는 역할


	v_j가 존재하는 이유는 a_ij와는 다르게 j하나를 관통하는 value의 역할을 하기 때문


	V가 없다면, 해당 시퀀스 안에서의 직접적 연결에 대해서만 output이 연관됨. J를 관통하는 value가 있어야 이전 정보까지 다 아우를 수 있다

3. output인 contextual representation $\mathbf{h}_i$ of $\mathbf{x}_i$ 는 sequence의 values에 대한 linear combination (query에 대한 values)

![11](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/11.png)


$\alpha_{ij}$ : scalar,  $v_j$ : vector (**d x 1**)


![12](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/12.png)


$q_i,~~ k_j$ : vector (**d x 1**)


⇒ **h_i들을 concatenate해서 최종 output인 h 생성**


---


**Key와 Value 간의 관계?**


→ word2vec skip-gram의 **center vector - context vector 관계**와 일정 부분 유사


 


![13](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/13.png)


<word2vec skip-gram’s center & context>


center → word 그 자체


context → word가 나타나는 context


<self-attention’s key & value>


**key** → used to **match the query** with appropriate value


**value** → **information related to the input sequence**


---



### Sequence order problem - Position representations


지금까지 나온 정보로만 파악했을 때, 문제는 **sequence order를 나타낼 information이 없다는 것.** 


기존 attention의 경우는 이전 hidden state가 현재 hidden state에 반영되기 때문에 이런 order가 표현됐지만, 위에는 그런 역할을 수행하는 요소가 X



#### Position representations


example> _“the oven cooked the bread so”_


	            “_the bread cooked the oven so”_


![14](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/14.png)


![15](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/15.png)


$\alpha_{so, n}$ 는 **so가 query**일 때, **n번째 word에 대한 기여도** (index n)


$$
\alpha_{so, 0} = \frac{exp(\mathbf{q}_{so}^T\mathbf{k}_{the})}{exp(\mathbf{q}_{so}^T\mathbf{k}_{the})+...+exp(\mathbf{q}_{so}^T\mathbf{k}_{bread})}
$$


$$
\alpha_{ij} = \frac{exp(\mathbf{q}_i^T\mathbf{k}_j)}{\Sigma^n_{j^\prime=1}exp(\mathbf{q}_i^T\mathbf{k_{j^\prime}})}
$$


$\alpha \in \mathbb{R}^5$는 weight, 이를 통해 $\mathbf{h}_{so}$를 계산함


연산과정에서 봤듯이, order에 대한 고려 아예 x



#### **sol** 


1) use vectors that are **already position-dependent as inputs**


$P\in\mathbb{R}^{N\times d}$, N은 maximum length of any sequence


($p_i \in \mathbb{R}^d$, for $i\in\{1, 2, ..., T\}$ → **position vectors)**


**self-attention block에** $p_i$**를 더해서 사용**


![16](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/16.png)


2) **change the self-attention operation itself**


→ $\alpha$ **자체를 바꾸는 방법**


![17](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/17.png)


$\mathbf{k}_{1:n}\mathbf{q}_i \in \mathbb{R}^n$ are the original attention scores (key와 query 곱)



### Elementwise nonlinearity


**stack self-attention layers → stacked LSTM layers로 가능?**


![18](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/18.png)


No. 위와 같이 단순히 두 self-attention layer를 쌓는다고 해서 non-linearity를 만들어 낼 수 없음 → 한 self-attention layer와 차이가 없다


한 self-attention layer 이후에, 독립적으로 activation 함수를 먹여야 함


![19](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/19.png)


$W_1 \in \mathbb{R}^{5d\times d}, W_2 \in \mathbb{R}^{d\times 5d}$



### Future masking


**autoregressive modeling**


training 과정에서 **미래의 input을 확인할 수 없도록** masking (decoder)


일반적인 Language Modeling에서, word를 다음 상황에서 예측을 한다.


![20](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/20.png)


$f$ : sequence를 $R^{|V|}$로 **mapping 해주는 함수**


일반적인 모델에선 예측을 수행할 때 미래를 볼 수 없다.


![21](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/21.png)


![22](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/22.png)


⇒ -$\infin$를 넣어주면 나중에 softmax 통과 시 0이 됨 (exponential)


후에 transformer encoder-decoder 구조가 등장하는데, encoder가 아닌 decoder에 future masking 기법 사용. 


![23](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/23.png)



### Summary


1) self-attention


2) position representations


3) elementwise nonlinearities


4) future masking (in LM) 



## Structure of Transformer


2023; most used architecture in NLP → Transformer


Transformer → stacked Blocks로 구성되어 있는 self-attention 구조에 기반한 모델


![24](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/24.png)


![25](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/25.png)


**encoder**


→ input sequence를 인코딩해 fixed-length vector representation으로 나타내는 것


**decoder**


→ encoder에서 생성된 token과 내부에서 이전 self-attention을 통해 생성된 token을 통해 output sequence token을 생성하는 것이 목적


4가지 특징

- **multi-head** self-attention
- **layer normalization**
- **residual connections**
- **attention scaling**


### Multi-head Self-Attention


![26](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/26.png)

- 같은 input
- **key, query, value를 나누어** **각기 다른 key, query, value**를 정의
- **self-attention을 동시에 여러 번 진행 (parallelization)**
- **output을 concatenate**

![27](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/27.png)


![28](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/28.png)

- $h$개의 heads
- $K^{(l)},Q^{(l)},V^{(l)} \in \mathbb{R}^{d\times d / h}$,    $l $ in $\{1, ..., h\}$
- head 별로 key, query, value matrix  $\mathbf{k}^{(l)}_{1:n},\mathbf{q}^{(l)}_{1:n},\mathbf{v}^{(l)}_{1:n}$ 각각 존재

![29](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/29.png)


$\mathbf{h}_i = O[\mathbf{v}_i^{(1)};...~; \mathbf{v}_i^{(h)}]$  ($O \in \mathbb{R}^{d\times d}$)


$\mathbf{h}_i^{(l)}$ 는 $\frac{d}{h}$ dimension이므로, 이들을 최종적으로 concatenate해서 최종 $\mathbf{h}_i$를 구함


$\mathbf{h}_i$를 concatenate해서 최종적으로 $\mathbf{h}$ 생성



#### Sequence-tensor form


각 head output의 축소된 차원을 얻는 과정


single head의 경우, $\mathbf{x}_{1:n} $  in  $\mathbb{R}^{n\times d}$

- value vectors → $\mathbf{x}_{1:n}V$
- key vectors → $\mathbf{x}_{1:n}K$
- query vectors → $\mathbf{x}_{1:n}Q$

![30](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/30.png)


(query $\cdot$ key)


![31](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/31.png)


![32](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/32.png)_single-head operation_



#### **이걸 multi-head attention한다고 생각해보자**

1. $\mathbf{x}_{1:n}Q,\mathbf{x}_{1:n}K,\mathbf{x}_{1:n}V$를 $\mathbb{R}^{n,h,d/h}$ 형태로 reshape 해야 함 (d → h x d/h 로)
2. matrices를 $\mathbb{R}^{h,n,d/h}$로 transpose (n, h, d/h → h, n, d/h)

	(h sequences, n length, dimension d/h)


	→ **head를 마치 batch 처럼 사용**하는 효과 ($\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i$)


	→ 각각이 lower-rank에서 연산됨


![33](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/33.png)_multi-head operation_



### Layer Normalization


→ model이 더 빠르게 train할 수 있도록 돕는 기법

- hidden vector values에 있는 **uninformative한 variation을 줄여 더 안정적인 input을 전달**
- **각 layer** 에서의 **normalization**(mean & standard deviation)

$x \in \mathbb{R}^d$ 을 모델 각각의 word vector라고 하자


$\mu = \Sigma^d_{j=1}x_j~~~\mu\in\mathbb{R}$ : mean


$\sigma = \sqrt{\frac{1}{d}\Sigma^d_{j=1}(x_j-\mu)^2}~~~\sigma\in\mathbb{R}$ : standard deviation


$\gamma\in\mathbb{R}^d$  **gain** parameters


$\beta\in\mathbb{R}^d $  **bias** parameters


![34](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/34.png)_𝜖은 zero division 피하기 위함; 새롭게 Output 정의_



#### Residual Connections


layer의 input을 그 layer의 output에 추가해주는 것


**⇒ skip-connection**


**기존**


![35](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/35.png)


$X^{(i)} = Layer(X^{(i-1)})$


**Residual Connections**


![36](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/36.png)


$X^{(i)} = X^{(i-1)} + Layer(X^{(i-1)})$


![37](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/37.png)_other representation_


**효과**


1) add 연산은 **gradient vanishing 문제로부터 자유롭다**


	→ 언제나 **local gradient가 1**


	→ **더 깊은 network**를 구성할 수 있음


2) 증분변환을 통해 **새롭게 randomly initialized layers로 학습하지 않고도 좋은 학습 가능**


	→ 증분변환: 변환이 연속적으로 일어나는 경우에 매번 계산을 다시 하지 않고, 직전의 변환으로 얻어진 값에 비교적 간단한 계산을 적용하여 변환


![38](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/38.png)


**Add**


![39](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/39.png)


**→ pre-normalization (better)**


**Norm**


![40](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/40.png)


**→ post-normalization**



### Attention logit scaling


**dimension이 커짐에 따라, dot product의 결과도 커지는 경향**이 있다.


→ 이를 scaling 해주는 것


기존 Transformer의 최종 output은 softmax로 산출


![41](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/41.png)


**softmax의 값이 한 label로 쏠린다**는 것


→ 특정 단에만 강하게 attention이 가해짐


→ **input 변화에 따른 output 변화가 작아지는 것이므로, gradient가 작아지는 효과**


→ 학습에 악영향


그러므로 gradient를 잘 전파시키기 위해서 dot product의 결과가 너무 커지지 않도록 scaling을 취함


![42](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/42.png)



#### Transformer


![43](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/43.png)



### Transformer Encoder


![44](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/44.png)


(future masking 사용 x)

1. single sequence $\mathbf{w}_{1:n}$ input 받음
2. Embedding $E$  →  $\mathbf{x}_{1:n}$
3. Add Position Embedding (position vector 추가)
4. multi-head attention
5. Add & Norm (Residual connection + Layer Normalization)
6. Feed-Forward network
7. Add & Norm (Residual connection + Layer Normalization)

이 과정을 거쳐 encoding이 수행됨


**Uses of the Transformer Encoder**


Transformer Encoder는 autoregressively하게 텍스트를 생성하는 상황이 아닐 때 좋은 성능을 바뤼함


전체 sequence에 대해 강한 representation을 원할 때 in



### Transformer Decoder


![45](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/45.png)


(future masking 사용)

1. single sequence $\mathbf{y}_{1:m}$ input 받음
2. Embedding $E$
3. Add Position Embedding (position vector 추가)
4. Masked multi-head attention
5. Add & Norm (Residual connection + Layer Normalization)
6. Multi-head Attention
7. Add & Norm (Residual connection + Layer Normalization)
8. Feed-Forward
9. Add & Norm (Residual connection + Layer Normalization)
10. Linear
11. Softmax

**Uses of the Transformer Decoder**


→ GPT-2, GPT-3, BLOOM



### Transformer Encoder-Decoder


**input을 two sequence 받음**



#### cross-attention


**한 sequence에서 key와 value**를, **다른 sequence에서 query를 정의**


![46](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/46.png)


(Encoder의 output hidden state 이용해서 key, value 정의)


$\mathbf{h}^{(y)}$ of sequence $\mathbf{y}_{1:m}$ 


![47](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/47.png)


즉, query는 y, key와 value는 x


![48](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/48.png)_위: query, 아래: key, value_



#### **Application**

- **Language Modeling**: In language modeling, there is only one sequence, which is the input sequence. Therefore, cross-attention is not used in this task.
- **Text Classification**: In text classification, the two sequences used in cross-attention are the **input sequence and a fixed set of class embeddings**, which represent the possible output classes. The class embeddings are learned during training and are used to help the model attend to relevant information in the input sequence for the given classification task.
- **Question Answering**: In question answering, the two sequences used in cross-attention are **the input sequence and a set of question embeddings**. The question embeddings are learned during training and are used to help the model attend to relevant information in the input sequence for answering the given question.
- **Summarization**: In summarization, the two sequences used in cross-attention are the encoder output sequence and the decoder input sequence. The encoder output sequence contains the **contextual representation of the input sequence**, while the **decoder input sequence contains the summary generated so far**. The cross-attention mechanism allows the decoder to attend to relevant parts of the input sequence while generating the summary.


### Great Results with Transformers


![49](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/49.png)


**machine translation**


![50](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/50.png)


**document generation**



### 개선점?


![51](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/51.png)


**sequence length T에 따른** $O(T^2d)$**의 급격한 상승이 문제**


**⇒ 해결책: Linformer**


![52](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/52.png)


**⇒ 해결책: BigBird**


![53](/assets/img/2023-07-18-CS224n---Lecture-9-(Self-Attention,-Transformer).md/53.png)

