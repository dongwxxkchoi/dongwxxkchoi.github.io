---
layout: single
date: 2023-07-15
title: "CS224n - Lecture 7 (Translation, Seq2Seq, Attention)"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
---


### Machine Translation


<u>**Machine Translation이란 특정 언어의 문장을 다른 언어의 문장으로 번역하는 것을 말한다**</u>. 
이때 input(x)으로 들어간 언어의 문장을 source language, output(y)으로 나오는 언어의 문장을 target language라고 한다.



### Statistical Machine Translation


주요 아이디어는 <u>**데이터로부터 확률 모델을 학습**</u>하는 것이다.


프랑스어 → 영어로 번역을 한다고 가정하자.


![0](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/0.png)


_**P**_**(**_**x**_**∣**_**y**_**): Translation Model (Bilingual Corpus)**

- 두 개의 언어 모두로 표현된 데이터인 Parallel 데이터를 통해 확률을 계산
- y는 기존 영어 문장보다 짧은 구나 단어
- 영어 문장이 주어졌을 때 프랑스어 문장의 확률 분포 생성

_**P**_**(**_**y**_**): Language Model (Mono-lingual Corpus)**

- 단일언어 데이터인 Monolingual 데이터를 통해 확률 계산
- y의 문장이 얼마나 영어 문장으로서 자연스러운지를 확률분포를 통해 표현


#### _**P**_**(**_**x**_**∣**_**y**_**)는 어떻게 학습?**


우선 엄청나게 많은 양의 parallel data가 필요합니다. 그리고 이 데이터로부터 _P_(_x_∣_y_)를 학습시킬 때 사용되는 개념이 바로 alignment이다.


![1](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/1.png)


![2](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/2.png)


![3](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/3.png)


![4](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/4.png)


![5](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/5.png)


---


![6](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/6.png)


> 💡 alignment는 다양한 경우의 수가 존재하기 때문에 <u>**특정 단어가 특정 단어에 정렬될 확률, 특정 단어가 fetility할 확률 등 고려해야 할 사항이 많아 매우 복잡하다.**</u>


![7](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/7.png)


![8](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/8.png)



#### **SMT의 단점**

- 성능은 좋지만 구조가 복잡: 각 system은 분리된 많은 subcomponent들로 이루어졌다.
- 많은 feature engineering이 필요하다.
- 추가적인 많은 자료가 필요하다.
- 유지보수를 위해 사람의 노력이 많이 필요: 각각의 언어쌍(영어-프랑스어, 영어-독일어 등)을 만들어야 해서 많은 노력이 필요하다.

---



### Seq2Seq



#### Neural Machine Translation(NMT)

- 하나의 신경망 네트워크를 이용한 Machine Translaion
- 두 개의 RNN을 사용한다.(인코더, 디코더에 하나씩)

![9](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/9.png)


위 그림과 같이 NMT는 두개의 RNN으로 이어져있다.


<u>**Encoder부분에서는 단어 정보들을 학습하고 Decoder 부분에서 RNN의 many-to-many구조를 이용한다.**</u>


**Encoder**

- 번역할 프랑스어 문장 입력
- <u>**각 단어의 임베딩 벡터가 각 시점마다 입력값으로 사용**</u>된다.
- 소스 문장을 마지막에 인코딩시켜 디코더에 넘긴다.
- RNN은 bidirectional rnn, LSTM, GRU 등 모두 가능하다.

**Decoder**

- 인코더의 마지막 hidden state에서 넘어온 프랑스어 정보와 문장의 시작을 의미하는 start 토큰을 입력 받아 <u>**다음에 나올 단어의 확률분포 argmax를 취한다.**</u>
- <u>**디코더의 출력값은 다음 디코더의 입력값**</u>이 된다.
- 조건부 언어 모델
- end 토큰을 출력하면 종료된다.

![10](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/10.png)


NMT는 조건부 언어 모델이다.

- 언어 모델: 디코더가 target sentence인 y의 다음 단어를 예측
- 조건부: 디코더에 source sentence인 x가 주어지면 y를 뽑고, 첫 번째 y와 x를 이용해 두 번째 y를 뽑는 과정을 반복


### How to learn?


![11](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/11.png)


먼저 본래 병렬 말뭉치에 존재하는 이전 시점의 영어 단어를 고정된 입력값으로 디코더에 넣어준다. <u>**디코더 RNN의 모든 단계에서 교차 엔트로피 등으로**</u> <u>_**y**_</u><u>**^의 손실을 계산한 후, 손실들의 평균을 낸 최종 Loss를 구한다.**</u>


이렇게 손실값부터 모델의 입력값까지 한 번에 역전파가 일어나는 방식을 end-to-end라고 하고, end-to-end는 시스템을 전반적으로 최적화한다고 할 수 있다.



#### 번외) end-to-end?


> 💡 모델의 <u>**모든 매개변수가 하나의 손실함수에 대해 동시에 훈련되는 경로가 가능한 네트워크로써 역전파 알고리즘 (Backpropagation Algorithm) 과 함께 최적화 될 수 있다**</u>는 의미이다.


---



#### 다음은 디코더에서 사용하는 디코딩의 여러 방식에 대해 소개하겠다.



### 1. Greedy Decoding


![12](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/12.png)


greedy decoding은 위의 NTM예시에서 사용한 디코딩 방식이다. 
디코더의 각 step에서 argmax를 취하는 것을 greedy decoding이라고 하는데, greedy decoding으로 인해 문제가 발생한다.


![13](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/13.png)


위의 예시와 같이 <u>**greedy decoding은 한번이라도 틀린 단어로 예측한다면 되돌아 갈 수 없다.**</u> 위의 예시에서 a로 잘못 예측되었는데 이를 수정할 수 없어 틀린 문장으로 계속 이어진다.



### 2. **Exhaustive Search Decoding**


![14](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/14.png)


우리는 위의 확률을 최대화하는 y를 찾는 것이 목표이다. exhaustive search decoding은 <u>**모든 경우의 수를 다 계산해서 그 중 최댓값을 갖는 y를 찾는다.**</u> 정해진 시퀀스 길이 T에 대해 각 시점마다 일어날 수 있는 모든 토큰 조합의 확률을 계산하고, 이 중 최대값을 가지는 토큰 조합을 최종 생성 문장으로 선택한다. 하지만 이 방법은 <u>**모든 경우의 수를 다 계산하기 때문에 시간복잡도가 너무 크다.**</u>



### 3. **Beam Search Decoding**


beam search decoding의 핵심 아이디어는 <u>**디코더의 각 step에서 가장 가능성이 높은 k개의 번역을 선택하여 가지치기를 하는 것이다.**</u> 이때 k는 beam size를 의미하고, k가 커질수록 각 단계에서 더 많은 것을 고려한다.


![15](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/15.png)


각 가설은 로그 확률로서 위의 식과 같은 score를 가진다. t시점까지 생성된 문장 후보 y의 score는 언어모델을 이용해 계산한 조건부 확률의 로그값이다. 따라서 그럴듯한 문장일수록 0에 가깝다. 우리는 여기서 높은 score를 갖는 k개의 가설을 각 step마다 선택할 것이다. beam search decoding은 최고의 성능을 보장하지는 않지만, 모든 경우의 수를 탐색하는 exhaustive search decoding보다는 훨씬 효율적다.


![16](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/16.png)



### **Stopping Criterion**


Greedy decoding은 토큰이 나오면 종료한다. 하지만 Beam Search Decoding은 한 가설이 토큰을 생성하더라도 다른 가설은 계속 탐색을 이어간다. 따라서 <u>**최대 문장 길이 T를 설정하거나(Decision Tree에서 max_depth같은 느낌), n개의 완료된 가설이 생성되면 종료**</u>하도록 할 수 있다. 하지만 두 번째 방법에는 문제가 있다.


![17](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/17.png)


위 문장을 보면 각 확률들의 합으로 score가 계산되어 최대 문장 길이를 설정하는 방법으로 사용할 경우 <u>**길이가 길수록 더해지는 term이 많아지기 때문에 불리하다.**</u> 그래서 우리는 아래 수식과 같이 normalize를 해준다.


![18](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/18.png)



### **NMT의 장단점**


![19](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/19.png)



### 평가지표 : BLEU(Bilingual Evaluation Understudy)


BLEU는 Machine Translation를 평가하는 지표 중 하나이다.
이 지표는 <u>**기계에 의해 번역된 문장과 사람이 번역한 문장을 비교해서 유사도를 측정하여 성능을 측정한다.**</u>



#### **n-gram precision**


![20](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/20.png)


<u>**Unigram precision은 기계가 번역한 문장에 대해서, 사람이 번역한 문장에 기계가 번역한 unigram이 얼마나 등장하는지 비율을 계산**</u>한 것이다. 그런데 이 방식에는 문제가 있다.


![21](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/21.png)


위의 문장에서 기계가 번역한 문장인 candidate는 the가 반복되는 터무니 없는 번역지만, reference에 the가 등장해기 때문에 성능은 1로 나온다.


이를 개선하기 위해, 중복을 제거하여 식을 보정한다.


![22](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/22.png)


![23](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/23.png)


하지만 이 또한 문제가 존재한다.


![24](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/24.png)


위의 예시에서 ca2는 ca1의 모든 unigram의 순서를 랜덤으로 섞은 것이다. ca2는 문법에는 전혀 맞지 않지만 unigram precision은 ca1과 동일하다.


따라서 순서를 고려하여 n-gram으로 확장하게 된다.


![25](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/25.png)


n-gram은 다음에 등장하는 단어까지 함께 고려하여 카운트를 한다. 카운트 단위를 몇 개로 보느냐에 따라 2-gram, 3-gram 4-gram이 된다.


이를 바탕으로 나온 BLEU 식은 다음과 같다.


![26](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/26.png)


하지만 위의 식을 그대로 사용하면 Candidate의 문장이 짧으면 점수가 높게 나오는 문제가 발생한다. precision에서 분모가 Candidate의 count이기 때문이다.


이를 해결하기 위해 <u>**Candidate의 길이가 짧을 때 패널티를 주는 Brevity Penalty**</u>가 등장한다.



#### **Brevity Penalty**


![27](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/27.png)


candidate가 reference보다 길이가 길다면 패널티를 주지 않지만, 길이가 더 짧다면 패널티를 주어 BLEU 점수를 낮춰준다.


BLEU는 속도도 빠르고 활용성이 높지만 완벽하지는 않다. 단순히 단어 단위로 직역하지 않고 맥락을 파악하여 의역을 했을 때 엉뚱한 단어가 등장할 수 있는데, 이러면 BLEU score가 낮아지게 된다.


![28](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/28.png)


<u>**NMT는 등장한지 2년만에 SMT의 성능을 뛰어넘었다.**</u> 실제로 구글은 번역시스템을 SMT에서 NMT로 전환한 후 수백명의 엔지니어가 수년간 한 작업들을 소수의 엔지니어로 몇 달만에 더 나은 성과를 냈다.



### **여전히 남은 Machine Translation의 한계**

- Out of Vocabulary: 학습 데이터에 존재하지 않는 단어가 입력되면 목표 단어 생성 불가능하다.
- Domain Mismatch: train 데이터와 test 데이터 사이에 domain이 일치하지 않으면 성능 하락한다.
- Context: 긴 문장의 문맥을 유지하기 힘들다.
- Low-resource language pairs: NMT 학습을 위한 많은 양의 병렬 코퍼스 구축이 어렵다.
- 관용적인 표현이 잘 학습되지 않는다.
- 인터넷 상에 존재하는 문서들에는 암묵적인 사회적 편향이 담겨 있는데, MT는 이 데이터들을 그대로 학습하기 때문에 편향된 표현으로 번역한다.

![29](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/29.png)

- 왼쪽 문장은 단순히 ag를 반복한 해석이 불가능한 문장인데, MT는 이를 마음대로 해석해버다.

<u>**이러한 MT의 한계들을 보완하기 위해 나온 것이 바로 Attention이다.**</u>


---



### Attention


![30](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/30.png)


seq2seq는 인코더의 마지막 hidden state에 번역하고자 하는 문장의 모든 정보가 담겨 있어야 하기 때문에, 너무 많은 압력이 가해지게 된다. 즉 문장의 정보가 마지막 hidden state에서 다 인코딩 되어 버려 정보가 쏠리는 <u>**정보병목현상**</u>이 발생한다.


**Attention**은 디코더의 각 step에서 인코더에 대한 직접 연결을 사용하여 소스 문장의 특정 부분마다 집중함으로써 정보병목현상을 해결합니다.



### **1. Model Architecture**


![31](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/31.png)

- **attention score** : <u>**하나의 디코더와 각 인코더를 내적하여 스칼라 값을 구하면 그것이 바로 각각의 attention score**</u>이다. 즉 attention score는 현재 시점의 디코더의 정보와 인코더의 매 시점의 정보간 유사도를 의미다.

![32](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/32.png)

- **attention distribution** : <u>**attention score를 softmax 함수에 통과시켜 생성된 확률 분포**</u>이다. 위의 예시에서는 il에 가장 분포가 집중되었으므로 가장 먼저 he를 생성한다.

![33](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/33.png)

- **attention output** : <u>**attention distribution을 가중치로 하여 인코더의 각 hidden state를 가중합**</u>한 것이다. attention output은 높은 attention을 가진 hidden state의 정보를 포함하고 있다.

![34](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/34.png)


마지막으로 <u>**attention output과 디코더의 hidden state를 결합하여**</u><u>_**y**_</u><u>**^를 산출**</u>한다. 그리고 각 디코더에서 위의 과정을 반복한다. SMT는 완전 대응하거나 대응하지 않는 har binary이지만, attention은 softmax를 취해서 더 유연한 정렬을 할 수 있다.


아래는 Attention을 더 이해하기 쉬운 예시이다.


![35](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/35.png)


![36](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/36.png)


![37](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/37.png)


[bookmark](https://www.youtube.com/watch?v=WsQLdu2JMgI)



### 2. Attention: in equations


![38](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/38.png)



### **3. Advantages**

- NMT 성능을 비약적으로 향상시켰다.
- 병목현상을 해결했다. seq2seq에선 인코더의 정보가 마지막 hidden state에 집중되어 제대로 전달되지 못하는 병목현상이 발생했지만, <u>**attention은 디코더가 직접 인코더의 모든 시점에서 정보를 가져오게 함으로써 이를 해결했다.**</u>
- vanishing gradient problem을 완화했다. attention은 디코더와 인코더를 직접 연결한 구조이다. 이는 <u>**그래디언트가 인코더의 마지막 시점과 디코더의 첫 시점의 연결 뿐 아니라 인코더와 디코더의 각 시점으로 직접 흘러가도록 만들어 vanishing gradient problem을 완화했다.**</u>
- 모델을 어느정도 해설할 수 있게 만들어준다. attention score를 분석하면 모델이 각 시점마다 어디에 집중하고 있는지 알 수 있다.

![39](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/39.png)


![40](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/40.png)


![41](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/41.png)


왼쪽 사진은 seq2seq with attention에서 attention score를 시각화한 것이다. 오른쪽 사진은 본래 alignment이다. <u>**두 도표는 매우 유사한 것을 볼 수 있다.**</u> SMT의 alignment는 직접 사람이 작성해야 하는 수고가 발생하는 것에 비해 attention은 모델이 직접 구축하고, 보다 유연한 형태라는 점에서 더 우수하다고 할 수 있다.



### **4. Generalization**


attention은 단순히 seq2seq에만 사용되지 않고, 많은 모델에서 사용다. 그래서 좀 더 확장된 버전의 attention 정의가 필요하다. 원문을 가져오자면


> 💡 Given a set of vector values, and a vector query, attention is a technique to compute a weighted sum of the values,dependent on the query.


즉, <u>**벡터인 value들의 집합과 하나의 벡터인 query가 있을 때, attention은 query를 이용해 value들의 가중합을 구하는 방법론**</u>이다.


그리고 종종 논문 등에서 query attends to the values 와 같은 표현을 볼 수 있는데, 이것이 바로 attention 메커니즘을 설명하고 있는 것이다.

- attention output은 query가 집중하고자 하는 value의 요약된 정보입니다.
- attention은 고정된 벡터 사이즈를 통해 query가 value들의 정보에 접근하는 방식입니다.

![42](/assets/img/2023-07-15-CS224n---Lecture-7-(Translation,-Seq2Seq,-Attention).md/42.png)


[bookmark](https://velog.io/@stapers/Lecture-8-Translation-Seq2Seq-Attention)

