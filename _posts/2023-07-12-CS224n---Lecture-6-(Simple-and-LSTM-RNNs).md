---
layout: single
date: 2023-07-12
title: "CS224n - Lecture 6 (Simple and LSTM RNNs)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---


## RNN Language Model


![0](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/0.png)_대략적인 RNN 모델_


RNN을 부분별로 나눠서 보자면, 크게

- **Input**
- **Embedding**
- **Hidden state**
- **Output**

로 볼 수 있음


**<Input>**


![1](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/1.png)

- $x^{(t)} \in \mathbb{R}^{|V|}$  **(V: Vocab, t: Timestep)**
- **word vectors** $x_t$는 각각 **T개의 words에 대응**
- 각 $x^{(t)}$은 **각 time step이 t**일 때 **input되는 word**를 의미
- $x^{(t)}$는 **모델이 알고 있는 Vocabulary**에 속함

**<Embedding>**


$e^{(t)} = Ex^{(t)}$


input으로 받은 **x를 embedded vector로 embedding**해주는 수식

- $E$: **Embedding Matrix**, $e^{(t)} : x^{(t)}$의 **embedded vector**

<**hidden state>**


$h^{(t)} = \sigma(W_hh^{(t-1)}+W_ee^{(t)} + b_1)$


time-step **t에 hidden layer output features를 계산**하는 수식

- $\sigma$: **non-linearity function** (ex. Sigmoid function)
- $W_h \in \mathbb{R}^{D_h\times D_h}$ : t-1 time-step의 hidden state에 곱해지는 weight matrix
- $h^{(t-1)}$ : t-1 time-step의 hidden state vector
- $W_e \in \mathbb{R}^{D_h\times d}$  : input embedded word vector $e^{(t)}$에 곱해지는 weight matrix
- $e^{(t)} \in \mathbb{R}^d$ (embedded vector는 dx1 vector)
- $b_1$ : hidden layer의 bias

<**Output>**


$\hat y^{(t)} = \textrm{softmax}(Uh^{(t)} + b_2) \in \mathbb{R}^{|V|}$


이 모델의 경우는, **앞의 t개의 words를 통해, 앞으로 나올 word를 예측** 


그 **word**는 **V에 속해 있는 word 중 하나**이고, **softmax를 통해 확률 분포 반환**

- $\hat y^{(t)}$ : 다음 predicted word ($x^{(t)}, h_{t-1}$를 통해 예측)
- $softmax$ : softmax function
- $U \in \mathbb{R}^{|V|\times D_h}$ :  hidden state vector $h^{(t)}$를 |V|로 mapping 해주는 weight vector (learned)
- $b_2$ : output layer의 bias

![2](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/2.png)



## Loss Function


**Loss Function**으로는 **Cross Entropy Loss**를 종종 사용함


![3](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/3.png)

<details>
  <summary>cross entropy loss</summary>


[bookmark](https://velog.io/@rcchun/머신러닝-크로스-엔트로피cross-entropy)



  </details>
![4](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/4.png)


식을 잘 살펴보면, **t 시점의 Loss Function**인데, 그 결과가 $x_{t+1}$**과 관련**이 있는 것을 볼 수 있음


**why?**


cross entropy loss는 확률이 가장 높은 것을 1로 만들고 나머지는 모두 0으로 만들어버리기 때문에,


확률이 높다고 판단되는 $\hat y^{(t)}_{x_{t+1}}$과 동일한 것


![5](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/5.png)_cross entropy error over a corpus of size T / eq(8)_


![6](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/6.png)



## Back Propagation for RNN 


![7](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/7.png)


![8](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/8.png)


![9](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/9.png)


![10](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/10.png)



## Perplexity

<details>
  <summary>**what is perplexity?**</summary>


[bookmark](https://wikidocs.net/21697)



  </details>
**perplexed : 헷갈리는**


→ **Perplexity** : “**헷갈리는 정도**” 라고 이해하면 편함


그렇기 때문에, Perplexity 즉, **PPL이 높을수록 언어 모델의 성능은 낮다.**


**(PPL 낮을수록 언어 모델 성능 좋음)**

<details>
  <summary>**In** **CS224n**</summary>


**standard 한 evaluation metric**


$Perplexity = 2^J$ is called the perplexity relationship; it is basically 2 to the power of the negative log probability of the cross entropy error function shown in Equation 8. 


Perplexity is a measure of confusion where **lower values imply more confidence in predicting the next word in the sequence** (compared to the ground truth outcome).


![11](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/11.png)



  </details>
![12](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/12.png)

- **W: sentence**
- **len(W): N**

이 문장의 확률에 chain rule 적용시 다음과 같음


![13](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/13.png)_n-gram의 예시_


![14](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/14.png)_bigram의 예_

<details>
  <summary>**PPL의 분기 계수(Branching factor)로써의 의미**</summary>


**PPL**은 **선택할 수 있는 가능한 경우의 수를 의미**하는 분기계수


이 언어 모델이 **특정 시점에서 평균적으로 몇 개의 선택지를 가지고 고민하고 있는지**를 의미


언어 모델에 어떤 테스트 데이터를 주고 측정했더니 PPL이 10이 나왔다


해당 언어 모델은 테스트 데이터에 대해서 **다음 단어를 예측하는 time step마다 평균 10개의 단어를 가지고 어떤 것이 정답인지 고민**한다는 뜻


![15](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/15.png)


그렇기 때문에, PPL이 낮을수록 **테스트 데이터 상에서 높은 정확도** (실제엔 적용 전)


사람이 느끼기에 좋은 언어 모델은 아니라는 점


언어 모델의 PPL은 테스트 데이터에 의존하므로 두 개 이상의 언어 모델을 비교할 때는 정량적으로 양이 많고, 또한 도메인에 알맞은 동일한 테스트 데이터를 사용해야 신뢰도가 높다는 것입니다.



  </details>
![16](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/16.png)


**n-gram 모델에 비해, RNN, LSTM으로 내려갈수록 성능 상승 확인**



## Advantages, Disadvantages and Applications of RNNs 



### Advantages & Disadvantages



#### **advantages**

1. They can process **input sequences of any length**
2. The **model size does not increase** for longer input sequence lengths
3. Computation for step t can (in theory) use information from many steps back.
4. The **same weights are applied** to every timestep of the input, so there is **symmetry** in how inputs are processed


#### **disadvantages**

1. **Computation is slow** - because it is **sequential**, it cannot be parallelized
2. In practice, it is difficult to access information from many steps back due to problems like **vanishing and exploding gradients**, which we discuss in the following subsection


### Vanishing Gradients and Exploding Gradients


**Example**


<Sentence 1>
"Jane walked into the room. John walked in too. Jane said hi to ___"
<Sentence 2>
"Jane walked into the room. John walked in too. It was late in the day, and everyone was walking home after a long day at work. Jane said hi to ___"


이 두 경우에, **RNN 모델은 2보다는 1을 더 잘 예측함**


→ Vanishing Gradient

<details>
  <summary>**Vanishing Gradient**</summary>


[image](https://youtube.com/clip/UgkxZrFO6O6HKEbn_7Q-__yz8i2Ybhkutox1)



  </details><details>
  <summary>**Exploding Gradient**</summary>


[image](https://youtube.com/clip/UgkxXC7Ywe0jbrT0VTMhRT1aKylR7sPTE2Nb)



  </details>
**Solution**

- **fix vanishing gradient**

	**→ LSTM (seperate memory)**

- **fix exploding gradient**

	→ **Gradient clipping**


	**norm of the gradient is greater than some threshold, scale it down before applying SGD update**



### **Applications**



#### 1) sequence tagging


![17](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/17.png)


**각 word에 대해 품사 태깅**



#### 2) sentence classification


![18](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/18.png)_general_


![19](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/19.png)_general_


![20](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/20.png)_for sentence classification_



#### 3) language encoder module


![21](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/21.png)


**question answering, machine translation …**


sequence를 받아 한 ouptut을 내놓으니, encoder?



#### 4) generate text (Decoding)


![22](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/22.png)


raw data를 받아 기계어를 내놓으니 decoder?



#### 5) Generating text with a RNN Language Model


![23](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/23.png)


repeated sampling → **output을 예측 sample로 사용**



#### 6) Translation


![24](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/24.png)

<details>
  <summary>**encoder & decoder**</summary>


![25](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/25.png)



  </details>

## Bidirectional RNNs



#### 등장 배경


지금까지의 sequence data 학습에서의 목표


→ **지금까지 주어진 것을 보고 다음을 예측**


하지만, 다음의 경우는?


![26](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/26.png)


i)  happy, sad, angry 등등


ii)  not, very 등등


iii)  very 등


하지만, 기존의 sequence 모델에서는 똑같이 “**I am”** 만 확인한다.


→ 이러한 문제를 해결하기 위해서 **Bidirectional RNN 모델**이 제안됨



#### 구조


위와 같은 문제점을 해결하려면, forward가 아닌 backward 방식이 필요하다. 즉, 뒤에서부터 sequence를 읽는 작업이 필요한 것. 


⇒ forward, backward를 위한 RNN 모델이 동시에 존재


![27](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/27.png)_Sentiment Classification_


위와 같은 상황에서, **네모친 부분의 hidden state(**$h$**)**는 **“the movie was terribly”**까지에 대한 정보만 포함하기 때문에, h는 negative 정보만 담게 된다.


하지만, 실제로는 뒤에 exciting이 붙기 때문에, sentence 전체의 의미는 positive여야 함


![28](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/28.png)


따라서 Bidirectional RNN을 설계해서, “the movie was terribly exciting !”을 감정분석할 때, 왼쪽에서부터, 그리고 오른쪽에서부터 모두 탐색해서, 해당 네모친 부분의 **concatenated hidden states(**$h^\prime$**)**은 두 문맥상 의미를 모두 담게 된다.


간단하게 도식화하면 이렇게 표현할 수 있다.


![29](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/29.png)


수식으로 표현하면 아래와 같이 나타낼 수 있다.


![30](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/30.png)_ppt 속 수식_


![31](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/31.png)_note 속 수식 (more detailed)_



### Deep Bidirectional RNNs


![32](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/32.png)


![33](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/33.png)


![34](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/34.png)


이렇게 여러 개의 층을 쌓음으로, deep bidirectional RNNs은 더 뛰어난 성능을 보여주지만, computational resources를 더 많이 요구하고, train이 어렵다는 단점도 존재


(활용분야: nlp, speech recognition, computer vision, audio processing…)



## Gated Recurrent Units (GRU)


RNNs은 **더 복잡한 activation function**을 사용했을 떄 더 좋은 성능 보임


affine transformation(기하학적 성질 보존 변환법)과 point-wise nonlinearity를 사용한 hidden state ht-1에서 ht로의 transition을 고안


RNN이 더 long-term dependency를 가질 수 있도록 persistent한 memory를 갖을 수 있게 함


GRU가 ht-1과 xt을 어떻게 사용해 다음 hidden state ht를 만드는지 설명


![35](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/35.png)


![36](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/36.png)







#### Reset Gate


![37](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/37.png)

- $r^{(t)}$ : **reset signal (리셋 신호)**

	**이전 time step의 hidden state** $h^{(t-1)}$ 이 **summarization (new memory)**인 $\tilde h_t$**에 얼마나 중요한지**를 결정합니다. reset gate에서 $h^{(t-1)}$ 이 **새로운 메모리 생성에 관련이 없다** 판단하면**,** $h^{(t-1)}$**을 무시하도록 신호를 생성**합니다. 


	(0~1 사이의 값, 0에 가까울수록 관련 X, 1에 가까울수록 관련 O)

- $U^{(r)}$ : $h^{(t-1)}$ 를 **처리**하는 **가중치 행렬** (학습 O)
- $W^{(r)}$ : $x^{(t)}$를 **처리** 하는 **가중치 행렬** (학습 O)


#### New memory generation


![38](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/38.png)

- $x^{(t)}$ : **새로운 input words**
- $h^{(t-1)}, h^{(t)}$ : **t-1, t timestep의 hidden state**
- $U$ : $h^{(t-1)}$ 를 **처리**하는 **가중치 행렬** (학습 O) (reset의 U와 다름)
- $\tilde h^{(t)}$ **: 새로운 memory**
- tanh : input을 **[-1,1] 사이의 값으로 반환**해 $\tilde h^{(t)}$를 생성

	(평균을 0으로 맞추고, hidden state의 분산을 제어하기 좋음; 경험적 이유도 있음)



#### Update Gate


![39](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/39.png)

- $z^{(t)}$ : **update signa**l로 $h^{(t-1)}$이 **다음 state로** **얼마나 많이 전달되어야 하는지**를 결정함

	예를 들어, $z^{(t)} $**이 1에 가깝다면,** $h^{(t-1)}$**은 거의 모두** $h^{(t)}$**로 복사**됨, **0에 가깝다**면 대부분의 **새로운 메모리** $\tilde h^{(t)}$**가 다음 hidden state로 전달**됨

- $U^{(z)}$ : $h^{(t-1)}$ 를 **처리**하는 **가중치 행렬** (학습 O)
- $W^{(r)}$ : $x^{(t)}$를 **처리하는 가중치 행렬** (학습 O)


#### Hidden state


![40](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/40.png)





- $h^{(t)}$ : 이전 hidden state $h^{(t-1)}$ 과, update gate의 영향을 받은 새로운 memory $\tilde h^{(t)}$를 받아 더해서 생성됨

**중요한 점**


update해야 할 parameter가 많음


→ $W, U, W^{(r)}, U^{(r)},W^{(z)},U^{(z)}$



## Long Short-Term Memory RNNs (LSTMs)


> 💡 **hidden state → hidden state +** **cell state** **(read, erase, write)**


![41](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/41.png)_LSTM_


![42](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/42.png)_RNN_


**차이점**


→ **cell state (with many gates)**


![43](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/43.png)


![44](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/44.png)



#### Forget gate


![45](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/45.png)

- $f^{(t)}$ : **과거 메모리 셀** $c^{(t-1)}$ 이 **현재 메모리 셀** $c^{(t)} $의 계산에 유용한지 여부를 평가. $h^{(t-1)}$와 $x^{(t)}$를 각각 처리해 dot product 해준 후, **sigmoid**를 통해 0~1 사이의 **확률값(forget할 정도)**($f^{(t)}$**)으로 반환**해 $c^{(t-1)}$ 와 element-wise product

![46](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/46.png)



#### Input gate


![47](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/47.png)

- $i^{(t)}$ : **reset signal (리셋 신호)**

	$x^{(t)}$와 $h^{(t-1)}$을 통해 $x^{(t)}$을 **preserve할 가치가 있는지 여부**를 결정.


	이 **정보의 지표(**$i^{(t)}$**) (기억할 확률 반환)**


![48](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/48.png)



#### New memory generation


![49](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/49.png)

- $\tilde c^{(t)}$**:** $x^{(t)}$와 $h^{(t-1)}$를 tanh를 통해 계산한 값에 preserve할 정도인 $i^{(t)}$를 element-wise product해 **새로운 메모리** $\tilde c^{(t)}$를 **생성**

![50](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/50.png)



#### Final memory generation


![51](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/51.png)

- $c^{(t)}$: $i^{(t)}$**를 반영한** $\tilde c^{(t)}$**와** $f^{(t)}$**를 반영한** $c^{(t-1)}$**을 더해서 최종 메모리 셀인** $c^{(t)}$**를 생성**

![52](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/52.png)



#### Output/Exposure Gate


![53](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/53.png)

- $o^{(t)}$: final memory cell인 $c^{(t)}$는 hidden state $h^{(t)}$에 저장할 필요가 없는 많은 정보를 포함. $c^{(t)}$ 중 어떤 부분이 hidden state $h^{(t)}$에서 exposed 되어야 하는지의 확률($o^{(t)}$)에 대한 평가를 수행. tanh를 통과한 $c^{(t)}$와 element-wise product되어 최종적으로 $h^{(t)}$를 반환
- $h^{(t)}$: final hidden state를 의미

![54](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/54.png)

- **언제 tanh, sigmoid?**
	- tanh : input을 **[-1,1] 사이의 값으로 반환**해 평균을 0으로 맞추고, hidden state의 분산을 제어한 후 sigmoid를 통해 나온 확률과 element wise되어 결괏값을 생성할 때 사용

		(평균을 0으로 맞추고, hidden state의 분산을 제어하기 좋음; 경험적 이유도 있음)

	- sigmoid : LSTM에 특히, forget할 정도, input할 정도 등의 확률 값이 많이 필요한데, 이 때 sigmoid를 통해 확률을 반환
- **어떻게 vanishing gradients를 해결?**

	→ **preserve information** over many timesteps **(by memory cell)**


		(만약, $f^{(t)}=1, i^{(t)}= 0$ 이면 이전 정보가 완벽 preserved)


	→ 아예 vanishing/exploding gradient가 완벽하게 없다고는 할 수 없지만, **long-distance dependency를 쉽게 학습할 수 있도록 하는 모델**

- **real-word success**

	**2013-2015 dominant approach**


	(현재는 transformers 등이 dominant)

- **vanishing/exploding gradient**가 RNN 만의 문제?

	**아님**   


	→ feed-forward, convolutional 등등 많은 **deep neural network의 문제**   


	**solution?**


	→ add more direct connections 


	**ex) Residual connections → ResNet**


	![55](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/55.png)


	**레이어가 일부 정보를 추가하거나 수정하더라도 이전 정보를 보존하는 방식**


	→ 이를 통해 층이 깊어지더라도 그래디언트가 사라지지 않고 전달되어 학습이 잘 되도록 도움

	- **스킵 연결 skip connection**: 레이어의 출력값을 다음 레이어로 바로 전달하는 연결로, 출력값과 입력값을 더하는 방식으로 이루어집니다. 이전 레이어의 출력값을 현재 레이어의 입력값에 더함으로써, 이전 레이어의 정보가 그대로 보존되며 그래디언트가 잘 전달되게 됩니다.

	**ex) Dense connections → DenseNet**


	![56](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/56.png)


	**직접적으로 모든 layer를 future layers와 연결**


	**ex) Highway connections → HighwayNet**


	![57](/assets/img/2023-07-12-CS224n---Lecture-6-(Simple-and-LSTM-RNNs).md/57.png)


	resNet과 유사하지만, transformation layer가 dynamic gate에 의해 조절


	LSTMs에 영감을 받은 방식


RNNs 등은 계속 같은 weight matrix가 반복적으로 곱해지기 때문에, 반복 학습이 없다는 장점도 있지만, back-propagation에서의 vanishing gradient는 어쩔 수 없음

