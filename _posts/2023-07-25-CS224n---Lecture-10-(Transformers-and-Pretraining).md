---
layout: single
date: 2023-07-25
title: "CS224n - Lecture 10 (Transformers and Pretraining)"
use_math: true
tags: [강의/책 정리, ]
categories: [AI, ]
---


### What is Transformer?


Model pretraining three ways

1. Decoders
2. Encoders
3. Encoder-Decoders

📘**The byte-pair encoding algorithm : 정보 압축 알고리즘**

1. 하나의 Character로 시작한다. or end-of-word
2. Corpus of text를 이용해 인접한 문자끼리 subword를 만든다.
3. 반복
4. 새로운 vocabulary를 완성

EX. 

- taaaaasty → taa## aaa## sty, aaa## 가 강조하는 것으로 꽤나 쓰이는 것을 발견할 수도 있음.
- Transformerify → Transformer## , ify 아하 , ify는 접미사로 많이 쓰이는 구나

❗해당 강의에서 얘기하는 word는 full word일 수도 있고 subword일 수도 있다. 그런데 transformer는 self-attention operation시에 이러한 정보를 갖지 않는다. 


---



#### Model pre-training and embedding


> “ You shall know a word by the company it keeps “ J.R. Firsth 1957


❗embedding의 문제점은 주변을 고려하지 않는다는 것. 예를 들어 같은 단어에 대해서 의미가 달라지더라도 Word2vec embedding은 같을 것이다. 


![0](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/0.png)


즉, 문맥을 고려하지 않은 word embedding만을 pretraind하였지만 우리의 downstream task는 contextual aspect가 필요하다! 또한 많은 parameter가 randomly initialized된다.


**∴ NLP 네트워크의 모든 prameter에 대해 pretraining이 필요하다.** 

1. “Representation of language”
2. “Parameter initailizations” for strong NLP model
3. “Probability distributions” sample from

 


**Then, How to pretraining whole models?**


![1](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/1.png)


❓ RIsk overfitting on our data when they’re doing pretraining?


→ 언어는 매우 sparse 해서 거의 그렇지 않다. 워낙 데이터가 많아서~



#### Pretraining / Fine-tuning


Pretraining : 이전의 다른 태스크를 다루는 모델의 가중치를 활용하는 방법.


Fine-tuning : Pretraining한 가중치를 토대로 추가로 학습( 미세 조정 )하는 방법. ex) 추가 데이터 투입을 통한 weight 갱신


![2](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/2.png)



#### Stochastic gradient descent


pretraining의 minima에서 gradinent descent를 진행하며 fine tuning하는 것이 just work한다…


---



#### Pretraining for three types of architectures

1. Decoders
2. Encoders
3. Encoder-Decoders

![3](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/3.png)



#### Pretraining Decoder


![4](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/4.png)


![5](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/5.png)


backpropagation 과정을 통해 전체에 대한 fine-tuning 가능. But Linear layer wasn’t pretrained.


Contract is that it is defining probabiliy distributions. 


we don’t need to use it as a probability distribution(?)


![6](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/6.png)


![7](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/7.png)


As Generator( predict , like dialogue ) at fine tuning time. fine tune probability distribution.  


⭐`Unlike before, last layer is pretrained. Still fine tuning the whole thing. ‘



#### GPT ( Generative Pretrained Transformer )

- Transformer decoder with 12 layers.
- 768-dimensional hidden states, 3072-dimensional feed-forward hidden layers
- Byte-pair encoding with 40,000 merges
- Trained on BooksCorpus: over 7000 unique books.

⭐**How to finuend it?**


[https://eggplant-culotte-a36.notion.site/GPT-1-Improving-Language-Understanding-by-Generative-Pre-Training-8dfe31b8e4914ebeb6a64230cd9a4ecb](https://eggplant-culotte-a36.notion.site/GPT-1-Improving-Language-Understanding-by-Generative-Pre-Training-8dfe31b8e4914ebeb6a64230cd9a4ecb)


+) GPT-2 is just a bigger GPT, has larger hidden units, more layers.



#### Pretraining Encoder


 Benefit of Encoder is to get bidirectional context.



#### BERT : masked language modeling


[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)


![8](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/8.png)


![9](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/9.png)


![10](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/10.png)


기존 Bi-directional는 좌-우, 우-좌를 통해 문장 전체를 예측한다. BERT는 랜덤하게 문장의 15퍼에 대해 예측을 진행합니다. 그 중 1) 80퍼는 [MASK]로 바꾸고 2) 10퍼는 Random하게 바꾸고 3) 10퍼는 그대로 유지한다. ( 이 때 predict는 여전히 진행한다. ) 그리고 이를 예측하는 작업( MLM )으로 진행한다.

- BERT Pretraining : is expensive and was pretrained with 64 TPU.
- BERT Fintuning : is practical and common on a 1 GPU.

> Pretrain once, finetune many times


다양한 task에 대해 finetuning BERT의 결과


![11](/assets/img/2023-07-25-CS224n---Lecture-10-(Transformers-and-Pretraining).md/11.png)


! BERT > GPT in Bidirectional context. !


But , BERT는 각 토큰이 독립적이므로 sequence를 generate할 수 없다. 


🔗 **Extensions of BERT…**


*** 



#### Pretraining encoder-decoders : what pretraining objective to use?( T5 )


span corruption : 


***



### GPT 3, In-Context learning


learning “without gradient steps”, just to do pretraining. These models are still not well-understood.


whopping 175billion parameters!



#### In-Context Learning


프롬프트의 내용만으로 하고자 하는 task 를 수행하는 작업. 말 그대로 prompt 내 맥락적 의미(in-context)를 모델이 이해하고(learning), 이에 대한 답변을 생성하는 것


→ Zero-shot learning, One-shot Learning, Few-shot Learning

