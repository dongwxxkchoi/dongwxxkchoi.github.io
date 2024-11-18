---
layout: single
date: 2023-08-03
title: "CS224n - Lecture 13 (Coreference Resolution)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---


## 1. Coreference Resolution


> 💡 Identify all <u>**mentions**</u> that refer to the same entity in the word


![0](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/0.png)


위와 같이 <u>**같은 개체를 판별하는 작업**</u>을 Coreference Resolution라고 한다.


---



### Coreference Resolution in Two Steps


![1](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/1.png)

1. Detect the mentions : 말 그대로 <u>**mention을 찾는 작업**</u>이다. 그러므로 쉬운 과정이다.
2. Cluster the mentions : mention된 <u>**개체를 분류하는 작업**</u>이다. 언급된 부분이 어떤 개체인지 알아야 하므로 Detect the mentions을 하는 과정보다 어렵다.

---



## 2. Mention Detection



#### Mention의 종류


![2](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/2.png)

1. 대명사
2. 개체명
3. 명사구


#### How to Detect?


![3](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/3.png)

1. <u>**대명사는 POS로 구분이 가능**</u>하다. (POS는 문장 내 단어들의 품사를 식별하여 태그를 붙여주는 것) - POS Tagging
2. <u>**개체명은 앞에서 배운 NER을 통해 찾는다**</u>. (RNN을 통해 NER을 하는 과정을 이전에 배웠었다.
3. <u>**명사구는 parser을 통해 감지**</u>한다고 한다. (14주차에 배운다고 함)

---



#### Bad mentions


![4](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/4.png)


이는 위의 3가지 조건에 들어가지만 mention이라고 말하기 어렵다.



#### How to deal with these bad mentions?

- 안좋은 mention을 거를 수 있도록 classifer를 학습 시킨다.
- 하지만 classifier를 학습시키는 단계를 건너 뛸 때가 많은데 그 이유는 그냥 진행시켜도 <u>**특정한 entity를 나타내지 않는 mention들은 혼자서 분류되기 때문**</u>에 시스템에 나쁜 영향을 끼치지 않는다.

---



#### Avoiding a traditional pipeline system


> 💡 POS tagger, NER, parser를 하나하나 따로 사용하는 것이 아니라, 이러한 일을 한번에 할 수 있는, mention을 발견해내는 classifier를 학습시킬 수 있나?


→  <u>**Yes**</u> 
mention detection과 coreference resolution을 2 단계로 나누지 않고 <u>**end-to-end**</u>로 한번에 진행한다.


---



#### On to Coreference! First, some linguistics


![5](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/5.png)

- mention이 같은 entity를 나타낸다면 <u>**coreference**</u>라고 한다.
- 문장 속에서 앞에 나온 단어를 가리키는 것을 <u>**anaphora**</u>라고 한다. (뒤에 나오는 단어가 앞에 나오는 단어에 의해서 해석 될 때)

---



#### Anaphora vs. Coreference


![6](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/6.png)


![7](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/7.png)

- 위 사진의 예시 문장에서 concert와 tickets은 서로 다른 entity를 이야기 하고 있기에 coreference하다고 이야기할 수 없지만, ticket은 concert의 ticket을 뜻하기 때문에(ticket의 뜻이 앞에 나오는 concert라는 단어에 의해서 해석되기 때문에), concert와 ticket은 anaphoric한 관계에 있다고할 수 있다.


#### Cataphora


Cataphora는 Anaphora의 반대 말이다. Anaphora는 뒤에 나오는 단어의 의미를 앞에 나오는 단어에서 찾는다면, cataphora는 앞에나오는 단어의 의미를 뒤에서 찾는다.


최근에는 cataphra라는 개념은 잘 사용되지 않는다.



## 4. Hobbs’ naive algorithm : 대명사의 reference 찾기


![8](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/8.png)

- 여기서 우리는 him에 대한 reference를 찾고 싶다.
1. **him에 해당하는 NP에서 시작**한다.
2. tree를 거슬러 올라가서 **NP나 S를 찾는다.** (이 예시에서는 S를 찾을 수 있다)
3. 그 후 S를 기준으로 왼쪽에서 오른쪽으로 BFS식으로 tree를 훑어 내려가게 되는데, 이 때 NP를 발견하면 이 NP는 S와 NP 사이에 또 다른 NP나 S가 존재할 경우 him에 대한 reference 후보가 될 수 있다. (여기서는 NP와 S 사이에서 또 다른 NP나 S를 발견할 수 없기 때문에 him에 대한 reference가 될 수 없다.)
4. **한 문장을 다 살폈다면, 그 전의 문장을 tree를 BFS로 훑는다. 이때 발견된 NP는 him에 대한 reference 후보**가 된다.

→ 이 문장에서 him은 Niall Ferguson을 나타냄을 알 수 있다.



### Hobbs’ algorithm의 한계


![9](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/9.png)


이 두 문장은 같은 구조를 가지고 있지만 it이 가리키고 있는것은 다르다. 
이러한 경우에는 위에 설명한 <u>**Hobb's algorithm을 사용할 수 없다.**</u>


---


![10](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/10.png)


<u>**문장을 왼쪽에서 오른쪽으로 훑을 것인데, 이 때 새로운 mention을 발견할 때 마다 어떤 coreference인지 classify한다.**</u> 


![11](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/11.png)



#### Training


![12](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/12.png)


위 수식을 보면 loss funtion이 Binary Cross Entropy이다. mi와 mj가 coreferent하면 yij가 +1이 되므로 그 확률만큼 loss function이 줄어들고 아니면  -1로 loss function이 그 확률만큼 증가하는 형태이다.  



#### Test


![13](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/13.png)

- mention들을 pair로 묶어서 classifer에 집어 넣는다.
- Classifier의 결과가 어떠한 확률(임계값?)을 기준으로 yes 나 no 가 나올 것이다. 예를들어 0.5가 기준값이라면 0.5를 기준으로 coreference link를 결성할 것이다.
- 이 mention들을 clustering 하기 위해서 transitive closure 방법을 사용할 것이다. Trasitive closure란 A가 B와 coreferent 하고 B가 C랑 coreferent하다면 A와 C는 coreferent하다 라는 것을 의미한다.
- transitive closure를 사용하기 때문에 혹시나 실수가 일어나면 모든 mention이 하나로 묶일 수 있는 (over cluster) 위험성도 존재한다.
- coreferent를 이루지 않는 mention도 존재한다. 그렇다면 classifier에서는 모든 다른 mention에 대해서 no 값을 배출할 것이고 이 mention은 singleton mention이 될 것이다.


#### Disadvantage


![14](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/14.png)


만약 mention을 많이 포함하는 긴 문서가 있으면, 해당하는 mention을 모두 찾아내서 yes라는 값을 배출하는 것이 아니라 한 mention을 잘 표현하는 특정한 mention 하나를 찾아내고 싶다.


---



## 6. Coreference Models: Mention Ranking


![15](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/15.png)


말 그대로 she가 어떤 mention과 coreference한지 보고 싶을때 <u>**여러 mention 후보들을 ranking**</u>한다고 하여 Mention Ranking이다.


coreferent한 mention이 없을 경우가 있을텐데, 이 때를 위해서 NA값을 포함시킨다

- She와 다른 mention에 대한 pair에 softmax를 적용시킨다. 이 때 softmax를 적용시켜서 나온 값들의 합은 1이 될 것이다. 이때 우리가 원하는 것은 어떠한 antecedents에 대해서 높은 확률값을 가지게 되는 것이다. (prior reference 가 있다면 antecedent에 대해서 높은 확률값을 얻을 것이고, 없다면 NA와 높은 확률값을 얻을 것이다.)
- 그리고 가장 높은 값을 가진 것에 한하여 coreference link를 만든다.


#### Training


![16](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/16.png)


하나의 reference에 대해 확률을 모두 더한 후 negative log를 취한 후 document에 있는 모든 mention을 더한다.



#### How do we compute the probabilites?


1) Non-neural statistical classifier


2) Simple neural network


3) More advanced model using LSTMs, attention, transformers



### 1) Non-neural statistical classifier


![17](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/17.png)

- You have whole bunch of features and you had a feature based statistical classifier which gave a score. And the above are the features that we could use.
- Throw these all features into a statistical classifier and that's sort of 2000s decade coref systems as to how they're built.


### 2) Simple neural network


![18](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/18.png)


Word embeddings와 categorical features를 input값으로 하는 neural network를 구성해서 score값을 구한다.



## 7. End-to-end


![19](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/19.png)


![20](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/20.png)


![21](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/21.png)



#### Performance


![22](/assets/img/2023-08-03-CS224n---Lecture-13-(Coreference-Resolution).md/22.png)

