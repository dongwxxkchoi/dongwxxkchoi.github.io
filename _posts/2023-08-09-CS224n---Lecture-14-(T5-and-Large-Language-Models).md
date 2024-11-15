---
layout: single
date: 2023-08-09
title: "CS224n - Lecture 14 (T5 and Large Language Models)"
use_math: true
author_profile: false
tags: [강의/책 정리, ]
categories: [AI, ]
---


## T5 : **Text-To-Text Transfer Transformer**


![0](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/0.png)


같은 색깔로 이루어진 text는 모두 같은 task로 처리된것을 의미합니다. 즉, <u>**T5는 하나의 framework를 통해 여러 task를 처리**</u>할 수 있습니다.


---



### T5 vs BERT


![1](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/1.png)

- BERT : Encoder만 사용하기 때문에 Classification, Span prediction에 특화됨
- T5 : 모든 NLP task 수행 가능

---



### **T5 - architecture**


![2](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/2.png)


T5는 위와 같이 transformer encoder-decoder 구조를 가지고 있습니다. Encoder, decoder 모두 multi-head self-attention과 feed forward network 및 residual skip connection, dropout 기법을 사용하고 있습니다. 그리고 input token에 relative positional encoding을 사용했다는 점이 다릅니다. Relative positional encoding이란 input으로 들어오는각 token의 위치 별로 동일한 encoding 값을 부여하고 attention 계산을 하는것이 아니라, self-attention 계산할 때 일정 범위 내의 token들에 <u>**relative positional encoding**</u> 값을 줍니다


![3](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/3.png)


---


Transfer-learning을 위한 중요한 요소로는 pre-train에 사용되는 unlabeled corpus dataset입니다. Pre-train할 때 data크기를 키우는 효과를 정확하게 파악하려면 좋은 품질의 data와 여러 domain의 dataset이 필요합니다. 기존 Wikipedia는 품질은 우수한 data지만, 다양성이 부족하고 Common Crawl web scrapping data는 크기가 크고 다양하지만 품질이 낮다는 단점이 존재합니다. 그래서 <u>**google에서는 Wikipedia보다 2배 큰 새로운 Common Crawl 버전인 C4**</u>(Colossal Clean Crawled Corpus)를 사용했습니다. 그리고 unlabeled data에 대한 여러 실험을 위해 여러 기준에 따라 전처리를 진행했습니다.


---



### T5 - Pre-training


![4](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/4.png)


Pre-train에 쓰일 data가 있으므로 그 다음 필요한것은 pre-train에 쓰일 목적함수입니다. 먼저 위와같이 original text가 있을 때, 무작위로 token을 설정하고 masking 처리합니다. BERT와 많이 유사하지만 다른 부분은 <u>**하나의 random token을 masking 하는것이 아닌 연속된 token(span)을 하나로 masking 처리**</u>합니다. 또한 <u>**해당 token을 [MASK]로 변환하는것이 아닌 sentinel ID token으로 대체**</u>합니다. 마지막으로 input에서 masking되지 않은 부분을 target에서 맞춰야합니다.


![5](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/5.png)



### T5 - **Baseline experimental procedure**


![6](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/6.png)


먼저 BERT size의 encoder와 decoder 구조의 transformer를 340억개의 token으로 pre-train을 진행합니다. 훈련시간은 BERT에 비해 약 4분의 1정도라고 합니다. 이렇게 pre-train된 model을 가져와 각 downstream task에 맞게 fine-tuning을 진행합니다. 최대 170억개의 token을 fine-tuning한다고 합니다. 그리고 validation set에 대한 체크포인트를 평가하여 가장 좋은 성능을 보인 model을 찾습니다.


![7](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/7.png)



### T5 - **Experiments**



#### 1st Experiment - model


![8](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/8.png)


첫번째 실험은 서로 다른 모델들을 비교하는 것이었습니다.

1. encoder는 fully visible하게 attention을 진행하고 decoder의 경우 causal한 방법으로 attention을 진행하는 bi-directional한 **encoder-decoder** 모델
2. decoder만 존재하는, regressive한 성격을 가지고 있는 GPT와 같은 **LM**
3. encoder에서는 auto-encoder 형식을, 그리고 decoder에서는 auto-regressive 형식을 이용한 **Prefix LM**

![9](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/9.png)


위의 표를 보면 <u>**encoder-decoder의 구조를 가진 model이 성능이 제일 좋은**</u>것을 알 수 있고 LM의 성능이 좋지 않았습니다. 결국, bi-directional한 구조를 가진 model이 성능이 좋은것을 알 수 있습니다.



#### 2nd Experiment - dataset


![10](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/10.png)


기존 수집한 C4와 unfiltered C4 그리고 여러 다른 dataset을 가지고 pre-train한 후 여러 task에 대해 성능을 비교했는데 <u>**C4를 사용했을 때 성능이 제일 좋았습니다.**</u>



#### 3rd Experiment - size of data


![11](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/11.png)


T5는 다양한 task를 가지고 있고 얼만큼 학습하는것이 성능이 좋은지에 대한 실험입니다. Equal의 경우는 task에 상관없이 모두 같은 사이즈의 data를 학습시키는 방식입니다. K는 threshold로 기본적으로 data의 size만큼 학습하고 K이상의 데이터는 K만큼 학습합니다.


---


![12](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/12.png)


다양한 방식의 조합으로 훈련을 진행하여 성능을 비교했습니다. Pre-training을 진행하고 fine-tuning을 진행한 방식이 제일 성능이 높았습니다.


---



### **mT5 - What about all of the other languages?**


![13](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/13.png)


지금까지는 영어로 pretrained된 model인 T5에 대해서 얘기했고 그렇다면 다른 languages에는 어떻게 적용할까? 그래서 최근에 <u>**다국어 T5 model인 mT5**</u>를 도입했다고 합니다.


Text-to-text 형식은 그대로 갖고 가지만 input과 output이 다른 언어로 나올 수 있습니다. 그렇다면 이에 적절한 pre-train dataset이 필요한데, 그래서 101개 languages가 포함된 C4 data를 이용했습니다. 또한 질이 좋은 data를 더 얻기 위해 Common Crawl dump에서 data를 추출했습니다.


![14](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/14.png)


위 그래프를 통해 language 분포도를 확인할 수 있습니다. 각 language의 양이 많이 차이가 나는것을 확인할 수 있습니다. 이러한 차이를 좁히기 위해 <u>**scaling**</u>을 진행했다고 합니다.


![15](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/15.png)



### **How much knowledge does a language model pick up during pre-training?**


![16](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/16.png)


Model이 pre-train하면서 얼마나 많은 지식을 습득할 수 있을까? 


이번에는 T5가 reading comprehension에서 어떤 성능을 보여주는지 입니다. Reading comprehesion이란 질문이 주어지고 질문에 대한 답을 찾을 수 있는 <u>**knowledge가 주어졌을때, model이 답을 찾을수 있는지에 대한 task**</u>입니다.



### **Closed-book Question Answering**


Open-book은 질문과 지문이 주어지고 질문에 맞는 지문을 찾고 model이 해당 지문에서 정답을 찾는 task입니다. Closed-book은 대용량의 data를 학습한 model에 질문을 input으로 넣었을 때, output으로 정답이 나오게되는 task입니다. 이때 model이 얼마나 지식을 잘 기억하는지가 중요합니다.


![17](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/17.png)


---


![18](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/18.png)


위의 표는 natural question, web question, trivial QA에 대한 성능을 보여줍니다. Open-domain에 대한 SOTA model과 closed-book으로 훈련된 T5를 비교했습니다. <u>**Model의 크기가 증가할수록 학습하는 지식의 양이 많아지기 때문에 성능이 증가하지만, SOTA에는 미치지 못한것**</u>을 확인할 수 있습니다.



### Salient span masking


이러한 격차를 좁히기 위해 <u>**salient span masking**</u>을 도입했습니다. 


Salient span masking이란 pre-train할 때 random하게 masking을 하는것이 아닌 특정 entity에 masking을 하는것입니다.(사람 이름, 장소, 날짜, ...) 훈련을 하기전 entity recoginzer를 통해 entity를 파악하고 무작위 범위를 채우는 것이 아닌 두드러진 범위를 채우도록 model을 훈련합니다.


![19](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/19.png)


![20](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/20.png)



### **Do large language models memorize their training data?**


대량의 data를 학습할 경우 여러가지 문제가 발생할 수 있는데 그 중 lecture에서는 원하지 않은 <u>**개인정보가 training dataset에 포함**</u>될 수 있다고 했습니다.


![21](/assets/img/2023-08-09-CS224n---Lecture-14-(T5-and-Large-Language-Models).md/21.png)


예를 들어, GPT같이 대량의 web기반의 data로 학습된 model이 개인의 신상정보를 generate할 수 있는 문제점이 있습니다. 현재 이러한 이슈가 굉장히 중요한 issue가 된다고 합니다.

