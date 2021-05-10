---
layout: post
title: "4차과제 2번"
description: "다중 클래스 분류 알고리즘 구현하기."
date: 2021-05-11
comments: true
share: true
---

4차 과제2, 과제 1에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현하라. 단, 사이킷런을 전혀 사용하지 않아야 한다.

--- 
## 1. 데이터 준비하기

### 모듈 불러오기

```
import numpy as np
form sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
```

### 붓꽃 데이터셋의 특성 1가지와 한가지 품종을 선택

```
X = iris["data"][:, 3:]  # 1개의 특성 (꽃잎 너비)
y = (iris["target"] == 2).astype(np.int)  # 모든 품종의 붓꽃
```

### 모든 샘플에 편향을 추가

```
X_with_bias = np.c_[np.ones([len(X), 1]), X]
```

### 결과를 일정하게 유지하기 위한 랜덤 시드를 지정

```
np.random.seed(2042)
```

--- 
## 2. 데이터셋 분할

### 데이터셋 분할 비율 설정

```
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```

### 인덱스 무작위로 섞기

```
rnd_indices = np.random.permutation(total_size)
```

### 6:2:2 비율로 훈련, 검증, 테스트 세트를 분할

```
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```

--- 
## 3. 타깃 변환
타깃은 0, 1, 2로 설정되어 있다. 차례대로 세토사, 버시컬러, 버지니카 품종을 가리킨다. 훈련 세트의 첫 5개 샘플의 품종은 다음과 같다.

```
y_train[:5]
```
array([0, 1, 2, 1, 1])

학습을 위해 타깃을 원-핫 벡터로 변환해야 함. 이유는 소프트맥스 회귀는 샘플이 주어지면 각 클래스별로 속할 확률을 구하고 구해진 결과를 실제 확률과 함께 이용하여 비용함수를 계산하기 때문   

붓꽃 데이터의 경우 세 개의 품종 클래스별로 속할 확률을 계산해야 하기 때문에 품종을 0, 1, 2 등의 하나의 숫자로 두기 보다는 해당 클래스는 1, 나머지는 0인 확률값으로 이루어진 어레이로 다루어야 구현한 알고리즘이 계산한 클래스별 확률과 연결   

아래 함수 to_one_hot() 함수는 길이가 m이면서 0, 1, 2로 이루어진 1차원 어레이가 입력되면 (m, 3) 모양의 원-핫 벡터를 반환.   

```python
def to_one_hot(y):
    n_classes = y.max() + 1                 # 클래스 수
    m = len(y)                              # 샘플 수
    Y_one_hot = np.zeros((m, n_classes))    # (샘플 수, 클래스 수) 0-벡터 생성
    Y_one_hot[np.arange(m), y] = 1          # 샘플 별로 해당 클래스의 값만 1로 변경. (넘파이 인덱싱 활용)
    return Y_one_hot
```
샘플 5개에 대하여 잘 작동하는 것을 확인 가능

```
y_train[:5]
```
array([0, 1, 2, 1, 1])

```
to_one_hot(y_train[:5])
```
array([[1., 0., 0.],   
       [0., 1., 0.],   
       [0., 0., 1.],   
       [0., 1., 0.],   
       [0., 1., 0.]])   

이제 훈련, 검증, 테스트 세트의 타깃을 모두 원-핫 벡터로 변환

```
Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```

Y_train_ont_hot은 90x3 행렬, 각 열에는   
* 0 : 세토사(Iris-Setosa)
* 1 : 버시컬러(Iris-Versicolor)
* 2 : 버지니카(Iris-Virginica)
에 대한 정보가 true 일 경우 1로, false 일 경우 0으로 저장되어 있음.   
각 붓꽃의 품종에 대해 일대다(OVR)방식으로 로지스틱회귀를 취하기 위해선 각 열을 분리시켜 주어야 함

```
Setosa_train_one_hot = Y_train_one_hot[:,0]
Versicolor_train_one_hot = Y_train_one_hot[:,1]
Virginica_train_one_hot = Y_train_one_hot[:,2]

Setosa_valid_one_hot = Y_valid_one_hot[:,0]
Versicolor_valid_one_hot = Y_valid_one_hot[:,1]
Virginica_valid_one_hot = Y_valid_one_hot[:,2]

Setosa_test_one_hot = Y_test_one_hot[:,0]
Versicolor_test_one_hot = Y_test_one_hot[:,1]
Virginica_test_one_hot = Y_test_one_hot[:,2]
```

각 원-핫 벡터들을 mx1 행렬로 reshape, 이유는 행렬의 곱 연산을 할 때 행렬의 크기를 맞춰주기 위함

```
Setosa_train_one_hot = Setosa_train_one_hot.reshape(90,1)
Versicolor_train_one_hot = Versicolor_train_one_hot.reshape(90,1)
Virginica_train_one_hot = Virginica_train_one_hot.reshape(90,1)

Setosa_valid_one_hot = Setosa_valid_one_hot.reshape(30,1)
Versicolor_valid_one_hot = Versicolor_valid_one_hot.reshape(30,1)
Virginica_valid_one_hot = Virginica_valid_one_hot.reshape(30,1)

Setosa_test_one_hot = Setosa_test_one_hot.reshape(30,1)
Versicolor_test_one_hot = Versicolor_test_one_hot.reshape(30,1)
Virginica_test_one_hot = Virginica_test_one_hot.reshape(30,1)
```

--- 
## 4. 로지스틱 함수 구현

```python
def logistic_sigmoid(x): # 시그모이드 함수 정의
    return 1 / (1 + np.exp(-x))
```

--- 
## 5. 활용 훈련

```
n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))
```

### 0: 세토사(Iris-Setosa) 품종에 대한 조기 종료 활용

```
eta = 0.08
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta1 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta1)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Setosa_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta1[1:]]
    Theta1 = Theta1 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta1)
    Y_proba_veri = logistic_sigmoid(logits)
    xentropy_loss = -1/m*(np.sum(Setosa_valid_one_hot * np.log(Y_proba_veri + epsilon) + (1 - Setosa_valid_one_hot) * np.log(1 - Y_proba_veri + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta1[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```
0 0.1942305470183407   
500 0.0713123891503352   
694 0.07012518581596847   
695 0.0701251934308839 조기 종료!   

### 1: 버시컬러(Iris-Versicolor) 품종에 대한 조기 종료 활용

```
eta = 0.07
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005          # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta2 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta2)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Versicolor_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta2[1:]]
    Theta2 = Theta2 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta2)
    Y_proba_veri = logistic_sigmoid(logits)
    xentropy_loss = -1/m*(np.sum(Versicolor_valid_one_hot * np.log(Y_proba_veri + epsilon) + (1 - Versicolor_valid_one_hot) * np.log(1 - Y_proba_veri + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta2[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```
0 0.3724096545512761   
367 0.218993700746672   
368 0.21899372701315967 조기 종료!   

### 2: 버지니카(Iris-Virginica) 품종에 대한 조기 종료 활용

```
eta = 0.08
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005        # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta3 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta3)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Virginica_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta3[1:]]
    Theta3 = Theta3 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta3)
    Y_proba_veri = logistic_sigmoid(logits)
    xentropy_loss = -1/m*(np.sum(Virginica_valid_one_hot * np.log(Y_proba_veri + epsilon) + (1 - Virginica_valid_one_hot) * np.log(1 - Y_proba_veri + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta3[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```
0 0.30051317265766586   
500 0.11440446122769776   
1000 0.10241684411484213   
1500 0.09958112692021259   
1820 0.09926349977734265   
1821 0.09926350015409136 조기 종료!   

### 조기종료 검증세트에 대한 정확도 검사

```
logits_Setosa = X_valid.dot(Theta1)
logits_Versicolor = X_valid.dot(Theta2)
logits_Virginica = X_valid.dot(Theta3)

Y_proba_Setosa = logistic_sigmoid(logits_Setosa)
Y_proba_Versicolor = logistic_sigmoid(logits_Versicolor)
Y_proba_Virginica = logistic_sigmoid(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa, Y_proba_Versicolor, Y_proba_Virginica)) # 각 품종을 하나의 행렬로 합침

y_predict = np.argmax(Y_proba, axis=1)    # 가장 높은 확률을 갖는 클래스 선택

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```
0.9

--- 
## 6. 테스트 세트 평가

```
logits_Setosa = X_test.dot(Theta1)
logits_Versicolor = X_test.dot(Theta2)
logits_Virginica = X_test.dot(Theta3)

Y_proba_Setosa = logistic_sigmoid(logits_Setosa)
Y_proba_Versicolor = logistic_sigmoid(logits_Versicolor)
Y_proba_Virginica = logistic_sigmoid(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa, Y_proba_Versicolor, Y_proba_Virginica)) # 각 품종을 하나의 행렬로 합침

y_predict = np.argmax(Y_proba, axis=1)    # 가장 높은 확률을 갖는 클래스 선택

accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```
0.9333333333333333

--- 
이 문서는 [한글 Lorem Ipsum](http://guny.kr/stuff/klorem/)으로 생성되었습니다.
