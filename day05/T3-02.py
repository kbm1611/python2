
# [1]
import pandas as pd
df = pd.read_csv('./day05/Fish.csv')
# 어종 7개, Species
fish_target = df['Species']
# 특성 6개, Weight, Length1, Length2, Length3, Height, Width
fish_input = df[ ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
# 훈련 / 테스트용 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( fish_input, fish_target, test_size= 0.25, random_state=42 )

# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit( train_input )
train_scaled = ss.transform( train_input )
test_scaled = ss.transform( test_input )

# [*] 경사 하강법
# fit() 모델학습에서는 정답(target)도 같이 학습 중이다. 예측(y)값과 실제 정답간의 오차( 기울기, 가중치 -> 오차 줄이기 가능) 측정
# 예] 산꼭대기에서 내려가는 방법 중에 가장 최적의 경로로 내려오는 방법 = 경사 하강법( 컴퓨터 사양이 좋아야함. 수많은 경우의 수 게산하여 판단하기 때문에)
# (1) 전통 경사 하강법( 정확도는 좋지만 학습속도가 느림 )
# (2) 확률 경사 하강법( SGD -> 정확도가 비교적 낮지만 학습속도가 빠르다. : 미니배치 )

# [*] 로그 로스 / 손실 함수 , 손실( 예측과 정답 차이 )
# 로그 로스 함수는 0과 1 확률 값과 오차 값을 측정

# [*] dpvhzm

# [2] SGDClassifier, 분류 모델
from sklearn.linear_model import SGDClassifier
# loss = 'log_loss' : 로스 함수
# random_state : SGD가 전체 데이터 학습이 아닌 일부 자료(미니배치) 가지고 학습하는데 사용되는 분리 기준( 난수값 )
# max_iter : (반복)계산 횟수 # 미니 배치이므로 전체 데이터셋을 '10'이면 10 반복 학습하여 모델 성공 향상( 에포크 )
# tol = None : 최적의 정확도를 찾아도 계속 반복학습 설정
sc = SGDClassifier( loss= 'log_loss', random_state = 42, max_iter = 10, tol=None )
sc.fit( train_scaled, train_target )
print( sc.score( test_scaled, test_target ) ) # 0.85 로지스틱이랑 비슷하게
print( sc.predict( test_scaled[:3] ) ) # ['Perch' 'Smelt' 'Pike']

# [3] 점진적(부분) 학습 ( 중간에 일부 학습 가능하다.)
sc.partial_fit( train_scaled, train_target ) # 총 11번 학습( 기본 10번 + 이 줄 1번 )

# [4] 최적의 학습횟수(에포크) 찾기
sc = SGDClassifier( loss = 'log_loss', random_state=42)

train_score = []    # 학습용 정확도
test_score = []     # 테스트용 정확도

#
import numpy as np
classes = np.unique( train_target ) # 정답지의 중복제거한 정답분류의 고유 정답만 추출

for i in range( 0, 150 ):
    sc.partial_fit( train_scaled, train_target, classes=classes ) # 1학습
    train_score.append( sc.score( train_scaled, train_target ) )
    test_score.append( sc.score( test_scaled, test_target ) )

# 정확도 시각화 # 과대적합( 훈련/테스트 떨어져 있으면 과대 ) # 과소적합( 훈련/테스트 너무 붙어 있으면 과소 )
# 최적의 에포크는 학습용과 테스트용이 고르게 오르는 시점
import matplotlib.pyplot as plt
plt.plot( train_score )
plt.plot( test_score )
plt.show()

# [5] hinge / SVM( 서포트 벡터 머신 )
# alpha = 0.0001 : 기본값, 힌지 함수는 경계면(애매/아슬)에 있는 자료들을 찾는 기준
sc = SGDClassifier( loss='hinge', max_iter=100, random_state=42, alpha=0.0001 )
sc.fit( train_scaled, train_target )
print( sc.score( train_scaled, train_target ) )
print( sc.score( test_scaled, test_target ) )

# 로지스틱 회귀 : 확률 이용한 분류('log_loss')
# SGD(확률 경사하강법) : loss='log_loss' vs loss='hinge'
# 경사하강법: 손실( 예측과 정답 오차 ) 0 가깝게 처리하기 위한 반복 계산( *딥러닝* )

# loss= 'log_loss'
    # 도미 확률이 51%일 때 기울기(가중치)와 절편으로 수없이 조정하여 확률 100% 만드는 방법 ( 경사 하강법 )
# loss = 'hinge'
    # 도미 확률이 50%, 0인 지점인 애매/아슬한 (경계선) 자료만 가지고 확률 조정하는 방법
