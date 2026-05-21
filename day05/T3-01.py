
# [1] 여러가지 특성에 따른 어종 분류
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

# 로지스틱 회귀 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1 (확률/분류) 사이의 값으로 변환해주는 공식/함수
import numpy as np
# import matplotlib.pyplot as plt
# z = np.arange( -5, 5, 0.1 ) # -5 부터 5까지 0.1씩 증가하는 리스트
# phi = 1 / ( 1 + np.exp( -z ) ) # 시그모이드 공식
# plt.plot( z, phi ) # 시그모이드 시각화
# plt.show()

# [2] 이진 분류 # 로지스틱 회귀 모델
# 이진분류는 0 또는 1
indexs = (train_target == 'Bream') | (train_target == 'Smelt')
print( indexs )
train_bream_smelt = train_scaled[ indexs ]
target_bream_smelt = train_target[ indexs ]
print( train_bream_smelt )
print( target_bream_smelt )
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit( train_bream_smelt, target_bream_smelt ) # 도미와 빙어만 학습
# 이진분류 모델 예측
print( lr.predict( train_bream_smelt[ :3 ] ) ) # 3개만 예측
print( lr.predict_proba( train_bream_smelt[ :3 ] ) ) # 3개만 예측 확률
# 0.5 임계값 기준 이상이면 도미, 미만 빙어

# [3] 다중 분류 # 로지스틱 회귀
# 하이퍼파라미터
# c = 규제를 완화하여 릿지/라쏘 모델처럼 정확도 설정 가능하다.
# max_iter = 반복횟수 설정, 생략시 기본값100으로 최적의 정확도를 찾을 때 까지 계산 반복횟수 조정 # 넉넉하게
lr = LogisticRegression( C = 5, max_iter = 200 )
lr.fit( train_scaled, train_target )
# 모델 예측
print( lr.predict( test_scaled[ :3 ] ) )
print( lr.predict_proba( test_scaled[ :3 ] ) )
# 모델 평가, 선형 회귀와 다르게 *결정계수*라고 하지 않고 맞힌 비율
print( lr.score( test_scaled, test_target ) )
# 소프트맥스
from scipy.special import softmax
decison = lr.decision_function( test_scaled[ :3 ] )
print( decison )
print( softmax( decison ) )                     # 소프트맥스라는 함수로 결과값을 확인 햇을 때 predict 동일하게 출력된다.
print( np.round( softmax( decison ), decimals = 3 ) ) # np.round( 값, decimals = 소수점 ) #

# 다중 분류의 확률 검증할때는 .classes_ 종속변수들의 순서 확인
print( lr.classes_ ) # 종속변수들 출력 : ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']