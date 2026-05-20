
# [1] Fish.csv 가져오기
import pandas as pd
df = pd.read_csv( './day02/fish.csv')

# [2] Perch(농어)만 추출
target_fish = df[ df['Species'].isin(['Perch'])]
target_fish.info()

# 농어의 길이/무게 추ㅜㅊㄹ

perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values
print( perch_length, perch_weight )

# 농어 길이에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter( perch_length, perch_weight )
plt.show()

# [3] 학습 모델 만들기, (1) 준비 : 학습용과 테스트용 분리한다. 왜? 모델평가에 사용하기 위해
from sklearn.model_selection import train_test_split
# random_State = 분리할 때 사용되는 난수값, # 난수값에 따라 분리한다. 고정값 넣어주면 항상 동일한 분리 값 넣을 수 있다. # 0~32억 사이
train_input, test_input, train_target, test_target = train_test_split( perch_length, perch_weight, test_size = 0.3, random_state = 42)

# (2) 준비: 자료형식(모양) 구성, 대부분 2차원 사용한다.
import numpy as np
array = np.array( [1,2,3,4] )
print( array.shape ) # ( 4, )
array2 = np.array( [[ 1, 2], [3 , 4]] )
print( array2.shape ) # (2, 2)

print( train_input.shape ) # (39, ) 1차원 배열 -> 사이킷런 모델들은 1차원배열 학습이 불가능하다.
print( train_input )
# T1-01( zip활용 ), T1-02( column_stack 활용 ), T2-01( reshape ) : 1차원 -> 2차원

# [4]
train_input = train_input.reshape( -1, 1 )
print( train_input )
print( train_input.shape )
test_input = test_input.reshape( -1, 1) # 테스트 학습 2차원

# [5] 모델 학습
from sklearn.neighbors import KNeighborsClassifier # K최근접이웃 찾기
from sklearn.neighbors import KNeighborsRegressor # k최근접이웃 회귀
knr = KNeighborsRegressor() # 모델 객체 생성
knr.fit( train_input, train_target ) # 모델 학습 # (길이, 무게) # '길이'에 따른 '무게' 학습
print( knr.score( test_input, test_target ) )   # 모델 평가 # 0.9929281790592219  회귀모델에서는 결정계수라고한다.

print( test_input )                 # [8.4  18  27.5 ] # 길이
print( knr.predict( test_input ) )  # 모델(무게) 예측 # [ 61.4.  78.  248. ] ~~