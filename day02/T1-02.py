# T1-01.py
# Fish.csv -> https://www.kaggle.com/datasets/vipullrathod/fish-market?select=Fish.csv

# [1] CSV 불러오기
import pandas as pd
df = pd.read_csv('./day02/Fish.csv')
df.info()

# [2] 필요한 어종 추출 : 조건식 대신에 .isin(), .isna()
target_fish = df[ df['Species'].isin([ 'Bream', 'Smelt' ] ) ]
print( target_fish )

# [3] 필요한 특성 추출: Length2, Weight
# 넘파이 # np.column_stack( (리스트1), (리스트2) ): 두 리스트간에 동일한 요소로 2차원 리스트 구성
# T1-01.py 에서 [6] zip 함수 대신에 2차원 리스트로 구성방법
import numpy as np
fish_data = np.column_stack( (target_fish['Length2'], target_fish['Weight']) )
print( fish_data )

# [4] 모델 학습하기 위한 정답지
# np.ones( 개수 ): 개수만큼 1채워진 리스트 반환, np.zeros( 개수 ): 개수만큼 0 채워진 리스트 반환
# concatenate( 리스트, 리스트 ): 두 리스트 연결
fish_taget = np.concatenate( (np.ones( 35 ), np.zeros( 14 )) ) 
print( fish_taget )

# [5] 학습 모델 만들기 전 학습용, 테스트용 분리 # 방대한 자료( 억단위 이상 ) 학습용과 테스트용 구분하여 모델 구성하며 테스트한다.
from sklearn.model_selection import train_test_split
# 학습용자료, 테스트용자료, 학습용정답지, 테스트용정답지 = train_test_split( 학습자료, 정답지, test_size = 테스트자료 비율 )
# 4개의 반환 타입을 갖는다.
train_input, test_input, train_target, test_target = train_test_split( fish_data, fish_taget, test_size = 0.3 ) # 7:3 비율로
print( train_input.shape ) # ( 34, 2 ) 49개 중에 학습용 7에 해당하는 개수가 34개
print( test_input.shape ) # ( 15, 2 ) 49개 중에 테스트용 3에 해당하는 개수가 15개

# [6] 학습 모델: k-최근접 이웃 분류기 모델
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # 모델 객체 생성 # new 없음
kn.fit( train_input, train_target ) # 모델(지도) 학습
print( kn.score( test_input, test_target ) ) # 모델 평가

# [7] 임의의 값으로 학습 모델 예측하기
# 길이: 25, 무게: 150의 물고기가 도미[1]인지? 빙어[0]인지? 예측하기
print( kn.predict([[ 25, 150]])) # 모델 예측 # [0] 빙어로 예측 -> 잘못된 예측

#시각화
import matplotlib.pyplot as plt
# train_inpit[ :, 0] : [행슬라이싱, 열슬라이싱], 모든 행의 0번째 열만 추출 즉] 길이만 추출
# train_inpit[ :, 1] : [행슬라이싱, 열슬라이싱], 모든 행의 1번째 열만 추출 즉] 무게만 추출
plt.scatter( train_input[ : ,0], train_input[ : ,1]) # 학습용
plt.scatter( 25, 100 ) # 예측값
plt.show()

# [9] 예측하기 위한 이웃들 확인,  .kneighbors( [예측값] ), 예측에 사용된 이웃(거리, 인덱스)들을 반환
dist, indexs = kn.kneighbors( [ [25, 150] ] )
plt.scatter( train_input[:, 0], train_input[:, 1])               # 학습용
plt.scatter( 25, 100 )                                           # 예측값
plt.scatter( train_input[ indexs, 0 ], train_input[ indexs, 1] ) # 예측에 사용된 이웃 자료 # 문제 발견
plt.show()

# [10] 스케일, 표준화 필요성: < 공정하게 크기단위 맞추는 작업 > 길이와 무게값의 차이가 커서 일관된 비교가 어렵다.
# 예] 달리기점수 80 80 70, 몸무게 40 45 100 달리기 70~90, 몸무게 40~100
# 컴퓨터는 숫자가 더 큰 걸 중요하게 생각한다.
# 몸무게 40 = -1, 몸무게 100 = 1, 몸무게 70 = 0으로 취급하여 비교한다.
# 즉] 특정한 자료가 단위의 크기가 크면 큰 값이 모델을 지배하지 않도록 특정기준으로 맞춘다.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # 스케일 객체 생성
scaler.fit( train_input ) #
print( scaler.mean_ )     # 평균
print( scaler.scale_ )    # 표준편차
train_scaled = scaler.transform( train_input ) # 표준화(스케일링)
print( train_scaled )

# [11] 스케일링 이후 시각화, 차트 모양의 차이는 없지만 단위가 표준화 됨.
plt.scatter( train_scaled[ :, 0], train_scaled[ :, 1] )
plt.show()

# [12] 스케일링 이후 재학습 모델 만들기
kn.fit( train_scaled, train_target ) # 표준화된 자료로 재학습
# 임의의 예측값 (스케일링 된 )
test_scaled = scaler.transform( [[ 25, 150 ]] )
print( kn.predict( test_scaled  ) ) # 스케일링(표준화) 전에는 0, 이후에는 1예측했다.
# 예측에 사용된 이웃들 확인
dist, indexs = kn.kneighbors( test_scaled )

plt.scatter( train_scaled[:, 0], train_scaled[:, 1] )           # 스케일링된 학습용
plt.scatter( test_scaled[:, 0], test_scaled[:, 1] )             # 스케일링된 예측값
plt.scatter( train_scaled[indexs, 0], train_scaled[indexs, 1])  # 예측에 사용된 이웃 자료
plt.show()

# [9] 차트와 [12] 차트 비교하기