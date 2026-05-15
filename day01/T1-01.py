import pandas as pd
# T1-01.py
# Fish.csv -> 캐글 데이터셋
# Species( 종 ),Weight( 무게 ),Length1( 길이1 ),Length2( 길이2 ),Length3( 길이3 ),Height( 높이 ),Width( 넓이 )

# [1] 데이터 불러오기
df = pd.read_csv('./day01/Fish.csv', encoding='utf-8')
print( df.head() )
df.info()

# [2] 특정한 물고기 추출, 도미 = Bream
bream_df = df[df['Species'] == 'Bream']

# [3] 도미의 길이, 무게 추출
bream_length = bream_df['Length2'].tolist()
bream_weight = bream_df['Weight'].tolist()
print( bream_length, bream_weight )

# [4] 특정한 물고기 추출, 빙어/Smelt
smelt_df = df[df['Species'] == 'Smelt']

# [3] 빙어의 길이, 무게 추출
smelt_length = smelt_df['Length2'].tolist()
smelt_weight = smelt_df['Weight'].tolist()
print( smelt_length, smelt_weight )

# [4] 시각화
import matplotlib.pyplot as plt
plt.scatter( bream_length, bream_weight )
plt.scatter( smelt_length, smelt_weight )
plt.xlabel('length(CM)')
plt.ylabel('weight(gram)')
plt.show()

# [5] 도미와 빙어 자료 합치기, 길이는 길이끼리, 무게는 무게끼리
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# [6] 2차원 리스트 -> [ [길이, 무게]. [길이, 무게], [길이, 무게], [길이, 무게] ]
# zip( 1차원리스트, 1차원리스트 ): 두 리스트를 요소 하나씩 반복
# 리스트 내보, [ 표현식 for 반복변수 in 반복값 if 조건식 ]
fish_data = [ [l,w] for l, w in zip( length, weight ) ]
print( fish_data ) # [ [길이1, 무게1], [길이2, 무게2]]
# print( bream_df.info ) # 도미 35마리
# print( smelt_df.info ) # 빙어 14마리

# [7] target(정답지) 만들기, 1:도미 의미하고 35개 만든다. 0:빙어 의미하고 14개 만든다.
fish_target = [1] * 35 + [0] * 14
print( fish_target )
# [ 1, 1, 1, 1, 1, 1, 1]

# [8] 알고리즘 모델 중 : (1) k-최근접 이웃(K-NN 알고리즘): 임의의값을 넣었을 때 기존 값들 중에 가장 가까운 값 찾기
# (1) 설치: 사이킷런( 다양한 머신러닝 모델 제공 ) # https://scikit-learn.org/stable/
# pip install scikit-learn
# (2) K-NN 모델 호출
from sklearn.neighbors import KNeighborsClassifier
# (3) K-NN 모델 객체 생성
kn = KNeighborsClassifier()
# (4) k-nn 학습하기, 문제와 답을 같이 준다. --> 지도학습 방식
kn.fit( fish_data, fish_target ) # fish_data 도미와방어 자료, fish_target 도미인지 빙어인지 식별 자료

# (5) 학습된 모델의 점수(정확도) 측정, kn-score( )
print( kn.score( fish_data, fish_target ) )

# (6) 임의의 값 넣어서 예측 측정, kn.predict( [임의값] )
print( kn.predict( [ [30, 600] ] )) #임의의 물고기 길이cm와 무게600 -> 도미? 빙어?

# (7) 임의의 값 시각화 -> 인공지능을 안 사용하더라도 시각화를 통해 분류 할 수 있다.
plt.scatter( bream_length, bream_weight )
plt.scatter( smelt_length, smelt_weight )
plt.scatter( 30, 600 ) # 임의의 값 위치
plt.show()

# (8) 근접한 이웃 찾을 기준 정하기, 하이퍼파라미터
# KNeighborsclassifier( n_neighbors = 참조할이웃개수 ) # 접근한 49개 중에서 정답 찾기
kn = KNeighborsClassifier( n_neighbors = 49 )
kn.fit( fish_data , fish_target ) # 학습
print( kn.score( fish_data, fish_target ) ) #정확도 측정
# 49개를 보고 결정하기 때문에 무조건 도미 수가 많아서 '도미'로 예측함.