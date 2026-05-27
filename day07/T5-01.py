import pandas as pd
import numpy as np

# ==========================================
# 데이터 준비 (3차원 특성: 무게, 당도, 단단함)
# ==========================================
data = {  
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7],
    'hardness': [7.8, 6.5, 7.1, 4.2, 3.5, 3.9, 8.9, 8.4, 8.1, 5.8, 5.2, 6.1, 7.3, 7.5, 7.0, 5.9, 6.2, 6.4, 7.2, 7.6,
                 6.8, 4.5, 4.1, 7.0, 5.7, 6.9, 4.9, 6.6, 8.2, 8.5, 5.8, 4.0, 5.3, 6.7, 6.1, 5.0, 5.2, 4.7]
}
df = pd.DataFrame(data)

# 테스트용
newDf = pd.DataFrame({'weight': [110], 'sweetness': [7.0], 'hardness': [7.5]})
features = ['weight', 'sweetness', 'hardness']

# [1] K-means : 정해진(K)개수만큼의 그룹/군집, 중심점의 평균 계산
from sklearn.cluster import KMeans
# n_clusters = K, 그룹 수 설정, 2이면 2개의 그룹으로 군집화
# random_state,, 그룹/군집/클러스터 설정하기 위한 초기 중심점 무작위 난수 생성
km = KMeans( n_clusters = 3 ,random_state = 42 ) 
km.fit( df[features] )                  # 모델 (비지도)학습 #target(정답/레이블)이 없다.
print( km.predict( newDf[features] ) )  # 모델 예측(클러스터/군집화)
print( km.labels_ )                     # 행 마다의 군집 번호,
# 시각화
import matplotlib.pyplot as plt
plt.scatter( df['weight'], df['sweetness'], c = km.labels_ )
plt.scatter( newDf['weight'], newDf['sweetness'], marker = '^' )
plt.show()
# 특성들 간에 서로 다른 단위의 의미 => 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaledDf = ss.fit_transform( df[features] ) # fit + trans_from
scaledNewDf = ss.transform( newDf[features] )
# 스케일링 이후 시각화
plt.scatter( scaledDf[ :, 0], scaledDf[ :, 1], c = km.labels_)
plt.scatter( scaledNewDf[ :, 0], scaledNewDf[ :, 1], marker = '^' )
plt.show()

# [2] 최적의 K(그룹수) 찾기, 엘보우 방법
sse = [] #오차들을 저장하는 리스트
for k in range( 1, 11 ):
    km = KMeans( n_clusters = k, random_state=42 )
    km.fit( scaledDf ) # 스케일링된 자료 학습
    sse.append( km.inertia_ ) # 군집/그룹/클러스터 내 자료들 간 오차의 제곱합 측정
print( sse ) # 클러스터가 많아지면 오차의 제곱합이 줄어들 수 밖에 없다. 더 촘촘해지기 때문에.
# 오차 시각화
plt.plot( range(1, 11), sse, marker = 'o' )
plt.show()
# 엘보우 포인트 : SSE(오차제곱합) 차이가 급격하게 줄어든 포인트 => 최적의 K
# 최적의 K 모델 재학습
km = KMeans( n_clusters= 3, random_state= 42 )
km.fit( scaledDf )
df['cluster'] = km.labels_ # 클러스터 결과물
# weight, sweetness, hardness, cluster

# [3] 거리 예측/계산( 추론 계산식 ), 유클리드 거리
# (1) 클러스터들의 중심점
centerClus = km.cluster_centers_
print( centerClus )
# (2) 중심점에서 새로운자료의 오차(차이) 계산, 오차합의 제곱, 제곱근 씌운다.
# np.sum( 리스트, axis = 축기준 ), # 0: 열 / 1: 행
print( np.sum( [1,2,3] ) )
print( np.sum( [[1,2,3],[4,5,6]] ) ) # 21
print( np.sum( [[1,2,3],[4,5,6]], axis = 1 ) ) # [6 15] axis = 축 기준으로
# np.sqrt( 리스트 ) #제곱근(루트)
result = np.sqrt( np.sum( (centerClus - scaledNewDf) ** 2, axis = 1 )  )
print( result ) # [1.26432524 3.25608991 0.97575994]
print( km.predict( scaledNewDf ) ) # [2] # 유클리드 거리 계산과 predict 예측과 동일하다.

# [4] GMM : 가우시안 모델, 군집확률
from sklearn.mixture import GaussianMixture
gm = GaussianMixture( n_components= 3, random_state=42 )
gm.fit( scaledDf ) # 학습
print( '--가우시안--' )
print( gm.predict( scaledNewDf ) ) # [0]
print( np.round(gm.predict_proba( scaledNewDf ), 2) ) # [[8.44533923e-01 3.78321527e-12 1.55466077e-01]]
# 시각화
plt.scatter( scaledDf[ : , 0], scaledDf[ : , 1], c = df['cluster'] )
plt.scatter( scaledNewDf[ : , 0], scaledNewDf[ : , 1], marker = '^' )
plt.show()
# 현재 특성이 3개 이므로 3D차원 시각화 필요 -> 만약에 N차원이면 시각화 힘들다.


# [5] PCA : 주성분 분석 / 차원 축소, 차원이 크면 시각화 불가능하다. 주로 2차원/3차원으로 압축한다.
from sklearn.decomposition import PCA
# 여러개 특성/성분을 가진 모델들을 2/3차원 변경, 무게/당도/단단함 => 2차원 표현
pca = PCA( n_components=2 ) # 객체 생성 # 주로 2또는 3으로 사용된다.
# 예] pca = 무게*가중치1 + 당도*가중치2 + 단단함*가중치3
pcaDf = pca.fit_transform( scaledDf )
print( pcaDf ) # 행 = 데이터수, 열 = 주성분 수
# [[ 1.92249033 -0.42197445]
#  [ 0.17574959  0.23619089] ~~~
df['pca_x'] = pcaDf[ :, 0] # 첫번째 열을 제1주성분 # 데이터의 변동성을 가장 크게 설명하는 주성분
df['pca_y'] = pcaDf[ :, 1] # 두번째 열을 제2주성분 # 제1주성분을 직교하면서 두번째로 변동성(가중치)을 가장 크게 설명하는 주성분
# 주성분의 가중치 확인
components = pca.components_
print( components ) #[[-0.58085247 -0.56151577  0.58933051] # 무게가중치, 당도가중치, 단단가중치
# 예측한 값을 주성분 변경
pcaNewDf = pca.transform( scaledNewDf )

# 시각화
plt.scatter( df['pca_x'], df['pca_y'], df['cluster'], cmap='viridis', marker='o' )
plt.scatter( pcaNewDf[ :, 0], pcaNewDf[ :, 1], marker = '^')
plt.xlabel( 'pca 1' )
plt.ylabel( 'pca 2' )
plt.show()

# [6] 분석모델 스코어, 실루엣 스코어( 분리/응집 평가 )
from sklearn.metrics import silhouette_score
# 객체 생성 silhouette_score( 자료, 모델군집 )
sc = silhouette_score( scaledDf, km.labels_ )
print( sc ) # 0.443044585687068

# gm 가우시안 모델에서는 K-mean처럼 labels_ 속성이 없다. 그래서 예측을 통한 군집도 구한다.
sc = silhouette_score( scaledDf, gm.predict( scaledDf ) ) # GMM 평가 모델
print ( sc ) # 0.443044585687068

# 스코어 개선 : [1] 최적의 K [2] PCA 주성분(가중치) [3] 이상치 제거( 튀는 자료는 군집 제외 ) 등등

# [7] HDBCAN 모델 : 자동으로 최적의 K와 이상치 제거 모델
# pip install hdbscan
import hdbscan
# min_cluster_size = 최소의 클러스터 개수
# min_samples = 클러스터들의 중심점이 되기 위한 최소 자료(샘플) 수, 최소 2개 이상부터 군집 가능
# prediction_data = True, 학습된모델이 예측할 경우 캐시(임시메모리) 활성화
hdb = hdbscan.HDBSCAN( min_cluster_size=2, min_samples=2, prediction_data=True )
hdb.fit( scaledDf )
print( hdb.labels_ )
# [-1  2  2  0 -1  0 -1 -1 -1  3  4 -1  1  1  1 -1  2  2  1  1  2 -1  0  2
#   3  2 -1  2 -1 -1  3 -1  4  2  2  4  4 -1]
print( hdbscan.approximate_predict( hdb, scaledDf) )
print( hdbscan.approximate_predict( hdb, scaledNewDf) )