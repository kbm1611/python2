import pandas as pd
import matplotlib.pyplot as plt

# wine.csv: alcohol, sugar, pH, class
df = pd.read_csv("./day06/wine.csv")
data = df[["alcohol", "sugar", "pH"]]  # 와인들의 속성 3개
target = df["class"]  # 1: 화이트와인, 0: 레드와인

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, random_state=42
)

# [2] 결정트리 (분류 모델ㄹ )
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
print(dt.score(test_input, test_target))

# [3] 교차검증
from sklearn.model_selection import cross_validate

# cross_validate( 학습모델, 학습세트, 정답세트 )
# 교차검증은 전체 데이터를 N등분(폴드)하여 돌아가면서 검증 한다.
scores = cross_validate(dt, train_input, train_target)
print(scores)

#  'test_score': array([0.85128205, 0.84820513, 0.8788501 , 0.85112936, 0.84394251])}
import numpy as np

print(np.mean(scores["test_score"]))

#
from sklearn.model_selection import StratifiedKFold

# StratifiedKFold( n_splits = N등분 ), 기본값 5등분
splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splits)
print(scores)
# test_score': array([0.86680328, 0.84836066, 0.88090349, 0.83983573, 0.8788501 ,
#        0.85420945, 0.84804928, 0.85626283, 0.84804928, 0.86447639])}
print(np.mean(scores["test_score"]))

# [4] 그리드 서치, 최적의 파라미터( 변수/학습에 필요한 설정 값 ) 찾기
from sklearn.model_selection import GridSearchCV

# (1) 여러개 '최소불순도' 설정, 불순도란? 0에 가까울수록 예측값이 명확하다.(과대적합) 0.5에 가까울수록 예측값이 애매하다.(과소적합)
params = {"min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# (2)
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

# (3) 그리드 서치 학습
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(test_input, test_target))  # 0.8670769230769231
print(gs.best_score_)  # 0.8731517927657558
print(gs.best_params_)  # {'min_impurity_decrease': 0.0003}
# print( gs.cv_results_ ) # 기본값으로 교차검증 5가 적용된다.

# [5] 다중 파라미터
params = {
    # 최저불순도
    "min_impurity_decrease": np.arange(
        0.0001, 0.001, 0.0001
    ),  # 0.0001 ~ 0.001미만까지 0.0001씩 증가
    # 최대 뿌리 깊이
    "max_depth": range(5, 20, 1),
    # 노드 분할시 최저 샘플 수
    "min_samples_split": range(2, 100, 10),  # 10번
    # 리프노드(나머지 뿌리/노드) 최저 샘플수, 즉] 현재 리프노드가 최저 샘플수보다 작으면 노드 분할 안함.
    "min_samples_leaf": range(1, 100, 10),
}
# cv = 교차검증수(N등분), 기본값 5
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, cv=5)
# # 대략 학습 조합
# # 최저불순도( 9가지 ) * 뎁스( 15가지 ) * 최저분리샘플( 10번 ) * 최저리프샘플 ( 10번 ) = 13500 가지의 조합으로 학습
# # + 교차검증 ( N등분 ) => 67500 번의 학습 모델
# gs.fit(train_input, train_target)
# print(gs.best_params_)  # 최적의 파라미터 조합을 알려줌
# # -> {'max_depth': 13, 'min_impurity_decrease': np.float64(0.0001), 'min_samples_leaf': 11, 'min_samples_split': 2}
# print(gs.best_score_) # 0.8756162796819881

# [6] 랜덤 서치
# 조합 수가 많아지면 연산량이 많아져서 서버(컴퓨터)에 부하 발생할 수 있다.
# 랜덤서치란 고정된 값이 아니라 '확률 분포 함수'를 제공하여 무작위로 숫자를 뽑아 학습한다.
from sklearn.model_selection import RandomizedSearchCV

# n_iter = 100 # 정의된 조합수에서 무작위(랜덤)으로 N개의 조합만 추출하여 학습한다.
# 13500개의 조합에서 100개만 무작위로 추출 # 교차검증5 => 500번 학습
rs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1
)
rs.fit(train_input, train_target)
print(rs.best_params_)  # {'min_impurity_decrease': 0.0003}
# {'min_samples_split': 22, 'min_samples_leaf': 11, 'min_impurity_decrease': np.float64(0.00030000000000000003), 'max_depth': 15}
print(rs.best_score_)  # 0.872125309324488 # 학습속도는 빨라졌지만 정확도가 조금 낮아짐.
