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

# [2] 결정 트리( 분류 모델 )
from sklearn.tree import DecisionTreeClassifier  # 의사결정 트리 분ㄹ

dt = DecisionTreeClassifier()
dt.fit(train_input, train_target)  # 모델 학습
print(dt.score(train_input, train_target))  # 모델 정확도: 0.9973316912972086
print(dt.score(test_input, test_target))  # 0.8461538461538461
print(dt.predict(test_input[:5]))  # 모델 예측 [1. 0. 1. 1. 1.]
# [3] 결정 트리 시각화
from sklearn.tree import plot_tree

# filled = True 노드 색깔을 다르게 표현
plot_tree(
    dt, max_depth=3, feature_names=["alcohol", "sugear", "pH"], filled=True
)  # plot_tree( 트리모델 )
plt.show()
# 트리 : 전체적인 구조
# 노드 : 사각형 상자 하나하나, 가장 위에 있는 노드를 루트노드라 함
# 노드 속성 :
# value = [ 예측타겟수 ] # [ 85, 2097 ] 0으로 85개 예측, 1로 2097
# gini = 불순도(0.075)
# 0으로 가까울수록 정확/순수하다.
# 0.5으로 가까울수록 탁하다.

# [4] 특성 중요도
# 각 특성이 트리 모델에 얼마나 중요한 역할을 하는지 # 수치 합은 1
print(dt.feature_importances_)  # [0.23458836 0.51959358 0.24581806]
print(dt.feature_importances_[0])  # alcohol

# [5] 최소한의 불순도(gini) 설정, 최적의 파라미터
dt = DecisionTreeClassifier(random_state=42, min_impurity_decrease=0.0005)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))  # 0.8975779967159278
print(dt.score(test_input, test_target))  # 0.8590769230769231 과대적합 최소화
