# PythonML Practice 5: 다항 규제 회귀 기반 성적 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset

# [1] 데이터 분할: 범주형 변수를 제외한 6개 특성을 독립변수로, `exam_score`를 타깃으로 설정하고 8:2 비율 로 학습 및 검증 세트를 분리하시오.
# [2] 모델 전수 탐색: `LinearRegression`, `Ridge`, `Lasso` 모델과 다항 확장, 다양한 규제 강도 조합을 모두 학습시키시오.
# [3] 최적 모델 선정: 테스트 데이터셋(`X_test`) 기준 최고의 결정계수를 달성하는 최적의 알고리즘, 차수, 알파 값을 자동 도출하고 추론 엔진에 매핑하시오.
# [4] 추론 함수 구현: 새로운 학생의 6가지 특성 데이터를 인자로 받아 최적 모델의 다항 구조와 스케일링 기준을 거쳐 성적을 예측하는 함수를 구현하시오.
# [5] 샘플 데이터 검증: 구현된 함수에 두 가지 대조군 샘플을 대입하여 시험성적을 예측하시오.
    # study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
    # study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50

import pandas as pd
df = pd.read_csv('./day04/student_dataset_10000_rows.csv')
# 범주형( 기준으로 나누어진 자료들 ) 수치형( 연속된 수 )
# 독립변수(특성) : study_hours, attendance, sleep_hours, internet_usage, assigments_completed, previous_score
# 종속변수(타깃) : exam_score
student_full = df[['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']]
student_target = df['exam_score'].values
# 8:2 비율로 학습 및 검증 세트를 분리하시오.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( student_full, student_target, test_size = 0.2, random_state= 42 )

# [2] 모델 전수 탐색
# 다항 확장 : 특성(자료)들 간에 직선 관계 드물다.(단순회귀), 물고기길이, 물고기길이 제곱, 물고기길이 세제곱 ~ (다항 회귀)
# 직선 관계가 아닌 곡선 관계 만들고 다양한 경우의 수의 학습 자료를 만든다. 주의할 점: 과적합
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# [3] 최적 모델 찾기 위한 리스트
optimization = [ ]

for degree in [ 1, 2, 3, 4, 5 ]:
    poly = PolynomialFeatures( degree = degree, include_bias=False ) #degree=차수, include_bias=False : 기본값 제거
    poly.fit( train_input )
    train_poly = poly.transform( train_input )
    test_poly = poly.transform( test_input )
    print(f'{ degree } 차수의 특성 수')
    print( train_poly.shape )
    # 선형 회귀
    lr = LinearRegression()
    lr.fit( train_poly, train_target )
    r2 = lr.score( test_poly, test_target )
    # print(f'{ degree } 차수의 선형 회귀 결정계수 : { r2 }') # 결정계수란? 해당 모델이 예측한 결과가 얼마나 잘 설명 되는지 수치화(백분율)
    optimization.append( { 'r2' : r2, 'model' : lr, 'poly' : poly, 'degree' : degree, 'scaler' : None, 'alpha' : None } ) # 모델이랑, 결정계수를 뺌

    # 스케일링
    ss = StandardScaler()
    ss.fit( train_poly )
    train_scaled = ss.transform( train_poly )
    test_scaled = ss.transform( test_poly )
    for alpha in [ 0.01, 0.1, 1, 10, 100 ]:
        # 릿지 모델
        ridge = Ridge( alpha = alpha )
        ridge.fit( train_scaled, train_target )
        r2 = ridge.score( test_scaled, test_target )
        # print( f'{ degree } 차수의 릿지 강도: { alpha } 의 결정계수 : { r2 }')
        optimization.append( { 'r2' : r2, 'model' : ridge, 'poly' : poly, 'degree' : degree, 'scaler' : ss, 'alpha' : alpha } ) 
        # 라쏘 모델
        lasso = Lasso( alpha = alpha )
        lasso.fit(  train_scaled, train_target )
        r2 = lasso.score( test_scaled, test_target )
        # print( f'{ degree } 차수의 라쏘 강도 : { alpha }의 결정계수 : { r2 }')
        optimization.append( { 'r2' : r2, 'model' : lasso, 'poly' : poly, 'degree' : degree, 'scaler' : ss, 'alpha' : alpha } )

# [3] 최적 모델 찾기위한 리스트에서 'r2' 가장 큰 모델 찾기
# 예시] max( 리스트, key = lamda x : x['기준열'] )
best_optimization = max( optimization, key = lambda x : x['r2'] )
best_model = best_optimization['model']
best_poly = best_optimization['poly']
scaler = best_optimization['scaler']
print( f'최적의 모델 : { best_model }, 다항특성 : { best_poly }, 스케일링 : { scaler }')
# 최적의 모델 : Ridge(alpha=10), 다항특성 : PolynomialFeatures(degree=3, include_bias=False), 스케일링 : StandardScaler()

# [4] 추론_함수 구현 # 사이킷런 모델은 2차원 자료들만 사용
def exam_score_predict( *params ):
    list_data = [ list( params ) ]

    # 특성 공학( 다항 만들기 ) 적용
    list_poly = best_poly.transform( list_data )
    
    # 예측 모델이 스케일링 사용할지 여부 판단
    if scaler is not None:  # 만약에 scaler 사용한 모델만 스케일링
        input_data = scaler.transform( list_poly ) # 스케일링 적용
    else:
        input_data = list_poly
        
    result = best_model.predict( input_data )
    return result[0] # 예측한 값을 반환한다. # 항상 예측 결과도 리스트 반환하기 때문에 인덱스 [0]으로 하나 추출


# [5] 샘플 데이터 검증
# study_hours = 9, attendance=95, internet_usage=2, assignments_complete=18, previous_score=85
# study_hours = 2, attendance=60, internet_usage=5, assignments_complete=9, previous_score=50

result = exam_score_predict( 9, 95, 7, 2, 18, 85 )
print(f'result : {result:.2f}점')   # 97.87794156408047점 :.2f -> 97.88점
result = exam_score_predict( 2, 60, 5, 9, 4, 50 )
print(f'result : {result:.2f}점')   # 50.425715066659016점 :.2f -> 50.43점
    
