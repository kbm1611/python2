
# PythonML Practice 8: SGDClassifier 기반 취업 여부 분류 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./day04/student_dataset_10000_rows.csv')
# [1] 데이터 전처리 및 독립/종속 변수 분할
# 제시된 학생 데이터셋에서 취업 여부를 나타내는 범주형 문자열 데이터인 placement_status (Placed, Not Placed)를 종속변수(타깃)로 설정하시오.
# 타깃을 제외한 study_hours, attendance 등 총 7개의 수치형 특성을 독립변수로 지정하고, 전체 데이터를 학습 세트와 검증 세트 비율 8:2로 정확하게 분리하시오.
student_target = df['placement_status']
student_input = df[['study_hours','attendance','sleep_hours','internet_usage','assignments_completed','previous_score', 'exam_score']]
train_input, test_input, train_target, test_target = train_test_split( student_input, student_target, test_size= 0.2, random_state= 42 )

optimization = [ ]

# [2] 특성 다항 확장 및 경사하강법 필수 스케일링
for degree in [1,2,3,4,5]:
    poly = PolynomialFeatures( degree=degree, include_bias=False )
    poly.fit( train_input )
    train_poly = poly.transform( train_input )
    test_poly = poly.transform( test_input )

    # 스케일링
    ss = StandardScaler()
    ss.fit( train_poly )
    train_scaled = ss.transform( train_poly )
    test_scaled = ss.transform( test_poly )

    # [3] 로그손실 또는 힌지손실 기반의 규제 하이퍼파라미터
    sc = SGDClassifier( loss='log_loss', max_iter=100, random_state=442, tol=None )
    sc.fit( train_scaled, train_target )
    train_score = sc.score( train_scaled, train_target )
    test_score = sc.score( test_scaled, test_target )
    optimization.append({ 'model' : sc, 'loss' : 'log_loss', 'poly' : poly, 'degree' : degree, 'alpha' : None, 'scaler' : ss, 'train_score' : train_score, 'test_score' : test_score })

    alpha_list = [ 10**i for i in range(-4, 3) ]
    for alpha in alpha_list:
        sc = SGDClassifier( loss='hinge', max_iter=100, random_state=42, alpha=alpha)
        sc.fit( train_scaled, train_target )
        train_score = sc.score( train_scaled, train_target )
        test_score = sc.score( test_scaled, test_target )
        optimization.append({ 'model' : sc, 'loss' : 'hinge', 'poly' : poly, 'degree' : degree, 'alpha' : alpha, 'scaler' : ss, 'train_score' : train_score, 'test_score' : test_score })
   
# [4] 최고 정확도(Score) 선정 , *0.90 이상 찾기* 
best_optimization = max(optimization, key = lambda x : x['test_score'] )
print( best_optimization ) # SGDC 모델 중 경사하강법 vs 힌지 비교
# {'loss': 'hinge', 'degree': 2, 'alpha': 0.0001, 'train_score': 0.997375, 'test_score': 0.997}

# [5] 신규 학생 데이터 취업 예측 (추론 함수 구현 및 검증)
# 구현된 모델에 아래 두 가지 샘플 데이터를 입력하여  취업 성공(Placed)과 실패(Not Placed)를 올바르게 분류해내는지 최종 검증하시오.
def placement_status_predict( *params ):
    sample = [ list( params ) ]

    poly = best_optimization['poly']
    ss = best_optimization['scaler']
    model = best_optimization['model']

    sample_poly = poly.transform( sample )
    sample_scaled = ss.transform( sample_poly )

    result = model.predict( sample_scaled )
    return result[0]
#  - 샘플 A : study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85, exam_score = 97.88
result = placement_status_predict( 9, 95, 7, 2, 18, 85, 97.88 )
print( result )
#  - 샘플 B : study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50, exam_score = 50.43
result = placement_status_predict( 2, 60, 5, 9, 4, 50, 50.43 )
print( result )