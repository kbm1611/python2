import pandas as pd
import httpx
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

class UsedCarTradeService:
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaler = None
        self.r2 = None

    # 머신 러닝 함수
    async def retrain( self, training_data ):
        self.df = pd.DataFrame(training_data)
        print( self.df.head() )
        car_full = self.df[['averageFuelEfficiency', 'mileageKm', 'monthSinceRelease', 'accidentDepreciationCount', 'ownerChangeCount']]
        car_target = self.df['salePriceManwon'].values
        train_input, test_input, train_target, test_target = train_test_split( car_full, car_target, test_size=0.2, random_state=42 )

        optimization = [ ]

        for degree in [ 1, 2, 3, 4, 5 ]:
            poly = PolynomialFeatures( degree=degree, include_bias=False )
            poly.fit( train_input )
            train_poly = poly.transform( train_input )
            test_poly = poly.transform( test_input )
            # 선형회귀
            lr = LinearRegression()
            lr.fit( train_poly, train_target )
            r2 = lr.score( test_poly, test_target )
            optimization.append({
                'r2' : r2,
                'model' : lr,
                'poly' : poly,
                'degree' : degree,
                'scaler' : None,
                'alpha' : None               
                                 })
            
            # 스케일링
            ss = StandardScaler()
            ss.fit( train_poly )
            train_scaled = ss.transform( train_poly )
            test_scaled = ss.transform( test_poly )
            for alpha in [ 0.01, 0.1, 1, 10, 100]:
                # 릿지모델
                ridge = Ridge( alpha=alpha )
                ridge.fit( train_scaled, train_target )
                r2 = ridge.score( test_scaled, test_target )
                optimization.append({
                'r2' : r2,
                'model' : ridge,
                'poly' : poly,
                'degree' : degree,
                'scaler' : ss,
                'alpha' : alpha               
                                 })
                
                # 라쏘 모델
                lasso = Lasso( alpha=alpha )
                lasso.fit( train_scaled, train_target )
                r2 = lasso.score( test_scaled, test_target )
                optimization.append({
                'r2' : r2,
                'model' : lasso,
                'poly' : poly,
                'degree' : degree,
                'scaler' : ss,
                'alpha' : alpha               
                                 })
        best_optimization = max( optimization, key= lambda x : x['r2'] )
        print( f'선택된 모델: {best_optimization}' )
        self.model = best_optimization['model']
        self.poly = best_optimization['poly']
        self.scaler = best_optimization['scaler']
        self.r2 = best_optimization['r2']

        return True
    # 예측 매매값 함수
    async def pricePrediction( self, used_car_data ):
        print( "예측 데이터:", used_car_data )

        if self.model is None:
            return "학습 모델이 없습니다."
        
        input_df = pd.DataFrame([used_car_data])
        input_data = input_df[['averageFuelEfficiency', 'mileageKm', 'monthSinceRelease', 'accidentDepreciationCount', 'ownerChangeCount']]
        input_poly = self.poly.transform( input_data )

        if self.scaler is not None:
            result_data = self.scaler.transform( input_poly )
        else:
            result_data = input_poly
        
        result = self.model.predict( result_data )
        return round(float(result[0]), 2)


usedCarTradeService = UsedCarTradeService()