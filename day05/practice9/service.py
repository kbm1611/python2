import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import httpx


class FashionSystemService:
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaler = None
        self.score = None
        self.alpha = None

    # age, gender, inflow, style, category
    async def retrain(self, training_data):
        self.df = pd.DataFrame(training_data)
        data = self.df[["age", "gender", "inflow", "style"]]
        target = self.df["category"].values

        optimization = []
        test_size_list = np.arange(0.1, 0.2, 0.01)
        for test_size in test_size_list:
            train_input, test_input, train_target, test_target = train_test_split(
                data, target, test_size=test_size, random_state=45
            )

            for degree in [1, 2, 3, 4, 5]:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly.fit(train_input)
                train_poly = poly.transform(train_input)
                test_poly = poly.transform(test_input)

                # 스케일링
                ss = StandardScaler()
                ss.fit(train_poly)
                train_scaled = ss.transform(train_poly)
                test_scaled = ss.transform(test_poly)

                sc = SGDClassifier(loss="log_loss", max_iter=100, random_state=45, tol=None)
                sc.fit(train_scaled, train_target)
                score = sc.score(test_scaled, test_target)
                optimization.append(
                    {
                        "score": score,
                        "model": sc,
                        "poly": poly,
                        "degree": degree,
                        "scaler": ss,
                        "test_size" : test_size,
                        "alpha": None,
                    }
                )

                alpha_list = [10**i for i in range(-6, 4)]
                for alpha in alpha_list:
                    sc = SGDClassifier(
                        loss="hinge", max_iter=100, random_state=42, alpha=alpha
                    )
                    sc.fit(train_scaled, train_target)
                    score = sc.score(test_scaled, test_target)
                    optimization.append(
                        {
                            "score": score,
                            "model": sc,
                            "poly": poly,
                            "degree": degree,
                            "scaler": ss,
                            "test_size" : test_size,
                            "alpha": alpha,
                        }
                    )

        best_optimization = max(optimization, key=lambda x: x["score"])
        print(f"선택된 모델: {best_optimization}")
        self.model = best_optimization["model"]
        self.poly = best_optimization["poly"]
        self.scaler = best_optimization["scaler"]
        self.score = best_optimization["score"]

        return True

    async def categoryPredict(self, data):
        print("예측 데이터:", data)

        if self.model is None:
            return "학습 모델이 없습니다."

        input_df = pd.DataFrame([data])
        input_data = input_df[["age", "gender", "inflow", "style"]]
        input_poly = self.poly.transform(input_data)
        input_scaled = self.scaler.transform(input_poly)

        result = self.model.predict(input_scaled)
        return int(result[0])


fashionSystemService = FashionSystemService()
