import xgboost as xgb
import numpy as np

class Ranker:
    def __init__(self, model_path="saved_models/xgb_ranker.json"):
        self.model = xgb.XGBRanker()
        self.model.load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)