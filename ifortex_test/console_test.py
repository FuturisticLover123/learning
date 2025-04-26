import pandas as pd
import sys
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from joblib import load



class FeatureAdder(BaseEstimator, TransformerMixin):
  def __init__(self):
      pass

  def fit(self, X, y=None):
      return self

  def transform(self, X: pd.DataFrame):
      X = X.copy()
      if isinstance(X, pd.DataFrame):
        cols = X.columns[:3]

        X['Mean'] = X[cols].mean(axis=1, skipna=True)
        X['Min'] = X[cols].min(axis=1, skipna=True)
        return X


if len(sys.argv) < 2:
    print("Usage: python script.py filename")
    sys.exit()

filename = sys.argv[1]
data = pd.read_csv(filename)
x, y = data.drop(columns="Сдал"), data["Сдал"]



pipeline = load('pipeline.joblib')

predict = (pd.DataFrame(pipeline.predict(x)))
predict.to_csv("predictions", index=False, header=False)


print(accuracy_score(pipeline.predict(x), y), "Accuracy")

probs = pd.DataFrame(pipeline.predict_proba(x), columns = ["Вероятность сдать", "Не сдать" ])

predict.to_csv("probs", index=False, header=False)

print("Вероятности")
print(probs)
