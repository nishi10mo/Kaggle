import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import model_selection

random_state=1234

# データセットの読み込み
train = pd.read_csv('data/train.csv', index_col='id')
test = pd.read_csv('data/test.csv', index_col='id')

# 訓練データを説明変数と目的変数に分割
y_train = train[['Target']]
X_train = train.drop(columns=['Target'])

# numpyに変換
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = test.to_numpy()

# ハイパーパラメータの探索
def objective(trial):
    
    model = GBC()
    params = {"learning_rate":trial.suggest_int("learning_rate",
                                               0, 1
                                              ),
              "n_estimators":trial.suggest_int("n_estimators",
                                            1, 1000
                                           ),
              "min_samples_split":trial.suggest_int("min_samples_split",
                                            2, 500
                                           ),
              "max_depth":trial.suggest_int("max_depth",
                                            1, 100
                                           )
             }
    classifier_obj = model.set_params(**params)
    score = model_selection.cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

# コールバック関数を定義
def progress_callback(study, trial):
    n_trials = study._study.objective_trials[0]._n_trials
    print(f"Trial {trial.number + 1}/{n_trials}: Best value so far: {study.best_value}")

study = optuna.create_study(direction="maximize")
print(">>> start hyper-parameter tuning >>>")
study.optimize(objective, n_trials=10, callbacks=[progress_callback])
print("<<< end hyper-parameter tuning <<<")
print(f"The best value is : \n {study.best_value}")
print(f"The best parameters are : \n {study.best_params}")

# 探索したハイパーパラメータで学習
best_params = study.best_params
best_model = GBC(**best_params)
best_model.fit(X_train, y_train)

# テストデータで予測を行う
y_pred = best_model.predict(X_test)

# テストデータのIDを取得
test_ids = test.index.to_numpy()

# 予測結果とIDを結合
results = pd.DataFrame({'id': test_ids, 'Target': y_pred})

# CSVファイルに出力
results.to_csv('submission.csv', index=False)
