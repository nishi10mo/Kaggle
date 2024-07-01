import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC

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

# LightGBMで学習
model = GBC(learning_rate=0.03, n_estimators=680, min_samples_leaf=95, random_state=random_state)
model.fit(X_train, y_train)

# 学習データにおけるaccuracy
train_score = model.score(X_train, y_train)
print("score=" + str(train_score))

# テストデータで予測を行う
y_pred = model.predict(X_test)

# テストデータのIDを取得
test_ids = test.index.to_numpy()

# 予測結果とIDを結合
results = pd.DataFrame({'id': test_ids, 'Target': y_pred})

# CSVファイルに出力
results.to_csv('submission.csv', index=False)
