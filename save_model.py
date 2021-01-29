import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train_csv_path = "./train.csv"
model_path = "./model.txt"


if __name__ == '__main__':

    # 訓練データ読み込み
    train = pd.read_csv(train_csv_path)

    # 前処理
    train['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    train['Embarked'].fillna(('Nan'), inplace=True)
    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'Nan': 99}).astype(int)

    delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
    train.drop(delete_columns, axis=1, inplace=True)

    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    # 学習
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

    categorical_features = ['Embarked', 'Pclass', 'Sex']
    params = {
        'objective': 'binary',
        'max_bin': 300,
        'learning_rate': 0.05,
        'num_leaves': 40
    }

    lgb_train = lgb.Dataset(X_train, y_train,
                            categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train,
                           categorical_feature=categorical_features)
    model = lgb.train(params, lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval=10,
                      num_boost_round=1000,
                      early_stopping_rounds=10)

    model.save_model(model_path)
