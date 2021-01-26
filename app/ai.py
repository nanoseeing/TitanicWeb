import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train_csv_path = "./app/static/train.csv"


def calc_ypred(form_input):

    # 訓練データ読み込み
    train = pd.read_csv(train_csv_path)

    # テストデータ作成
    input_data = dict()
    input_data['Pclass'] = int(form_input['pclass'])
    input_data['Name'] = form_input['name']
    input_data['Sex'] = form_input['sex']
    input_data['Age'] = float(form_input["age_1"] + "." + form_input["age_2"])
    input_data['SibSp'] = int(form_input['sibsp'])
    input_data['Parch'] = int(form_input['parch'])
    input_data['Ticket'] = form_input['ticket']
    input_data['Fare'] = float(form_input["fare_1"] + "." + form_input["fare_2"])
    input_data['Cabin'] = form_input['cabin']
    input_data['Embarked'] = form_input['embarked']
    test = pd.DataFrame([pd.Series(input_data)])

    # 結合
    data = pd.concat([train, test], sort=False)

    # 前処理
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].fillna(('Nan'), inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'Nan': 99}).astype(int)

    delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
    data.drop(delete_columns, axis=1, inplace=True)

    train = data[:len(train)]
    test = data[len(train):]

    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    X_test = test.drop('Survived', axis=1)

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
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 予測確率は%表記で小数点2桁まで
    ret = round(y_pred[0] * 100, 2)

    return ret
