import pandas as pd
import lightgbm as lgb

model_path = "./app/static/model.txt"


def calc_ypred(form_input):

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

    # 前処理
    test['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    test['Embarked'].fillna(('Nan'), inplace=True)
    test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'Nan': 99}).astype(int)
    delete_columns = ['Name', 'Ticket', 'Cabin']
    test.drop(delete_columns, axis=1, inplace=True)

    # モデルを読み込んでテストデータを入力
    bst = lgb.Booster(model_file=model_path)
    y_pred = bst.predict(test, num_iteration=bst.best_iteration)

    # 小数点2桁で四捨五入(％表記)
    ret = round(y_pred[0] * 100, 2)

    return ret
