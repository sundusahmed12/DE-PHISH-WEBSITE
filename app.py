import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
import featureExtraction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getResult', methods=['POST'])
def get_result():
    url = request.form['url']

    # Importing dataset
    df = pd.read_csv("URLDATA.csv")

    # Separating features and labels
    y = df['Label'].astype(int)
    X = df.drop('Label', axis=1)

    # Perform one-hot encoding for categorical features
    X = pd.get_dummies(X)

    # Separating training features, testing features, training labels & testing labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
    clf = xgb
    xgb.fit(X_train, y_train)
    y_test_xgb = xgb.predict(X_test)
    y_train_xgb = xgb.predict(X_train)
    score = clf.score(X_test, y_test)
    print(score * 100)

    X_new = []

    X_input = url
    X_new = featureExtraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(0, 1) 

    try:
        print(X_new)
        prediction = clf.predict(X_new)
        print(prediction)
        if prediction == 1:
            return jsonify({'result': 'Phishing URL'})
        else:
            return jsonify({'result': 'Legitimate URL'})
    except:
        return jsonify({'result': 'Phishing URL'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='0.0.0.0')
