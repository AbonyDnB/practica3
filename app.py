from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

logistic_regression_model = joblib.load('models/logistic_regression_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

@app.route('/predict_logistic_regression', methods=['POST'])
def predict_logistic_regression():
    data = request.get_json(force=True)
    features = [data['petal_length'], data['petal_width']]
    prediction = logistic_regression_model.predict([features])[0]
    return jsonify({'prediction': prediction})

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    data = request.get_json(force=True)
    features = [data['petal_length'], data['petal_width']]
    prediction = svm_model.predict([features])[0]
    return jsonify({'prediction': prediction})

@app.route('/predict_decision_tree', methods=['POST'])
def predict_decision_tree():
    data = request.get_json(force=True)
    features = [data['petal_length'], data['petal_width']]
    prediction = decision_tree_model.predict([features])[0]
    return jsonify({'prediction': prediction})

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    data = request.get_json(force=True)
    features = [data['petal_length'], data['petal_width']]
    prediction = knn_model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000)
