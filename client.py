import requests

url_logistic_regression = 'http://localhost:5000/predict_logistic_regression'
url_svm = 'http://localhost:5000/predict_svm'
url_decision_tree = 'http://localhost:5000/predict_decision_tree'
url_knn = 'http://localhost:5000/predict_knn'

data = {'petal_length': 1.5, 'petal_width': 0.5}

response_logistic_regression = requests.post(url_logistic_regression, json=data)
prediction_logistic_regression = response_logistic_regression.json()['prediction']
print(f'Predicción (Regresión Logística): {prediction_logistic_regression}')

response_svm = requests.post(url_svm, json=data)
prediction_svm = response_svm.json()['prediction']
print(f'Predicción (SVM): {prediction_svm}')

response_decision_tree = requests.post(url_decision_tree, json=data)
prediction_decision_tree = response_decision_tree.json()['prediction']
print(f'Predicción (Árbol de Decisión): {prediction_decision_tree}')

response_knn = requests.post(url_knn, json=data)
prediction_knn = response_knn.json()['prediction']
print(f'Predicción (k-Nearest Neighbors): {prediction_knn}')
