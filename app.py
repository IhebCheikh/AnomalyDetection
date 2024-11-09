from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)

# Initialiser le modèle Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.1)
is_trained = False


# Endpoint pour entraîner le modèle avec des données historiques
@app.route('/train', methods=['POST'])
def train_model():
    global is_trained
    data = request.json.get('data', [])
    if not data:
        return jsonify({'error': 'Aucune donnée fournie pour l\'entraînement.'}), 400

    # Convertir les données en un tableau NumPy
    X_train = np.array(data)
    model.fit(X_train)
    is_trained = True
    return jsonify({'message': 'Modèle entraîné avec succès.'})


# Endpoint pour détecter les anomalies
@app.route('/detect', methods=['POST'])
def detect_anomaly():
    if not is_trained:
        return jsonify({'error': 'Le modèle doit être entraîné avant de pouvoir détecter des anomalies.'}), 400

    data = request.json.get('data', [])
    if not data:
        return jsonify({'error': 'Aucune donnée fournie pour la détection d\'anomalies.'}), 400

    # Convertir les données en un tableau NumPy
    X_test = np.array(data)
    predictions = model.predict(X_test)
    anomalies = predictions == -1  # -1 indique une anomalie

    return jsonify({
        'anomalies': anomalies.tolist(),
        'message': 'Détection des anomalies terminée.'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
