from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

app = Flask(__name__)

# Charger le fichier CSV et entraîner le modèle lors du démarrage
file_path = "processed_iot_telemetry_data.csv"  # Chemin fixe pour le fichier CSV sous la racine du projet
data = pd.read_csv(file_path, sep=';')  # Chargement des données avec le bon séparateur
print(data.head(10))

# Vérifiez si les colonnes sont correctement chargées
if set(['humidity', 'light', 'temp']).issubset(data.columns):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)  # Configuration du modèle
    model.fit(data)  # Entraînement du modèle
else:
    print("Les colonnes attendues ne sont pas présentes dans les données")

"""data = pd.read_csv(file_path)
print(data.head(10))# Chargement des données
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)  # Configuration du modèle
model.fit(data)  # Entraînement du modèle"""


@app.route('/detect', methods=['POST'])
def detect_anomalies():
    if model is None:
        return jsonify({"error": "Le modèle n'est pas initialisé"}), 500

    data = request.json.get('data')
    if not data:
        return jsonify({"error": "Aucune donnée fournie"}), 400

    # Convertir les données en tableau numpy
    try:
        data_array = np.array(data)
        print(data_array)
        predictions = model.predict(data_array)  # -1 pour anomalie, 1 pour normal
        anomalies = [True if pred == -1 else False for pred in predictions]

        return jsonify({"anomalies": anomalies}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
