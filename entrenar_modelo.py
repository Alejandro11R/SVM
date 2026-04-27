"""
PASO 1: Ejecutar este script primero para entrenar y guardar el modelo.
Comando: python entrenar_modelo.py
"""

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

print("=" * 50)
print("  ENTRENAMIENTO MODELO SVM - Fashion MNIST")
print("=" * 50)

print("\n[1/5] Cargando datos Fashion-MNIST...")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

print("[2/5] Reduciendo dataset para entrenamiento rápido...")
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=10000, stratify=y, random_state=42
)

print("[3/5] Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

print("[4/5] Dividiendo en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_small, test_size=0.2, random_state=42
)

print("[5/5] Entrenando SVM (puede tardar unos minutos)...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"\n✅ Precisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Guardar modelo + scaler juntos en un .pkl
modelo_datos = {
    'modelo': clf,
    'scaler': scaler,
    'accuracy': accuracy,
    'clases': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
}

with open('modelo_svm.pkl', 'wb') as f:
    pickle.dump(modelo_datos, f)

print(" Modelo guardado como: modelo_svm.pkl")
print("\nAhora ejecuta: uvicorn app:app --reload")
