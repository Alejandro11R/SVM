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
print("  Version mejorada - ~91% de precision")
print("=" * 50)

print("\n[1/5] Cargando datos Fashion-MNIST...")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
print(f"      Dataset completo: {X.shape[0]} imagenes")

# ─── CAMBIO PRINCIPAL: de 10,000 a 60,000 muestras ──────────────────────────
# Con 10,000  -> ~82% de precision  (entrena en ~5 min)
# Con 30,000  -> ~88% de precision  (entrena en ~20 min)
# Con 60,000  -> ~91% de precision  (entrena en ~50 min)
# Con 70,000  -> ~92% de precision  (entrena en ~70 min)
TRAIN_SIZE = 60000  # <-- ajusta este numero segun el tiempo que tengas

print(f"[2/5] Usando {TRAIN_SIZE} muestras para entrenamiento...")
X_small, _, y_small, _ = train_test_split(
    X, y,
    train_size=TRAIN_SIZE,
    stratify=y,
    random_state=42
)
print(f"      Muestras seleccionadas: {X_small.shape[0]}")

print("[3/5] Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

print("[4/5] Dividiendo en entrenamiento y prueba (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_small,
    test_size=0.2,
    random_state=42
)
print(f"      Entrenamiento: {X_train.shape[0]} imagenes")
print(f"      Prueba:        {X_test.shape[0]} imagenes")

# ─── CAMBIO SECUNDARIO: kernel RBF en lugar de linear ───────────────────────
# kernel='linear' -> mas rapido, menor precision
# kernel='rbf'    -> mas lento, mayor precision (~+3-5%)
# C=10          -> penaliza mas los errores
# gamma='scale' -> escala automaticamente segun numero de features
print("[5/5] Entrenando SVM con kernel RBF (esto tarda, ten paciencia)...")
clf = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"\nPrecision del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")

if accuracy >= 0.90:
    print("   Excelente! Superaste el 90%")
elif accuracy >= 0.85:
    print("   Muy bien! Entre 85-90%")
else:
    print("   Considera aumentar TRAIN_SIZE para mejor precision")

# Guardar modelo + scaler + metadatos
modelo_datos = {
    'modelo':      clf,
    'scaler':      scaler,
    'accuracy':    accuracy,
    'train_size':  TRAIN_SIZE,
    'kernel':      'rbf',
    'clases': [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
}

with open('modelo_svm.pkl', 'wb') as f:
    pickle.dump(modelo_datos, f)

print("\nModelo guardado como: modelo_svm.pkl")
print("Ahora ejecuta: uvicorn app:app --reload")