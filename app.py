from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pickle
import numpy as np
from PIL import Image, ImageEnhance
import io
import os

app = FastAPI(title="Clasificador de Ropa - SVM")

MODEL_PATH  = "modelo_svm.pkl"
TRAIN_SIZE  = 30000

def entrenar_y_guardar():
    print("=" * 50)
    print("  Modelo no encontrado. Entrenando...")
    print("=" * 50)
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    print("[1/5] Descargando Fashion-MNIST...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    print(f"[2/5] Seleccionando {TRAIN_SIZE} muestras...")
    X_s, _, y_s, _ = train_test_split(X, y, train_size=TRAIN_SIZE, stratify=y, random_state=42)
    print("[3/5] Escalando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_s)
    print("[4/5] Dividiendo 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_s, test_size=0.2, random_state=42)
    print("[5/5] Entrenando SVM (paciencia)...")
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"\nPrecision: {accuracy:.4f} ({accuracy*100:.2f}%)")
    modelo_datos = {
        'modelo': clf, 'scaler': scaler, 'accuracy': accuracy,
        'train_size': TRAIN_SIZE,
        'clases': ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(modelo_datos, f)
    print(f"Modelo guardado: {MODEL_PATH}")
    return modelo_datos

if not os.path.exists(MODEL_PATH):
    datos = entrenar_y_guardar()
else:
    print(f"Cargando modelo existente: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        datos = pickle.load(f)

modelo   = datos["modelo"]
scaler   = datos["scaler"]
clases   = datos["clases"]
accuracy = datos.get("accuracy", 0)
print(f"Listo | Precision: {accuracy:.4f}")

EMOJIS = {
    "T-shirt/top":"👕","Trouser":"👖","Pullover":"🧥","Dress":"👗",
    "Coat":"🧣","Sandal":"👡","Shirt":"👔","Sneaker":"👟",
    "Bag":"👜","Ankle boot":"👢",
}

def preprocesar(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert("L")
    img = img.crop(img.getbbox())
    img = ImageEnhance.Contrast(img).enhance(2.5)
    canvas = Image.new("L", (max(img.size), max(img.size)), 0)
    offset = ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2)
    canvas.paste(img, offset)
    img = canvas.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255 - arr
    return arr.flatten().reshape(1, -1)

@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    if not imagen.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Debe ser una imagen.")
    contenido = await imagen.read()
    try:
        arr = preprocesar(contenido)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    arr_scaled = scaler.transform(arr)
    pred = modelo.predict(arr_scaled)[0]
    clase_nombre = clases[int(pred)]
    probabilidades = {}
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(arr_scaled)[0]
        probabilidades = {clases[i]: round(float(p)*100, 2) for i, p in enumerate(probs)}
    return JSONResponse({
        "clase_id": int(pred),
        "clase_nombre": clase_nombre,
        "emoji": EMOJIS.get(clase_nombre, "🏷️"),
        "probabilidades": probabilidades,
        "precision_modelo": round(float(accuracy)*100, 2),
    })

@app.get("/info")
async def info():
    return {"modelo": "SVM kernel RBF", "dataset": "Fashion-MNIST",
            "clases": clases, "precision": round(float(accuracy)*100, 2)}
