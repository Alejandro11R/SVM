from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pickle
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import threading

app = FastAPI(title="Clasificador de Ropa - SVM")

MODEL_PATH = "modelo_svm.pkl"
TRAIN_SIZE = 30000

modelo     = None
scaler     = None
clases     = None
accuracy   = 0
entrenando = False
estado_msg = "iniciando"

def entrenar_y_guardar():
    global modelo, scaler, clases, accuracy, entrenando, estado_msg
    entrenando = True
    try:
        print("=" * 50)
        print("  Modelo no encontrado. Entrenando...")
        print("=" * 50)
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        import tensorflow.keras as keras

        print("[1/5] Descargando Fashion-MNIST via keras...")
        estado_msg = "descargando datos"
        (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.fashion_mnist.load_data()
        X_all = np.concatenate([X_train_full, X_test_full]).reshape(-1, 784).astype(np.float32)
        y_all = np.concatenate([y_train_full, y_test_full]).astype(str)

        print(f"[2/5] Seleccionando {TRAIN_SIZE} muestras...")
        estado_msg = "preparando datos"
        X_s, _, y_s, _ = train_test_split(X_all, y_all, train_size=TRAIN_SIZE, stratify=y_all, random_state=42)

        print("[3/5] Escalando datos...")
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X_s)

        print("[4/5] Dividiendo 80/20...")
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_s, test_size=0.2, random_state=42)

        print("[5/5] Entrenando SVM...")
        estado_msg = "entrenando SVM"
        clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        print(f"\nPrecision: {acc:.4f} ({acc*100:.2f}%)")

        nombres = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                   'Sandal','Shirt','Sneaker','Bag','Ankle boot']
        datos = {'modelo': clf, 'scaler': sc, 'accuracy': acc,
                 'train_size': TRAIN_SIZE, 'clases': nombres}

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(datos, f)

        modelo   = clf
        scaler   = sc
        clases   = nombres
        accuracy = acc
        estado_msg = "listo"
        print(f"Modelo guardado: {MODEL_PATH}")

    except Exception as e:
        estado_msg = f"error: {str(e)}"
        print(f"ERROR en entrenamiento: {e}")
    finally:
        entrenando = False

if os.path.exists(MODEL_PATH):
    print(f"Cargando modelo existente: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        datos = pickle.load(f)
    modelo   = datos["modelo"]
    scaler   = datos["scaler"]
    clases   = datos["clases"]
    accuracy = datos.get("accuracy", 0)
    estado_msg = "listo"
    print(f"Listo | Precision: {accuracy:.4f}")
else:
    hilo = threading.Thread(target=entrenar_y_guardar, daemon=True)
    hilo.start()

EMOJIS = {
    "T-shirt/top":"👕","Trouser":"👖","Pullover":"🧥","Dress":"👗",
    "Coat":"🧣","Sandal":"👡","Shirt":"👔","Sneaker":"👟",
    "Bag":"👜","Ankle boot":"👢",
}

def preprocesar(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert("L")
    
    # 1. Detectar si el fondo es claro u oscuro
    esquinas = [
        img.getpixel((0, 0)),
        img.getpixel((img.width-1, 0)),
        img.getpixel((0, img.height-1)),
        img.getpixel((img.width-1, img.height-1)),
    ]
    fondo_promedio = sum(esquinas) / len(esquinas)
    
    # 2. Si el fondo es claro (foto real), invertir para que objeto sea blanco
    if fondo_promedio > 128:
        img = Image.fromarray(255 - np.array(img))
    
    # 3. Umbralizar para eliminar fondos grises y sombras
    arr = np.array(img, dtype=np.float32)
    # Todo lo que sea menor al 20% del máximo lo ponemos en negro
    umbral = arr.max() * 0.20
    arr[arr < umbral] = 0
    img = Image.fromarray(arr.astype(np.uint8))
    
    # 4. Recortar solo el objeto
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # 5. Aumentar contraste para que el objeto sea bien visible
    img = ImageEnhance.Contrast(img).enhance(3.0)
    
    # 6. Centrar en canvas cuadrado con padding (como Fashion-MNIST)
    lado = max(img.size)
    padding = int(lado * 0.1)
    canvas_size = lado + padding * 2
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    offset = (
        (canvas_size - img.width) // 2,
        (canvas_size - img.height) // 2
    )
    canvas.paste(img, offset)
    
    # 7. Resize a 28x28
    img = canvas.resize((28, 28), Image.LANCZOS)
    
    # 8. Normalizar a rango Fashion-MNIST (objeto blanco, fondo negro)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    
    return arr.flatten().reshape(1, -1)

@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/estado")
async def estado_endpoint():
    return {"estado": estado_msg, "listo": modelo is not None,
            "precision": round(float(accuracy)*100, 2) if accuracy else 0}

@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    if modelo is None:
        raise HTTPException(status_code=503, detail=f"Modelo aún no listo: {estado_msg}")
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
        "clase_id": int(pred), "clase_nombre": clase_nombre,
        "emoji": EMOJIS.get(clase_nombre, "🏷️"),
        "probabilidades": probabilidades,
        "precision_modelo": round(float(accuracy)*100, 2),
    })

@app.get("/info")
async def info():
    return {"modelo": "SVM kernel RBF", "dataset": "Fashion-MNIST",
            "clases": clases, "precision": round(float(accuracy)*100, 2)}
