"""
PASO 2: Ejecutar el servidor FastAPI.
Comando: uvicorn app:app --reload
Luego abrir: http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from PIL import Image, ImageOps
import io
import os

app = FastAPI(title="Clasificador de Ropa - SVM")

# ─── Cargar modelo al iniciar ───────────────────────────────────────────────
MODEL_PATH = "modelo_svm.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"❌ No se encontró '{MODEL_PATH}'. "
        "Primero ejecuta: python entrenar_modelo.py"
    )

with open(MODEL_PATH, "rb") as f:
    datos = pickle.load(f)

modelo  = datos["modelo"]
scaler  = datos["scaler"]
clases  = datos["clases"]
accuracy = datos.get("accuracy", "N/A")

print(f" Modelo cargado | Precisión: {accuracy:.4f}")

# Emojis para cada clase
EMOJIS = {
    "T-shirt/top": "👕",
    "Trouser":     "👖",
    "Pullover":    "🧥",
    "Dress":       "👗",
    "Coat":        "🧣",
    "Sandal":      "👡",
    "Shirt":       "👔",
    "Sneaker":     "👟",
    "Bag":         "👜",
    "Ankle boot":  "👢",
}

# ─── Preprocesar imagen ──────────────────────────────────────────────────────
from PIL import ImageEnhance, ImageFilter

def preprocesar(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert("L")  # escala de grises
    
    # Recortar bordes vacíos (elimina fondo sobrante)
    img = img.crop(img.getbbox())
    
    # Aumentar contraste para destacar la prenda del fondo
    img = ImageEnhance.Contrast(img).enhance(2.5)
    
    # Centrar en un canvas cuadrado con fondo negro (como Fashion-MNIST)
    canvas = Image.new("L", (max(img.size), max(img.size)), 0)
    offset = ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2)
    canvas.paste(img, offset)
    
    # Redimensionar a 28x28
    img = canvas.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    
    # Invertir si el fondo es claro
    if arr.mean() > 127:
        arr = 255 - arr
        
    return arr.flatten().reshape(1, -1)


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    # Validar tipo de archivo
    if not imagen.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contenido = await imagen.read()

    try:
        arr = preprocesar(contenido)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

    arr_scaled = scaler.transform(arr)
    pred = modelo.predict(arr_scaled)[0]
    clase_nombre = clases[int(pred)]

    # Probabilidades (si el modelo las soporta)
    probabilidades = {}
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(arr_scaled)[0]
        probabilidades = {
            clases[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }

    return JSONResponse({
        "clase_id":       int(pred),
        "clase_nombre":   clase_nombre,
        "emoji":          EMOJIS.get(clase_nombre, "🏷️"),
        "probabilidades": probabilidades,
        "precision_modelo": round(float(accuracy) * 100, 2),
    })


@app.get("/info")
async def info():
    return {
        "modelo":    "SVM kernel lineal",
        "dataset":   "Fashion-MNIST",
        "clases":    clases,
        "precision": round(float(accuracy) * 100, 2),
    }
