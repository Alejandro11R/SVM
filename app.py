from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pickle
import numpy as np
from PIL import Image, ImageEnhance
import io
import os

app = FastAPI(title="Clasificador de Ropa - SVM")

HF_REPO   = "xAlejo/svm-fashion"
HF_FILE   = "modelo_svm.pkl"
MODEL_PATH = "/tmp/modelo_svm.pkl"

print("Descargando modelo desde Hugging Face...")
from huggingface_hub import hf_hub_download
ruta = hf_hub_download(
    repo_id=HF_REPO,
    filename=HF_FILE,
    repo_type="model",
    force_download=True,
    local_dir="/tmp"
)
print(f"Modelo en: {ruta}")

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
    esquinas = [
        img.getpixel((0, 0)),
        img.getpixel((img.width-1, 0)),
        img.getpixel((0, img.height-1)),
        img.getpixel((img.width-1, img.height-1)),
    ]
    fondo_promedio = sum(esquinas) / len(esquinas)
    if fondo_promedio > 128:
        img = Image.fromarray(255 - np.array(img))
    arr = np.array(img, dtype=np.float32)
    umbral = arr.max() * 0.20
    arr[arr < umbral] = 0
    img = Image.fromarray(arr.astype(np.uint8))
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = ImageEnhance.Contrast(img).enhance(3.0)
    lado = max(img.size)
    padding = int(lado * 0.1)
    canvas_size = lado + padding * 2
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    offset = ((canvas_size - img.width) // 2, (canvas_size - img.height) // 2)
    canvas.paste(img, offset)
    img = canvas.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 0:
        arr = arr / arr.max() * 255
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
        "clase_id": int(pred), "clase_nombre": clase_nombre,
        "emoji": EMOJIS.get(clase_nombre, "🏷️"),
        "probabilidades": probabilidades,
        "precision_modelo": round(float(accuracy)*100, 2),
    })

@app.get("/info")
async def info():
    return {"modelo": "SVM kernel RBF", "dataset": "Fashion-MNIST",
            "clases": clases, "precision": round(float(accuracy)*100, 2)}
