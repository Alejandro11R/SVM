# SVM — Clasificador de Ropa con SVM

## Estructura del proyecto
```
svm_fashion/
├── entrenar_modelo.py   ← Paso 1: entrenar y guardar el .pkl
├── app.py               ← Paso 2: servidor FastAPI
├── requirements.txt     ← dependencias
└── templates/
    └── index.html       ← interfaz web
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Paso 1 — Entrenar y guardar el modelo
```bash
python entrenar_modelo.py
```
Esto genera `modelo_svm.pkl` (~puede tardar 2-5 minutos).

### Paso 2 — Levantar el servidor
```bash
uvicorn app:app --reload
```

### Paso 3 — Abrir el navegador
```
http://localhost:8000
```

Sube una imagen de ropa y el modelo predice la categoría.

## Endpoint API
- `GET  /`          → Interfaz web
- `POST /predecir`  → Recibe imagen, retorna predicción JSON
- `GET  /info`      → Info del modelo

## Categorías detectadas
| ID | Categoría     |
|----|---------------|
| 0  | T-shirt/top   |
| 1  | Trouser       |
| 2  | Pullover      |
| 3  | Dress         |
| 4  | Coat          |
| 5  | Sandal        |
| 6  | Shirt         |
| 7  | Sneaker       |
| 8  | Bag           |
| 9  | Ankle boot    |

## Nota importante
El modelo fue entrenado con Fashion-MNIST (imágenes 28×28, escala de grises,
fondo oscuro). Fotos reales de prendas tendrán menor precisión que imágenes
del dataset original. Para mejores resultados usa imágenes con:
- Fondo blanco o muy claro
- La prenda centrada
- Sin texto ni ruido visual
