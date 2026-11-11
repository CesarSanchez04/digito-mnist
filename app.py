import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps, ImageFilter

app = FastAPI()


def build_model():  # misma arquitectura usada en el entrenamiento
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.load_weights("midigito.weights.h5")
    return model


model = build_model()


def otsu(a):
    hist = np.bincount(a.astype(np.uint8).ravel(), minlength=256).astype(float)
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256))
    return np.argmax((mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12))

def preprocesar(img):
    """Convierte una foto (digito escrito sobre papel) al formato MNIST 28x28."""
    img = ImageOps.exif_transpose(img).convert("L")
    img.thumbnail((300, 300))
    fondo = img.filter(ImageFilter.GaussianBlur(max(img.size) / 12))  # iluminacion del papel
    a, f = np.asarray(img, np.float32), np.asarray(fondo, np.float32)
    oscuro, claro = np.clip(f - a, 0, None), np.clip(a - f, 0, None)
    a = oscuro if oscuro.sum() >= claro.sum() else claro  # trazo oscuro sobre papel claro (o al reves)
    a = a / (a.max() + 1e-6) * 255
    a = np.where(a > max(otsu(a), 40), a, 0)
    m = max(2, int(0.03 * min(a.shape))); a[:m], a[-m:], a[:, :m], a[:, -m:] = 0, 0, 0, 0  # ignora bordes
    filas = np.where((a > 0).sum(1) > 0.01 * a.shape[1])[0]
    cols = np.where((a > 0).sum(0) > 0.01 * a.shape[0])[0]
    if len(filas) and len(cols):
        a = a[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]
    a = np.where(a > 0, 255, 0).astype(np.uint8)
    h, w = a.shape
    s = 20.0 / max(h, w)
    d = Image.fromarray(a)
    d = d.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    lienzo = Image.new("L", (28, 28))
    lienzo.paste(d, ((28 - d.width) // 2, (28 - d.height) // 2))
    return np.asarray(lienzo, np.float32).reshape(1, 28, 28, 1) / 255.0


@app.get("/")
def inicio():
    return {"mensaje": "Servicio de reconocimiento de digitos (MNIST). Use POST /predict/"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        prob = model.predict(preprocesar(img), verbose=0)[0]
        digito = int(np.argmax(prob))
        return {"prediction": [str(digito)], "confianza": float(prob[digito])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
