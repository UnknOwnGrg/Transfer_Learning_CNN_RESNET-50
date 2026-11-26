"""FastAPI backend that serves the Cats vs Dogs ResNet-50 model.

Run from the backend/ folder:
    uvicorn app:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 224  # must match the size used in the notebook
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# The model file lives in the project root by default (one level above backend/).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "cats_dogs_resnet50.keras"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

# Frontend addresses that are allowed to call this API.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500"
).split(",")

model: keras.Model | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once when the server starts, not on every request."""
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Train it with the notebook "
            "or set the MODEL_PATH environment variable."
        )
    model = keras.models.load_model(MODEL_PATH)
    yield


app = FastAPI(title="Cats vs Dogs API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def preprocess(image_bytes: bytes) -> tf.Tensor:
    """Same steps as the notebook: decode as RGB, resize to 224x224, ResNet preprocessing."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img[tf.newaxis, ...]  # add batch dimension -> (1, 224, 224, 3)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Upload an image; returns the predicted label and P(dog)."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")

    try:
        x = preprocess(data)
    except tf.errors.InvalidArgumentError:
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Use JPEG, PNG, BMP or GIF.",
        )

    p_dog = float(model(x, training=False).numpy()[0, 0])
    label = "dog" if p_dog > 0.5 else "cat"

    return {
        "label": label,
        "p_dog": round(p_dog, 4),
        "confidence": round(max(p_dog, 1 - p_dog), 4),
    }
