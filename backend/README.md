# Backend: Cats vs Dogs API

A FastAPI server that loads the trained model (`cats_dogs_resnet50.keras`) and predicts whether an uploaded image is a cat or a dog.

## Setup

From the project root, with the project's virtual environment active:

```bash
cd backend
pip install -r requirements.txt
```

The model file must exist. By default the server looks for it in the project root (`../cats_dogs_resnet50.keras`). It is created by the last cells of the notebook and is not in git because of its size.

## Run

```bash
uvicorn app:app --reload --port 8000
```

- API: http://127.0.0.1:8000
- Interactive docs (try uploads in the browser): http://127.0.0.1:8000/docs

Loading the model takes a few seconds at startup.

## Endpoints

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

Send an image as multipart form data in a field named `file`.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@../test/1.jpg"
```

Response:

```json
{"label": "dog", "p_dog": 0.9731, "confidence": 0.9731}
```

| Field        | Meaning                                    |
|--------------|--------------------------------------------|
| `label`      | `"dog"` if `p_dog > 0.5`, otherwise `"cat"`|
| `p_dog`      | Probability that the image is a dog (0–1)  |
| `confidence` | Probability of the predicted label (0.5–1) |

Errors: `415` (not an image), `400` (empty or unreadable image), `413` (larger than 10 MB).

## Configuration (optional)

Set these as environment variables if the defaults don't fit:

| Variable          | Default                                        | Meaning                              |
|-------------------|------------------------------------------------|--------------------------------------|
| `MODEL_PATH`      | `../cats_dogs_resnet50.keras`                  | Path to the trained model            |
| `ALLOWED_ORIGINS` | `http://localhost:5500,http://127.0.0.1:5500`  | Frontend URLs allowed to call the API (comma-separated) |
