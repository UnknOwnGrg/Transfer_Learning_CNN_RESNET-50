# Cats vs Dogs: CNN with Transfer Learning (ResNet-50)

An image classifier that tells cats from dogs, built with TensorFlow/Keras on top of a ResNet-50 pretrained on ImageNet. It is written for the Kaggle competition [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition). For every test image it predicts the **probability that the image is a dog** (1 = dog, 0 = cat).

All the work is in one notebook: [`cnn_using_resnet50.ipynb`](cnn_using_resnet50.ipynb).

## How it works

1. **Load the data.** It unzips `train.zip` (25,000 labelled images such as `cat.123.jpg` and `dog.45.jpg`) and `test.zip` (12,500 unlabelled images such as `1.jpg`). Extraction is skipped if the folders already exist.
2. **Sample and split.** It takes a shuffled sample of 20,000 training images and splits it 80/20 into train and validation sets, keeping the same cat/dog ratio in both (stratified).
3. **Input pipeline.** A `tf.data` pipeline reads images in batches, resizes them to 224×224 and applies ResNet-50's own `preprocess_input`.
4. **Model.**
   ```
   Input (224×224×3)
     → Data augmentation (random flip, rotation, zoom)
     → ResNet-50 (ImageNet weights, no top)
     → GlobalAveragePooling2D
     → Dropout(0.2)
     → Dense(1, sigmoid)   # P(dog)
   ```
5. **Training in two phases.**
   - **Phase 1, feature extraction:** ResNet-50 is frozen and only the new top layer is trained (Adam, lr = 1e-3). `EarlyStopping` stops training when validation loss stops improving.
   - **Phase 2, fine-tuning:** the last ResNet block (`conv5_*`) is unfrozen and trained with a learning rate 100× smaller (1e-5). BatchNorm layers stay frozen.
6. **Evaluation.** Training curves, validation accuracy, a classification report, a confusion matrix, and a grid of the images the model got wrong.
7. **Output.** It predicts the test set, clips the probabilities to [0.01, 0.99] (the competition is scored with log loss) and writes `submission.csv`. It also saves the trained model as `cats_dogs_resnet50.keras`.

## Project structure

```
Cnn/
├── cnn_using_resnet50.ipynb   # the full pipeline: data, model, training, submission
├── pyproject.toml             # project metadata and dependencies
├── uv.lock                    # locked dependency versions (uv)
├── backend/                   # FastAPI server that serves the model
├── frontend/                  # web page to upload a photo and see the prediction
├── submission.csv             # Kaggle submission: id,label  (generated)
├── cats_dogs_resnet50.keras   # trained model (generated, not in git)
├── train.zip / test.zip       # dataset from Kaggle (not in git)
└── train/ , test/             # extracted images (generated, not in git)
```

## Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Main libraries: TensorFlow, NumPy, pandas, scikit-learn, matplotlib, seaborn, OpenCV, ipykernel
- A GPU is optional but makes training much faster. On a CPU, expect each epoch to take a long time. Note that native TensorFlow on Windows runs on the CPU only; for GPU training on Windows, use WSL2.
- Internet access on the first run, so Keras can download the ImageNet weights for ResNet-50 (about 95 MB).

## Setup

### 1. Get the code

```bash
git clone <your-repo-url>
cd Cnn
```

### 2. Install the dependencies

With **uv** (uses the exact versions in `uv.lock`):

```bash
uv sync
```

Or with **pip**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python ipykernel jupyter
```

### 3. Download the dataset

1. Open the [competition data page](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) and accept the rules.
2. Download `train.zip` and `test.zip`.
3. Put both files in the project folder (next to the notebook). You do not need to unzip them; the notebook does it.

Or with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
```

### 4. Set the paths

In the configuration cell of the notebook, point `DATA_DIR` and `WORK_DIR` to your project folder:

```python
DATA_DIR = r"C:\path\to\Cnn"   # folder that holds train.zip and test.zip
WORK_DIR = r"C:\path\to\Cnn"   # folder where train/ and test/ are extracted
```

## Running

Open the notebook in Jupyter, VS Code or PyCharm and select the project's `.venv` as the kernel:

```bash
uv run jupyter notebook cnn_using_resnet50.ipynb
# or, if you installed with pip and activated .venv:
jupyter notebook cnn_using_resnet50.ipynb
```

Then run all cells from top to bottom (**Run → Run All Cells**). When it finishes you will have:

- `submission.csv`, ready to upload to Kaggle
- `cats_dogs_resnet50.keras`, the trained model

## Configuration

These values are at the top of the notebook:

| Setting       | Default | Meaning                                            |
|---------------|---------|----------------------------------------------------|
| `SEED`        | 2018    | Random seed for the train/validation split         |
| `VAL_size`    | 0.2     | Share of the sample used for validation            |
| `BATCH_SIZE`  | 32      | Images per batch                                   |
| `IMG_SIZE`    | 224     | Input size (ResNet-50 was trained on 224×224)      |
| `SAMPLE_SIZE` | 20000   | Training images to use (`None` = all 25,000)       |
| `EPOCHS_HEAD` | 5       | Max epochs for phase 1 (frozen ResNet)             |
| `EPOCHS_FINE` | 3       | Intended epochs for phase 2 (fine-tuning)          |

## Using the saved model

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

model = keras.models.load_model("cats_dogs_resnet50.keras")

img = tf.io.decode_jpeg(tf.io.read_file("test/1.jpg"), channels=3)
img = tf.image.resize(img, (224, 224))
img = preprocess_input(img)[tf.newaxis, ...]

p_dog = float(model.predict(img)[0, 0])
print("Dog" if p_dog > 0.5 else "Cat", f"(P(dog) = {p_dog:.2f})")
```

## Web app

You can also try the trained model in the browser: upload a photo and the page shows "Cat" or "Dog" with the confidence. It has two parts:

- **[`backend/`](backend/README.md):** a FastAPI server that loads `cats_dogs_resnet50.keras` and exposes `POST /predict`.
- **[`frontend/`](frontend/README.md):** a plain HTML/CSS/JS page that sends the photo to the backend and shows the result.

You need the trained model file first. Run the notebook to create it.

Then start both parts, each in its own terminal:

```bash
# Terminal 1: backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Terminal 2: frontend
cd frontend
python -m http.server 5500
```

Open http://127.0.0.1:5500 in your browser. The API docs are at http://127.0.0.1:8000/docs.

See each folder's README for the endpoints and configuration options.

## Submission format

```
id,label
1,0.99
2,0.99
...
```

`label` is the predicted probability that the image is a dog.

## References

1. [Dogs vs. Cats Redux: Kernels Edition (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
2. [Keras guide: Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)
3. He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
4. [TensorFlow guide: Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance)
