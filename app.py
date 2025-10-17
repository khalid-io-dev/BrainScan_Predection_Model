# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import tensorflow as tf
import cv2

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# CONFIG - change if needed
MODEL_CANDIDATES = [
    "models/best_cnn.h5",
    "models/best_model.h5",
    "models/final_model.h5",
    "models/best_model_weights.h5"
]
LABELS_PATH = "models/label_classes.npy"
DATA_SPLIT_TRAIN = Path("./Data_split/train")
IMG_SIZE = (224, 224)
# New: Path to saved figures (save plots from notebook as PNGs)
FIGURES_DIR = Path("figures")
PLOT1_PATH = FIGURES_DIR / "loss_acc_plot1.png"
PLOT2_PATH = FIGURES_DIR / "loss_acc_plot2.png"

st.title("Brain Tumor Classifier — Demo")
st.markdown(
    "Upload an image or provide a local path. The app will load a Keras model from `models/` and a `label_classes.npy` file."
)

# ---- Integrated Documentation: Model Performance Section ----
# This section integrates the training results as docs/explanations.
# It uses expanders for collapsible info, with markdown for plots/report/examples.
with st.expander("📊 Model Performance Overview (Training Results)"):
    st.markdown("""
    ### Training History Plots
    These plots show loss and accuracy over 20 epochs from the training process (generated in `train_cnn.py` or `train_and_evaluate_cell.py`).
    
    - **Plot 1** (Mild fluctuations in validation):
      - Loss: Train decreases smoothly from ~1.2 to ~0.1; Val from ~1.0 to ~0.3 with some ups/downs (mild overfitting).
      - Accuracy: Train rises to ~1.0; Val to ~0.9 with oscillations (possible due to batch size or data noise).
    """)
    if PLOT1_PATH.exists():
        st.image(str(PLOT1_PATH), caption="Training History Plot 1", use_column_width=True)
    else:
        st.info("Plot 1 image not found. Save as 'figures/loss_acc_plot1.png' from your notebook.")

    st.markdown("""
    - **Plot 2** (Smoother validation):
      - Similar trends, but less fluctuation in val curves (perhaps from a refined run).
    """)
    if PLOT2_PATH.exists():
        st.image(str(PLOT2_PATH), caption="Training History Plot 2", use_column_width=True)
    else:
        st.info("Plot 2 image not found. Save as 'figures/loss_acc_plot2.png' from your notebook.")

    st.markdown("""
    ### Classification Report (Test Set)
    Evaluated on 1271 samples (from `train_and_evaluate_cell.py`). Overall accuracy: 96.85%. Strong performance, but meningioma has slightly lower recall.
    
                precision    recall  f1-score   support
    glioma     0.9448    0.9448    0.9448       290
    meningioma     0.9414    0.9223    0.9317       296
    notumor     0.9925    1.0000    0.9962       395
    pituitary     0.9863    0.9966    0.9914       290
    accuracy                         0.9685      1271
    macro avg     0.9663    0.9659    0.9661      1271
    weighted avg     0.9683    0.9685    0.9684      1271 
                


    ### Example Correct Predictions
    The model shows high confidence on clear cases (from `train_and_evaluate_cell.py` and `save_and_predict.py`):

    Predicted: glioma  
    Top probabilities:  
      glioma: 0.9999  
      meningioma: 0.0001  
      notumor: 0.0000  
      pituitary: 0.0000

    This demonstrates the model's decisiveness. For visualization, 6 correct samples were shown in the notebook.
    """)

# ---- utility functions ----
@st.cache_resource
def load_tf_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_label_classes(label_path):
    if Path(label_path).exists():
        return list(np.load(label_path, allow_pickle=True))
    # infer from Data_split/train if available
    train_root = DATA_SPLIT_TRAIN
    if train_root.exists():
        class_dirs = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
        if class_dirs:
            # save for future runs
            os.makedirs(Path(label_path).parent, exist_ok=True)
            np.save(label_path, np.array(class_dirs, dtype=object))
            return class_dirs
    return None

def find_model():
    for p in MODEL_CANDIDATES:
        if Path(p).exists():
            return p
    # fallback: look for any .h5 in ./models
    models_dir = Path("models")
    if models_dir.exists():
        files = list(models_dir.glob("*.h5"))
        if files:
            return str(files[0])
    return None

def preprocess_bytes(image_bytes, img_size=IMG_SIZE):
    """Return batched numpy array (1,H,W,C) normalized to [0,1]"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Option 1: use PIL resize (keeps aspect) then convert to numpy
    img = img.resize(img_size, Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    # ensure shape (H,W,3)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr, img  # return PIL image for display

def predict_from_bytes(model, classes, image_bytes):
    X_batch, pil_img = preprocess_bytes(image_bytes, IMG_SIZE)
    probs = model.predict(X_batch, verbose=0)
    probs1 = probs[0] if probs.ndim==2 and probs.shape[0]==1 else probs.flatten()
    pred_idx = int(np.argmax(probs1))
    pred_label = classes[pred_idx] if classes and pred_idx < len(classes) else str(pred_idx)
    prob_pairs_sorted = sorted(list(zip(classes if classes else [str(i) for i in range(len(probs1))], probs1.tolist())),
                               key=lambda x: x[1], reverse=True)
    return {
        "pred_label": pred_label,
        "pred_index": pred_idx,
        "probs": probs1,
        "probs_sorted": prob_pairs_sorted,
        "pil_img": pil_img
    }

# ---- UI ----
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png","bmp"])
path_input = st.text_input("Or enter a local image path (leave blank if using upload)")

# Try locate model and labels
model_path = find_model()
if model_path is None:
    st.error("No model file found in `models/`. Put your .h5 model inside ./models and refresh.")
    st.stop()
else:
    st.info(f"Using model: {model_path}")

classes = load_label_classes(LABELS_PATH)
if classes is None:
    st.warning("Label file not found and could not infer from Data_split/train. Please create models/label_classes.npy or ensure Data_split/train exists.")
else:
    st.write(f"Detected classes: {classes}")

# load model (cached)
try:
    model = load_tf_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

test_path = None
image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.read()
    test_path = "uploaded"
elif path_input:
    test_path = path_input
    try:
        with open(test_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        st.error(f"Failed to read path: {e}")
        image_bytes = None

if st.button("Predict") and image_bytes is not None:
    try:
        with st.spinner("Running prediction..."):
            res = predict_from_bytes(model, classes, image_bytes)
        st.success(f"Predicted: **{res['pred_label']}** (index {res['pred_index']})")
        st.image(res["pil_img"], caption="Input image", use_column_width=True)
        st.subheader("Top probabilities")
        # show sorted probabilities
        for cls, p in res["probs_sorted"]:
            st.write(f"{cls}: {p:.4f}")
        # bar chart using pandas for nicer ordering
        try:
            import pandas as pd
            df = pd.DataFrame(res["probs_sorted"], columns=["class","prob"])
            df = df.set_index("class")
            st.bar_chart(df)
        except Exception:
            st.write(res["probs_sorted"])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    if image_bytes is None:
        st.info("Upload an image or enter a path and click Predict.")