import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict

MODELS = Path(__file__).resolve().parent.parent / "models"
state = {}


@asynccontextmanager
async def lifespan(app):
    # Chapter 2: load the trained model and the scaler ONCE, when the server starts
    import keras
    state["model"] = keras.models.load_model(MODELS / "credit_mlp.keras")
    sc = json.loads((MODELS / "scaler.json").read_text())
    state["columns"] = sc["columns"]
    state["mean"] = np.array(sc["mean"], dtype="float32")
    state["sd"] = np.array(sc["sd"], dtype="float32")
    # Chapter 3: the threshold chosen with the bank's cost rule (D3-07). No meta.json yet -> 0.5 like Class 2
    meta = MODELS / "meta.json"
    state["threshold"] = json.loads(meta.read_text())["threshold"] if meta.exists() else 0.5
    # Chapter 4: the rice leaf CNN (only when rice_cnn.keras is in models/)
    if (MODELS / "rice_cnn.keras").exists():
        rice = json.loads((MODELS / "rice_meta.json").read_text())
        state["classes"] = rice["classes"]
        state["cnn"] = keras.models.load_model(MODELS / "rice_cnn.keras")
        state["see"] = keras.Model(state["cnn"].inputs, state["cnn"].get_layer("relu_16").output)
        # Chapter 5: rice_meta.json names the best model (MobileNetV2) and its not-sure threshold
        best = rice.get("model_file", "rice_cnn.keras")
        state["rice"] = state["cnn"] if best == "rice_cnn.keras" else keras.models.load_model(MODELS / best)
        state["rice_model"] = rice.get("model_name", "CNN (Class 4)")
        state["rice_threshold"] = rice.get("not_sure_threshold", 0.0)
    yield
    state.clear()


app = FastAPI(title="Practical Deep Learning - credit network and rice leaf doctor", lifespan=lifespan)

# ---------- Chapter 1: hand-traced 2-3-2 network (unchanged) ----------
# Same weights as notebook cell D1-09
W1 = np.array([[0.5, -0.3, 0.8],
               [0.2,  0.4, -0.6]])
b1 = np.array([0.1, -0.1, 0.2])
W2 = np.array([[ 0.7, -0.5],
               [-0.2,  0.6],
               [ 0.3,  0.9]])
b2 = np.array([0.0, 0.1])


class Inputs(BaseModel):
    x1: float
    x2: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "threshold": state.get("threshold"),
            "rice_cnn_loaded": "rice" in state, "rice_model": state.get("rice_model"),
            "rice_threshold": state.get("rice_threshold")}


@app.post("/forward")
def forward(inp: Inputs):
    x = np.array([inp.x1, inp.x2])
    a1 = np.maximum(0, x @ W1 + b1)      # hidden layer: ReLU
    z2 = a1 @ W2 + b2                    # output logits
    p = np.exp(z2 - z2.max())            # softmax (stable form)
    p = p / p.sum()
    return {"probabilities": p.round(4).tolist()}


# ---------- Chapter 2: trained credit network ----------
class Client(BaseModel):
    LIMIT_BAL: float
    SEX: float
    EDUCATION: float
    MARRIAGE: float
    AGE: float
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    model_config = ConfigDict(json_schema_extra={"example": {
        "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2,
        "AGE": 46, "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
        "PAY_4": -1, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 1540,
        "BILL_AMT2": 26060, "BILL_AMT3": 0, "BILL_AMT4": 2650, "BILL_AMT5": 3320,
        "BILL_AMT6": 2764, "PAY_AMT1": 26060, "PAY_AMT2": 0, "PAY_AMT3": 3472,
        "PAY_AMT4": 2320, "PAY_AMT5": 1764, "PAY_AMT6": 2841}})


@app.post("/predict")
def predict(c: Client, threshold: float | None = Query(None, ge=0.0, le=1.0)):
    t = state["threshold"] if threshold is None else threshold     # Chapter 3: 0.17 unless the caller asks
    row = np.array([[getattr(c, col) for col in state["columns"]]], dtype="float32")
    xs = (row - state["mean"]) / state["sd"]   # same as D2-04
    p = float(state["model"].predict(xs, verbose=0)[0, 1])
    return {"p_default": round(p, 6), "predicted_class": int(p >= t), "threshold": t}


# ---------- Chapter 4: rice leaf CNN ----------
def read_leaf(data: bytes):
    """Photo bytes -> 1 x 160 x 160 x 3, exactly like load_image in notebook cell D4-04."""
    if "rice" not in state:
        raise HTTPException(503, "rice_cnn.keras is not in models/ yet (Chapter 4)")
    try:
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
    except Exception:
        raise HTTPException(400, "This file is not a photo")
    return tf.image.resize(img, [160, 160])[None]


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), threshold: float | None = Query(None, ge=0.0, le=1.0),
                        model: str = Query("best", pattern="^(best|cnn)$")):
    x = read_leaf(await file.read())
    p = state["rice" if model == "best" else "cnn"].predict(x, verbose=0)[0]
    t = state["rice_threshold"] if threshold is None else threshold    # Chapter 5: below t -> "not_sure"
    top = state["classes"][int(p.argmax())]
    return {"top_class": top, "probability": round(float(p.max()), 4),
            "probabilities": dict(zip(state["classes"], p.round(4).tolist())),
            "decision": top if p.max() >= t else "not_sure", "threshold": t,
            "model": state["rice_model"] if model == "best" else "CNN (Class 4)"}


@app.post("/feature_maps")
async def feature_maps(file: UploadFile = File(...)):
    x = read_leaf(await file.read())
    m = state["see"](x)[0].numpy()[::2, ::2, :8]   # first 8 filters, 80 x 80
    m = 255 * m / (m.max(axis=(0, 1)) + 1e-6)                                # each picture scaled to 0-255
    return {"maps": m.astype("uint8").transpose(2, 0, 1).tolist()}
