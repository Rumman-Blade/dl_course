import json
import random
from pathlib import Path

import numpy as np
import requests
import streamlit as st

st.title("My first neural network")
x1 = st.number_input("x1", value=1.0)
x2 = st.number_input("x2", value=2.0)

if st.button("Run forward pass"):
    r = requests.post("http://localhost:8000/forward", json={"x1": x1, "x2": x2})
    st.write(r.json())

# ---------- Chapter 2: credit default form ----------
st.header("Credit default check (Chapter 2)")
CLIENT = {  # the known test client from D2-24
    "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2,
    "AGE": 46, "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
    "PAY_4": -1, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 1540,
    "BILL_AMT2": 26060, "BILL_AMT3": 0, "BILL_AMT4": 2650, "BILL_AMT5": 3320,
    "BILL_AMT6": 2764, "PAY_AMT1": 26060, "PAY_AMT2": 0, "PAY_AMT3": 3472,
    "PAY_AMT4": 2320, "PAY_AMT5": 1764, "PAY_AMT6": 2841}
values = {}
cols = st.columns(3)
for i, (name, default) in enumerate(CLIENT.items()):
    values[name] = cols[i % 3].number_input(name, value=float(default))

if st.button("Predict default"):
    r = requests.post("http://localhost:8000/predict", json=values)
    out = r.json()
    st.metric("P(default)", f"{out['p_default']:.1%}")
    label = "default" if out["predicted_class"] == 1 else "no default"
    st.write("Predicted class:", label)

# ---------- shared by Chapter 3 and 4 ----------
API = "http://localhost:8000"
DATA = Path(__file__).parent.parent / "data"
GAME = DATA / "game_clients.json"   # from Class 3 (D3-10)
s = st.session_state

# ---------- Chapter 3: you are the bank officer (shows only after Class 3 files are uploaded) ----------
if GAME.exists():
    st.header("You are the bank officer (Chapter 3)")
    CLIENTS = json.loads(GAME.read_text())   # D3-10
    SHOW = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
    COST = {("approve", 1): 5, ("reject", 0): 1}   # approve a defaulter = 5, reject a good client = 1

    s.setdefault("n", 0); s.setdefault("lost", {"Class": 0, "Model": 0}); s.setdefault("shown", False)
    t = st.slider("Model threshold", 0.05, 0.95, float(requests.get(f"{API}/health").json()["threshold"]), 0.01)
    c = CLIENTS[s.n % len(CLIENTS)]
    st.write(f"Client {s.n % len(CLIENTS) + 1} of {len(CLIENTS)}")
    st.dataframe([{k: c["features"][k] for k in SHOW}], hide_index=True)
    vote = st.radio("The class decides:", ["approve", "reject"], horizontal=True)
    if st.button("Reveal") and not s.shown:
        out = requests.post(f"{API}/predict", params={"threshold": t}, json=c["features"]).json()
        model = "reject" if out["predicted_class"] == 1 else "approve"
        s.lost["Class"] += COST.get((vote, c["true_label"]), 0); s.lost["Model"] += COST.get((model, c["true_label"]), 0)
        s.shown = True
        st.info(f"This client {'DEFAULTED' if c['true_label'] == 1 else 'paid back'}.  Model: P(default) {out['p_default']:.0%} -> {model}")
    left, right = st.columns(2)
    left.metric("Class lost", s.lost["Class"]); right.metric("Model lost", s.lost["Model"])
    if st.button("Next client"):
        s.n, s.shown = s.n + 1, False
        st.rerun()

# ---------- Chapter 4: rice leaf doctor (shows only after Class 4 files are uploaded) ----------
if (DATA / "samples").exists():
    st.header("Rice leaf doctor (Chapter 4)")
    CLASSES = ["Blast", "Narrow_Brown_Spot", "Normal_Leaf", "Sheath_Blight", "Tungro"]
    SAMPLES = sorted((DATA / "samples").glob("*.jpg"))   # 15 test photos (D4-12)

    def ask(path_or_file, endpoint="predict_image", **params):   # send one photo to the API
        data = path_or_file.read_bytes() if isinstance(path_or_file, Path) else path_or_file.getvalue()
        return requests.post(f"{API}/{endpoint}", params=params, files={"file": ("photo.jpg", data)}).json()

    health = requests.get(f"{API}/health").json()
    st.caption(f"Doctor in charge: {health['rice_model']}")

    game, yours, see = st.tabs(["Human vs CNN", "Your photo", "What the CNN sees"])
    s.setdefault("pic", random.choice(SAMPLES)); s.setdefault("score", {"Class": 0, "CNN": 0}); s.setdefault("done", False)

    with game:
        st.image(str(s.pic), width=320)
        guess = st.radio("The class votes:", CLASSES, horizontal=True)
        two = "CNN" not in health["rice_model"]                # Chapter 5: CNN (Class 4) vs MobileNet (Class 5)
        if st.button("Show the answer") and not s.done:
            truth, cnn = s.pic.name.split("__")[0], ask(s.pic, model="cnn")
            s.score["Class"] += guess == truth; s.score["CNN"] += cnn["top_class"] == truth; s.done = True
            msg = f"Truth: {truth}  |  CNN: {cnn['top_class']} ({cnn['probability']:.0%})"
            if two:
                mob = ask(s.pic)
                s.score["MobileNet"] = s.score.get("MobileNet", 0) + (mob["top_class"] == truth)
                msg += f"  |  MobileNet: {mob['top_class']} ({mob['probability']:.0%})"
            st.success(msg)
        board = st.columns(3 if two else 2)
        for col, who in zip(board, ["Class", "CNN", "MobileNet"]):
            col.metric(who, s.score.get(who, 0))
        if st.button("Next photo"):
            s.pic, s.done = random.choice(SAMPLES), False
            st.rerun()

    with yours:
        photo = st.camera_input("Take a photo") if st.toggle("Use the camera") else st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        t = st.slider("Not-sure threshold", 0.0, 1.0, float(health["rice_threshold"]), 0.01)   # Chapter 5
        if photo:
            out = ask(photo, threshold=t)
            st.image(photo, width=320)
            if out["decision"] == "not_sure":
                st.warning(f"Not sure ({out['top_class']}?, {out['probability']:.0%}). Ask an agriculture officer.")
            else:
                st.success(f"{out['model']}: {out['top_class']} ({out['probability']:.0%})")
            st.bar_chart({"probability": out["probabilities"]}, horizontal=True)

    with see:
        pick = st.selectbox("Photo", SAMPLES, format_func=lambda p: p.name.split("__")[0])
        st.image(str(pick), width=200, caption="What you see")
        st.image([np.array(m, dtype=np.uint8) for m in ask(pick, "feature_maps")["maps"]], width=120,
                 caption=[f"filter {k + 1}" for k in range(8)])
