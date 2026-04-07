# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Detection Cancer Pulmonaire", layout="wide")
st.title("Detection du Cancer Pulmonaire")
st.markdown("Application d'aide au diagnostic -- Machine Learning + Deep Learning")

@st.cache_resource
def load_models():
    model_tab = joblib.load("model1_tabular.pkl")
    scaler = joblib.load("scaler.pkl")
    model_multi = tf.keras.models.load_model("model2_multimodal.keras")
    return model_tab, scaler, model_multi

model_tab, scaler, model_multi = load_models()

st.header("Donnees cliniques du patient")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 90, 60)
    sexe = st.selectbox("Sexe", ["Femme", "Homme"])
    presence_nodule = st.selectbox("Presence de nodule", ["Non", "Oui"])
    subtilite = st.slider("Subtilite du nodule", 1, 5, 3)
    taille = st.selectbox("Taille du nodule (px)", [0, 1])

with col2:
    x_nodule = st.slider("X nodule (normalise)", 0.0, 1.0, 0.5)
    y_nodule = st.slider("Y nodule (normalise)", 0.0, 1.0, 0.5)
    tabagisme = st.slider("Tabagisme (paquets-annees)", 0.0, 50.0, 20.0)
    toux = st.selectbox("Toux chronique", ["Non", "Oui"])
    dyspnee_val = st.selectbox("Dyspnee", ["Non", "Oui"])

with col3:
    douleur = st.selectbox("Douleur thoracique", ["Non", "Oui"])
    perte_poids = st.selectbox("Perte de poids", ["Non", "Oui"])
    spo2 = st.slider("SpO2 (%)", 85, 100, 95)
    antecedent = st.selectbox("Antecedent familial", ["Non", "Oui"])

st.header("Radio thoracique")
uploaded_file = st.file_uploader("Charger une radio thoracique (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Radio chargee", width=300)

if st.button("Lancer la prediction", type="primary"):
    features = np.array([[age, 1 if sexe=="Homme" else 0,
                          1 if presence_nodule=="Oui" else 0,
                          subtilite, taille, x_nodule, y_nodule,
                          tabagisme, 1 if toux=="Oui" else 0,
                          1 if dyspnee_val=="Oui" else 0,
                          1 if douleur=="Oui" else 0,
                          1 if perte_poids=="Oui" else 0,
                          spo2, 1 if antecedent=="Oui" else 0]])
    features_scaled = scaler.transform(features)

    risk_pred = model_tab.predict(features_scaled)[0]
    risk_proba = model_tab.predict_proba(features_scaled)[0]
    risk_labels = ["Faible", "Intermediaire", "Eleve"]

    st.subheader("Modele 1 : Risque de malignite")
    st.markdown(f"### Risque **{risk_labels[risk_pred]}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("P(Faible)", f"{risk_proba[0]:.1%}")
    c2.metric("P(Intermediaire)", f"{risk_proba[1]:.1%}")
    c3.metric("P(Eleve)", f"{risk_proba[2]:.1%}")

    if uploaded_file:
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        cancer_proba = model_multi.predict([img_array, risk_proba.reshape(1, -1)])[0][0]

        st.subheader("Modele 2 : Cancer pulmonaire")
        if cancer_proba > 0.5:
            st.error(f"Cancer PROBABLE (probabilite : {cancer_proba:.1%})")
        else:
            st.success(f"Cancer NON PROBABLE (probabilite : {cancer_proba:.1%})")
        st.progress(float(cancer_proba))
    else:
        st.warning("Chargez une radio pour obtenir la prediction du Modele 2.")