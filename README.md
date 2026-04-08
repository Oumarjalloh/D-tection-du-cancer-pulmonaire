

Description du projet

Ce projet propose un systeme complet de detection du cancer pulmonaire combinant :

- Un **Modele 1 (Machine Learning)** qui predit le risque de malignite d'un nodule pulmonaire (faible / intermediaire / eleve) a partir de donnees cliniques tabulaires.
- Un **Modele 2 (Deep Learning)** qui predit la probabilite de cancer pulmonaire a partir de radiographies thoraciques, enrichi par les probabilites du Modele 1 (fusion multimodale).
- Une **application web Streamlit** permettant a un utilisateur de saisir les donnees d'un patient, charger une radio thoracique et obtenir une prediction.

Structure du projet

projet_cancer/
├── app.py                        # Application Streamlit (interface utilisateur)
├── TP_Cancer_Pulmonaire.ipynb    # Notebook complet (EDA, modeles, analyse)
├── patients_cancer_poumon.csv    # Donnees tabulaires (184 patients)
├── jsrt_subset/                  # Radiographies thoraciques
│   ├── sain/                     # 30 images (patients sains)
│   ├── benin/                    # 54 images (nodules benins)
│   └── malin/                    # 100 images (nodules malins)
├── model1_tabular.pkl            # Modele 1 sauvegarde (scikit-learn)
├── scaler.pkl                    # StandardScaler sauvegarde
├── model2_image.keras            # Modele 2 image seul (CNN)
├── model2_multimodal.keras       # Modele 2 multimodal (CNN + probas tabulaires)
├── requirements.txt              # Dependances Python
└── README.md                     # Ce fichier

## Installation et lancement en local

### Prerequis

- Python 3.11
- pip

### Installation

bash
git clone https://github.com/oumarjalloh/d-tection-du-cancer-pulmonaire.git
cd d-tection-du-cancer-pulmonaire
pip install -r requirements.txt
```

### Entrainer les modeles

Ouvrir et executer le notebook :

```bash
jupyter notebook TP_Cancer_Pulmonaire.ipynb
```

Executer toutes les cellules dans l'ordre. Les modeles seront sauvegardes automatiquement.

### Lancer l'application

```bash
streamlit run app.py
```
## Technologies utilisees

| Outil | Usage |
|---|---|
| Python 3.11 | Langage principal |
| pandas, numpy | Manipulation de donnees |
| scikit-learn | Machine Learning (Random Forest, SVM, Gradient Boosting) |
| TensorFlow / Keras | Deep Learning (MobileNetV2, CNN) |
| matplotlib, seaborn | Visualisations |
| Streamlit | Interface utilisateur |
| Streamlit Cloud | Deploiement |
| joblib | Sauvegarde des modeles ML |
