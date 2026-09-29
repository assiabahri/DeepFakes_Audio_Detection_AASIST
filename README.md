# DeepFake Audio Detection - AASIST

## Description

Ce projet a pour objectif de détecter si un fichier audio est authentique
(Bonafide) ou généré/manipulé (Spoof / Deepfake).

Le projet utilise le modèle AASIST (Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks), entraîné sur des données
du dataset ASVspoof 2019 LA.

Une interface Streamlit permet également de tester le modèle sur des
fichiers audio.

---

## Structure du projet

```text
DeepFakes_Audio_Detection_AASIST/
│
├── checkpoints/
│   └── Modèles entraînés et checkpoints
│
├── config/
│   └── Fichiers de configuration du modèle
│
├── docs/
│   └── Documentation du projet
│
├── notebooks/
│   └── Notebooks utilisés pour l'exploration,
│       la préparation des données et les expérimentations
│
├── src/
│   └── models/
│       └── AASIST.py
│           Implémentation du modèle AASIST
│
├── app.py
│   └── Interface Streamlit pour tester le modèle
│
├── style.css
│   └── Styles CSS de l'interface Streamlit
│
├── requirements.txt
│   └── Dépendances Python du projet
│
├── .gitignore
│   └── Fichiers et dossiers exclus de Git
│
└── README.md
    └── Documentation principale du projet
```
