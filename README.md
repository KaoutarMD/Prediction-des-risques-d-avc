# Analyse des Règles d'Association pour la Prédiction des Risques d'AVC

Ce projet vise à analyser les facteurs de risque d'AVC en utilisant des règles d'association sur le "Stroke Prediction Dataset".

## Structure du Projet

- `requirements.txt` : Liste des dépendances Python
- `stroke_analysis.py` : Script principal d'analyse
- `utils.py` : Fonctions utilitaires
- `data/` : Dossier contenant les données

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placer le dataset dans le dossier `data/`
2. Exécuter le script principal :
```bash
python stroke_analysis.py
```

## Métriques Utilisées

Les métriques suivantes sont utilisées pour évaluer les règles d'association :
- Support
- Confiance
- Lift
- Conviction
- Leverage

Ces métriques sont choisies selon les recommandations de l'article "On selecting interestingness measures for association rules: user-oriented description and multiple criteria decision aid". 