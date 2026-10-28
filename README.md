# 🚗 Exploration Auto MPG & Prédiction de Consommation

Application interactive d'analyse exploratoire et de prédiction de consommation automobile basée sur le jeu de données historique Auto MPG de Seaborn.

---

## 📋 Fonctionnalités

### 1. Analyse descriptive (sans modèle ML)
- **Filtres interactifs (Barre latérale)** :
  - Filtrage par **Origine** (`Toutes`, `usa`, `japan`, `europe`)
  - Filtrage par **Nombre de cylindres** (`3`, `4`, `5`, `6`, `8`)
  - Filtrage par plage d'**Année du modèle** (`1970` à `1982`)
  - Filtrage par **Efficacité minimale (mpg)**
- **Indicateurs clés** :
  - Nombre d'observations filtrées
  - Efficacité moyenne (`mpg`)
  - Consommation moyenne (`L/100 km`, calculée selon la formule exacte $235.215 / \text{mpg}$)
  - Puissance moyenne (`hp`)
- **Tableau de données filtrées** interactif avec conversions automatiques d'unités (`weight_kg`, `consumption_l_100km`, `year`).
- **Sélection individuelle d'un véhicule** pour inspecter sa fiche technique détaillée.
- **Visualisations graphiques** :
  - Diagramme en barres : *Nombre d'observations par origine*
  - Nuage de points : *Relation entre poids (kg) et consommation/efficacité (mpg)*
  - Graphique linéaire : *Évolution de l'efficacité moyenne par année de modèle*

---

### 2. Modèle de Prédiction ML (Random Forest Regressor)
- **Algorithme** : `RandomForestRegressor` (Scikit-Learn)
- **Pipeline de pré-traitement** :
  - Imputation des valeurs manquantes (`horsepower` via médiane)
  - Encodage des variables catégorielles (`origin` via One-Hot Encoding)
- **Métriques d'évaluation** calculées sur un jeu de test indépendant (20%) :
  - Score $R^2$ (Coefficient de détermination)
  - Erreur Absolue Moyenne (MAE)
  - Erreur Quadratique Moyenne (RMSE)
- **Formulaire de saisie** : Cylindres, cylindrée, puissance, poids (en kg), accélération, année du modèle, origine.
- **Résultat de prédiction** instantané en **mpg** et converti en **L/100 km**.
- **Importance des variables** (Feature Importance) affichée sous forme graphique.

---

## 🛠️ Installation des dépendances

Toutes les dépendances nécessaires sont listées dans [requirements.txt](file:///c:/Users/user/Desktop/dep_projet1/requirements.txt) :

```bash
pip install -r requirements.txt
```

---

## 🚀 Lancement de l'application

Pour lancer l'application Streamlit dans votre navigateur :

```bash
streamlit run app.py
```
