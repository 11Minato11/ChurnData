# 🏦 Bank Customer Churn Prediction & Segmentation

Ce projet propose une approche complète de Data Science pour analyser, prédire et segmenter les départs clients (churn) au sein d'une banque.

## 🚀 Fonctionnalités Clés

- **Pipeline Modulaire** : Code structuré en fonctions (`ChurnData.py`) pour une lecture professionnelle.
- **Analyse Exploratoire (EDA)** : Génération automatisée de graphiques de distribution, de boxplots et de corrélations.
- **Modélisation Avancée** : Utilisation de Random Forest avec optimisation du seuil (Threshold tuning) pour maximiser le rappel (Recall).
- **Segmentation Client** : Clustering hiérarchique pour identifier des segments à haut risque.
- **Reporting Automatique** : Génération d'un rapport professionnel format PDF incluant les visualisations.

## 📂 Structure du Projet

```text
├── ChurnData.py            # Pipeline DS principal
├── export_to_pdf.py        # Script de conversion du rapport vers PDF
├── Rapport_Final_Churn.pdf  # 📄 Rapport de synthèse final (PDF)
├── Report_Churn_Analysis.md # Source du rapport en Markdown
├── plots/                  # 📊 Dossier contenant les 11+ graphiques
├── requirements.txt        # Dépendances du projet
└── .gitignore              # Fichiers exclus du versionnement
```

## 🛠️ Installation & Utilisation

1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Exécuter l'analyse complète** :
   ```bash
   python ChurnData.py
   ```

3. **Générer le rapport PDF** :
   ```bash
   python export_to_pdf.py
   ```

## 📊 Aperçu des Résultats

Le projet identifie des segments à haut risque, notamment le **Cluster 2 (Séniors)** avec un taux de churn de **35%**.

![Clusters](plots/cluster_02_visuals.png)

*Pour plus de détails, consultez le [Rapport d'Analyse](Report_Churn_Analysis.md).*

---
*Projet développé dans un cadre d'analyse de données bancaires par Othmane / Data Scientist.*
