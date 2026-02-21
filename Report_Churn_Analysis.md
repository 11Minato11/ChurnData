# 🏦 Rapport : Analyse du Churn & Stratégie de Rétention

**Date :** 21 Février 2026  
**Sujet :** Stratégie basée sur les données pour réduire le départ des clients (Churn)

---

## 1. Résumé Exécutif
Ce rapport présente les conclusions de notre analyse sur 10 000 clients bancaires. L'objectif était d'identifier les causes de départ et de développer un modèle prédictif. Les facteurs dominants sont l'**Âge**, le **Nombre de Produits**, et la **Localisation (Allemagne)**.

---

## 2. Insights Clés de l'Analyse Exploratoire (EDA)

### 🌍 Zones Géographiques à Risque
L'Allemagne présente un risque majeur avec un taux de churn de **32.4%**, contre environ 16% pour la France et l'Espagne.

![Distribution Géographique](plots/eda_06_cat_vs_churn.png)

### 👥 Facteurs Démographiques
*   **Âge :** C'est le facteur le plus corrélé au churn. Les seniors (50+ ans) partent nettement plus.
*   **Genre :** Les femmes partent plus (25%) que les hommes (16%).

![Analyse Numérique](plots/eda_05_num_vs_churn.png)

### 📊 Corrélations entre variables
La heatmap ci-dessous montre les relations directes entre les variables. On note le lien fort entre l'âge et la décision de quitter la banque.

![Matrice de Corrélation](plots/eda_07_correlation.png)

---

## 3. Résultats du Modèle Prédictif

Nous avons entraîné une **Forêt Aléatoire (Random Forest)** pour prédire les départs.

### Performance du modèle :
*   **ROC-AUC :** 0.8586 (Excellente capacité de distinction).
*   **Optimisation du Seuil :** En fixant le seuil à **0.34**, nous capturons **61% des futurs churners**, permettant des actions préventives efficaces.

### Importance des variables (Permutation Importance) :
L'âge et le nombre de produits arrivent en tête des prédicteurs.

---

## 4. Segmentation Clients (Clustering)

Grâce au clustering hiérarchique, nous avons identifié 4 personas :

| Segment             | Profil                     | Taux de Churn |
| :------------------ | :------------------------- | :------------ |
| **A (Stables)**     | Jeunes, peu de produits    | ~14%          |
| **B (À Risque)**    | Actifs, multi-produits     | ~30%          |
| **C (Prioritaire)** | **Seniors, hauts revenus** | **~35%**      |
| **D (Dormants)**    | Nouveaux ou soldes bas     | ~15%          |

![Visualisation des Clusters](plots/cluster_02_visuals.png)

---

## 5. Recommandations Stratégiques

1.  **Rétention Seniors :** Créer des programmes de fidélité spécifiques pour les 50-65 ans (gestion de patrimoine).
2.  **Audit Multi-Produits :** Analyser pourquoi les clients possédant 3+ produits partent massivement.
3.  **Réactivation :** Cibler les membres "inactifs" avec des offres personnalisées pour augmenter leur engagement.

---
*Rapport généré automatiquement par le pipeline ChurnData.py*
