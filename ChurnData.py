"""
Bank Customer Churn Prediction & Segmentation
Author: Othmane / Data Scientist
Description: This script performs end-to-end data analysis, prediction, and 
             clustering for a bank customer dataset to reduce churn.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from imblearn.over_sampling import SMOTE

# Configuration
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def load_data(path):
    """Loads and performs initial check on the dataset."""
    print("\n" + "="*50)
    print("🚀 1. LOADING DATASET")
    print("="*50)
    df = pd.read_csv(path)
    print(f"✅ Shape: {df.shape}")
    print("\n📋 Missing Values:")
    print(df.isnull().sum())
    return df

def perform_eda(df):
    """Generates exploratory visualizations and returns summary stats."""
    print("\n" + "="*50)
    print("📊 2. EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # Univariate Analysis
    metrics = ['CreditScore', 'Balance', 'Age', 'Tenure']
    for i, col in enumerate(metrics):
        plt.figure(figsize=(10,6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{PLOTS_DIR}/eda_{i+1:02d}_{col.lower()}.png', bbox_inches='tight')
        plt.close()

    # Bivariate: Num Features vs Exited
    num_cols = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary', 'NumOfProducts']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Numerical Variables vs Churn (Exited)', fontsize=16)
    for i, col in enumerate(num_cols):
        row, c = divmod(i, 3)
        sns.boxplot(x='Exited', y=col, data=df, ax=axes[row][c])
        axes[row][c].set_title(f'{col} vs Exited')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/eda_05_num_vs_churn.png', bbox_inches='tight')
    plt.close()

    # Bivariate: Cat Features vs Exited
    cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Categorical Variables vs Churn (Exited)', fontsize=16)
    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, hue='Exited', data=df, ax=axes[i])
        axes[i].set_title(f'{col} vs Exited')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/eda_06_cat_vs_churn.png', bbox_inches='tight')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{PLOTS_DIR}/eda_07_correlation.png', bbox_inches='tight')
    plt.close()

    print("✅ EDA plots saved in 'plots/' directory.")

def preprocess_data(df):
    """Encodes variables, handles outliers, and splits data."""
    print("\n" + "="*50)
    print("🛠️ 3. PREPROCESSING DATA")
    print("="*50)
    
    # Feature Engineering & Encoding
    df_proc = pd.get_dummies(df, columns=["Geography"], dtype=int)
    df_proc["Gender"] = df_proc["Gender"].map({"Male": 1, "Female": 0})
    
    # Drop irrelevant columns
    df_proc = df_proc.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    
    # Split Data
    X = df_proc.drop("Exited", axis=1)
    y = df_proc["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling with RobustScaler (handles outliers)
    cols_to_scale = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']
    scaler = RobustScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    print(f"✅ Preprocessing complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, X, scaler

def train_model(X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle Random Forest avec gestion du déséquilibre 
    et évalue sa performance via différentes métriques.
    """
    print("\n" + "="*50)
    print("🤖 4. MODELING - RANDOM FOREST")
    print("="*50)
    
    # RandomForestClassifier avec poids de classe équilibrés
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    
    # Evaluation standard (Seuil 0.5)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\n📊 Matrice de Confusion (Seuil standard 0.5):")
    print(confusion_matrix(y_test, y_pred))
    print("\n📝 Rapport de Classification:")
    print(classification_report(y_test, y_pred))
    print(f"🎯 Score ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Optimisation du seuil pour favoriser le Recall (rappel)
    # Dans le churn, il est plus coûteux d'oublier un client qui part que de se tromper sur un client stable.
    threshold = 0.34
    y_pred_th = (y_prob >= threshold).astype(int)
    print(f"\n📊 Matrice de Confusion (Seuil Optimisé {threshold}):")
    print(confusion_matrix(y_test, y_pred_th))
    print("\n📝 Rapport de Classification (Optimisé):")
    print(classification_report(y_test, y_pred_th))
    
    # Importance des variables par permutation (plus robuste qu'au sein de l'arbre)
    print("\n🔍 Importance des variables (Permutation):")
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, scoring="roc_auc", random_state=42)
    perm = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print(perm)
    
    return rf

def perform_segmentation(df, scaler):
    """Performs hierarchical clustering to identify customer segments."""
    print("\n" + "="*50)
    print("👥 5. CUSTOMER SEGMENTATION")
    print("="*50)
    
    # Data Prep for Clustering
    X_cluster = df.copy()
    X_cluster = pd.get_dummies(X_cluster, columns=["Geography"], dtype=int)
    X_cluster["Gender"] = X_cluster["Gender"].map({"Male": 1, "Female": 0})
    X_cluster = X_cluster.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
    
    scale_cols = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']
    X_cluster[scale_cols] = scaler.transform(X_cluster[scale_cols])
    
    # Dendrogram (on sample)
    sample = X_cluster.sample(min(2000, len(X_cluster)), random_state=42)
    linked = linkage(sample, method='ward')
    plt.figure(figsize=(14, 7))
    dendrogram(linked, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig(f'{PLOTS_DIR}/cluster_01_dendrogram.png', bbox_inches='tight')
    plt.close()
    
    # Fit HC
    n_clusters = 4
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['Cluster'] = hc.fit_predict(X_cluster)
    
    # Profiling
    print("\n📈 Segment Characterization (Means):")
    profile = df.groupby('Cluster')[['Age', 'Balance', 'NumOfProducts', 'IsActiveMember', 'Exited']].mean().round(2)
    print(profile.T)
    
    # Cluster Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x='Age', y='Balance', hue='Cluster', data=df, palette='Set2', ax=axes[0])
    axes[0].set_title('Segments by Age & Balance')
    sns.barplot(x='Cluster', y='Exited', hue='Cluster', data=df, palette='Set2', legend=False, ax=axes[1])
    axes[1].set_title('Churn Rate per Segment')
    plt.savefig(f'{PLOTS_DIR}/cluster_02_visuals.png', bbox_inches='tight')
    plt.close()
    
    return df

def main():
    path = r"C:\Users\othma\.cache\kagglehub\datasets\halimedogan\churn-dataset\versions\1\churn2.csv"
    
    # Execute Pipeline
    df = load_data(path)
    perform_eda(df)
    X_train, X_test, y_train, y_test, X_raw, scaler = preprocess_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    df_final = perform_segmentation(df, scaler)
    
    print("\n" + "="*50)
    print("✨ PIPELINE COMPLÈTE & RAPPORT DISPONIBLE")
    print("-" * 50)
    print("👉 Consultez 'Report_Churn_Analysis.md' pour les résultats métier.")
    print("👉 Les graphiques sont dans le dossier '/plots'")
    print("="*50)

if __name__ == "__main__":
    main()