"""Nettoyage modulaire des données RH (HR Analytics : Job Change of Data Scientists).
Chaque fonction peut être exécutée et corrigée indépendamment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLONNES_CATEGORIELLES = [
    'gender', 'enrolled_university', 'education_level', 'major_discipline',
    'experience', 'company_size', 'company_type', 'last_new_job'
]
COLONNES_NUMERIQUES = ['experience_num', 'last_new_job_num']
COLONNES_OUTLIERS = ['city_development_index', 'training_hours',
                     'experience_num', 'last_new_job_num']


def charger_donnees(train_path='data/raw/aug_train.csv', test_path='data/raw/aug_test.csv'):
    """Lire les fichiers sources sans les modifier."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def examiner_donnees(df):
    """Retourner dimensions, types et statistiques de valeurs manquantes."""
    manquantes = pd.DataFrame({
        'nombre': df.isna().sum(),
        'pourcentage': (df.isna().mean() * 100).round(2)
    }).sort_values('nombre', ascending=False)
    return {'dimensions': df.shape, 'types': df.dtypes, 'manquantes': manquantes}


def verifier_doublons(df):
    """Compter les doublons complets et les IDs dupliqués."""
    return {'lignes_dupliquees': int(df.duplicated().sum()),
            'ids_dupliques': int(df['enrollee_id'].duplicated().sum())}


def traiter_valeurs_manquantes_categorielles(df):
    """Remplacer les NaN catégoriels par 'Non renseigné' sans toucher à l'original."""
    resultat = df.copy()
    for col in COLONNES_CATEGORIELLES:
        resultat[col] = resultat[col].fillna('Non renseigné')
    return resultat


def convertir_variables(df):
    """Ajouter deux variables numériques; conserver les colonnes d'origine.
    >20 -> 21, <1 -> 0 et >4 -> 5 sont des approximations documentées.
    """
    resultat = df.copy()
    resultat['experience_num'] = pd.to_numeric(
        resultat['experience'].replace({'<1': 0, '>20': 21, 'Non renseigné': np.nan, 'Unknown': np.nan}),
        errors='coerce'
    )
    resultat['last_new_job_num'] = pd.to_numeric(
        resultat['last_new_job'].replace({'never': 0, '>4': 5, 'Non renseigné': np.nan, 'Unknown': np.nan}),
        errors='coerce'
    )
    return resultat


def calculer_medianes(train):
    """Calculer les médianes sur TRAIN seulement (évite la fuite de données)."""
    return {col: train[col].median() for col in COLONNES_NUMERIQUES}


def traiter_valeurs_manquantes_numeriques(df, medianes):
    """Remplir les valeurs manquantes avec les médianes issues de TRAIN."""
    resultat = df.copy()
    for col in COLONNES_NUMERIQUES:
        resultat[col] = resultat[col].fillna(medianes[col])
    return resultat


def detecter_outliers(df):
    """Compter les valeurs extrêmes IQR sans supprimer de lignes."""
    bilan = {}
    for col in COLONNES_OUTLIERS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bilan[col] = int(((df[col] < q1 - 1.5 * iqr) |
                          (df[col] > q3 + 1.5 * iqr)).sum())
    return bilan


def tracer_boxplots(df):
    """Afficher les boxplots, uniquement à la demande."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, COLONNES_OUTLIERS):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    return fig


def verifier_categories(df):
    """Retourner le nombre de catégories et d'espaces de bord inutiles."""
    return pd.DataFrame({
        col: {'categories': df[col].nunique(dropna=False),
              'espaces_inutiles': int((df[col].dropna() !=
                                      df[col].dropna().str.strip()).sum())}
        for col in df.select_dtypes(include='object').columns
    }).T


def verifier_qualite(df):
    """Bilan des contrôles finaux pour un dataset."""
    return {
        'lignes': len(df), 'colonnes': len(df.columns),
        'valeurs_manquantes': int(df.isna().sum().sum()),
        **verifier_doublons(df),
        'training_hours_positives': bool(df['training_hours'].ge(0).all()),
        'city_index_dans_0_1': bool(df['city_development_index'].between(0, 1).all()),
    }


def verifier_coherence_train_test(train, test):
    """Contrôler les colonnes, les IDs et la cible."""
    return {
        'colonnes_identiques_hors_target': set(train.columns) - {'target'} == set(test.columns),
        'ids_communs': len(set(train['enrollee_id']) & set(test['enrollee_id'])),
        'target_binaire': set(train['target'].dropna().unique()).issubset({0, 1}),
    }


def sauvegarder_donnees(train, test, dossier='data/processed'):
    """Sauvegarder seulement quand tous les contrôles sont satisfaisants."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    train.to_csv(dossier / 'aug_train_clean.csv', index=False)
    test.to_csv(dossier / 'aug_test_clean.csv', index=False)
    return dossier / 'aug_train_clean.csv', dossier / 'aug_test_clean.csv'


def nettoyer_datasets(train, test):
    """Pipeline facultatif: enchaîne les fonctions, sans sauvegarder."""
    train_clean = convertir_variables(traiter_valeurs_manquantes_categorielles(train))
    test_clean = convertir_variables(traiter_valeurs_manquantes_categorielles(test))
    medianes = calculer_medianes(train_clean)
    return (traiter_valeurs_manquantes_numeriques(train_clean, medianes),
            traiter_valeurs_manquantes_numeriques(test_clean, medianes))
