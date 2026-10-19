# 🧹 Nettoyage des données RH — HR Analytics : Job Change of Data Scientists

Projet de **Data Cleaning** réalisé sur le dataset *HR Analytics: Job Change of Data Scientists*.
L'objectif est de préparer des données propres, cohérentes et prêtes pour la modélisation
(prédire si un candidat cherche à changer d'emploi — colonne `target`).

Le travail a été mené en deux temps :

1. **Exploration et nettoyage pas à pas** dans un notebook Jupyter.
2. **Refactorisation en module Python** (`nettoyage_rh_modulaire.py`) : chaque étape devient
   une fonction réutilisable, testable et corrigeable indépendamment.

---

## 📁 Structure du dépôt

```
rh-data-cleaning/
├── data/
│   ├── raw/                  # aug_train.csv, aug_test.csv (à télécharger, non versionnés)
│   └── processed/            # aug_train_clean.csv, aug_test_clean.csv (générés)
├── notebooks/
│   └── nettoyage_donnees_rh.ipynb   # Notebook d'exploration et de nettoyage
├── src/
│   └── nettoyage_rh_modulaire.py    # Fonctions de nettoyage modulaires
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Le dataset

| Fichier | Lignes | Colonnes | Remarque |
|---|---|---|---|
| `aug_train.csv` | 19 158 | 14 | contient la cible `target` |
| `aug_test.csv` | 2 129 | 13 | sans `target` |

Source : [Kaggle — HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).
Placez les deux fichiers CSV dans `data/raw/`.

**Variables principales :**

| Colonne | Type | Description |
|---|---|---|
| `enrollee_id` | int | Identifiant unique du candidat |
| `city` | catégorielle | Code de la ville |
| `city_development_index` | float [0–1] | Indice de développement de la ville |
| `gender` | catégorielle | Genre |
| `relevent_experience` | catégorielle | Expérience pertinente ou non |
| `enrolled_university` | catégorielle | Type d'inscription universitaire |
| `education_level` | catégorielle | Niveau d'études |
| `major_discipline` | catégorielle | Discipline principale |
| `experience` | catégorielle ordonnée | Années d'expérience (`<1`, `1`…`20`, `>20`) |
| `company_size` | catégorielle | Taille de l'entreprise actuelle |
| `company_type` | catégorielle | Type d'entreprise |
| `last_new_job` | catégorielle ordonnée | Années depuis le dernier changement (`never`, `1`…`4`, `>4`) |
| `training_hours` | int | Heures de formation suivies |
| `target` | binaire | 1 = cherche à changer d'emploi, 0 = non |

**Déséquilibre de la cible :** 14 381 « 0 » contre 4 777 « 1 » (≈ 25 % de positifs) — à prendre
en compte lors de la modélisation.

---

## 🔎 Étapes du nettoyage

### Étape 1 — Chargement des données
Lecture de `aug_train.csv` et `aug_test.csv` **sans les modifier** : toutes les transformations
se font sur des copies (`df.copy()`), ce qui permet de comparer avant/après à tout moment.

```python
train, test = nettoyage.charger_donnees()
# TRAIN : (19158, 14) — TEST : (2129, 13)
```

### Étape 2 — Exploration initiale
`info()`, `describe(include="all")` et calcul du nombre / pourcentage de valeurs manquantes
par colonne (`examiner_donnees`).

| Colonne | Valeurs manquantes | % |
|---|---|---|
| `company_type` | 6 140 | 32,05 % |
| `company_size` | 5 938 | 30,99 % |
| `gender` | 4 508 | 23,53 % |
| `major_discipline` | 2 813 | 14,68 % |
| `education_level` | 460 | 2,40 % |
| `last_new_job` | 423 | 2,21 % |
| `enrolled_university` | 386 | 2,01 % |
| `experience` | 65 | 0,34 % |
| autres colonnes | 0 | 0 % |

**Constat :** seules les colonnes catégorielles ont des valeurs manquantes ; les colonnes
numériques et la cible sont complètes.

### Étape 3 — Vérification des doublons
`verifier_doublons` compte les lignes entièrement dupliquées et les `enrollee_id` répétés.

> Résultat : **0 doublon** et **0 ID dupliqué** dans TRAIN comme dans TEST.

### Étape 4 — Traitement des valeurs manquantes catégorielles
Les `NaN` des 8 colonnes catégorielles sont remplacés par une catégorie explicite
**`"Non renseigné"`** (`traiter_valeurs_manquantes_categorielles`).

**Pourquoi ne pas supprimer ni imputer par le mode ?**
- Supprimer les lignes ferait perdre jusqu'à ~32 % des données.
- Imputer par la valeur la plus fréquente fausserait la distribution (ex. sur-représentation de `Male`).
- L'absence d'information peut elle-même être un signal prédictif (ex. un candidat qui ne
  renseigne pas son entreprise est peut-être sans emploi).

> Évolution : la première version du notebook utilisait `"Unknown"` ; la version modulaire
> utilise `"Non renseigné"`. Les comptages sont identiques (ex. `gender` : 4 508), seul le libellé change.

### Étape 5 — Conversion des variables ordinales en numérique
Deux nouvelles colonnes sont **ajoutées** (les colonnes d'origine sont conservées) via
`convertir_variables` :

| Valeur d'origine | `experience_num` | | Valeur d'origine | `last_new_job_num` |
|---|---|---|---|---|
| `<1` | 0 | | `never` | 0 |
| `1` … `20` | 1 … 20 | | `1` … `4` | 1 … 4 |
| `>20` | 21 | | `>4` | 5 |
| `Non renseigné` | NaN | | `Non renseigné` | NaN |

Les valeurs `21` et `5` sont des **approximations documentées** des bornes ouvertes.
Après conversion : 65 NaN dans `experience_num` et 423 dans `last_new_job_num` (TRAIN),
5 et 40 (TEST).

### Étape 6 — Imputation numérique par la médiane (sans fuite de données)
Les médianes sont calculées **uniquement sur TRAIN** (`calculer_medianes`) puis appliquées à
TRAIN et TEST (`traiter_valeurs_manquantes_numeriques`).

> Calculer la médiane sur TEST reviendrait à utiliser de l'information du jeu de test pendant
> la préparation : c'est une **fuite de données** (*data leakage*).

Médianes obtenues : `experience_num` = 9, `last_new_job_num` = 1.
Résultat : **0 valeur manquante** dans les deux jeux.

### Étape 7 — Détection des valeurs aberrantes (méthode IQR)
Boxplots (`tracer_boxplots`) et comptage par la règle de l'IQR
(`< Q1 − 1,5·IQR` ou `> Q3 + 1,5·IQR`) via `detecter_outliers` :

| Colonne | Min | Médiane | Max | Outliers IQR |
|---|---|---|---|---|
| `city_development_index` | 0,448 | 0,903 | 0,949 | 17 |
| `training_hours` | 1 | 47 | 336 | 984 |
| `experience_num` | 0 | 9 | 21 | 0 |
| `last_new_job_num` | 0 | 1 | 5 | 0 |

**Décision : aucune ligne supprimée.** Ces valeurs sont extrêmes mais **plausibles**
(336 h de formation est possible ; un indice de 0,448 reste dans [0, 1]). Les supprimer
retirerait de l'information réelle. Un modèle robuste (arbres) ou une transformation
(log, standardisation) pourra les gérer lors de la modélisation.

### Étape 8 — Vérification des catégories
`verifier_categories` compte les modalités et les espaces superflus en début/fin de chaîne.

> Résultat : **0 valeur avec espaces inutiles**. 123 villes dans TRAIN, 108 dans TEST ;
> les autres colonnes ont le même nombre de modalités dans les deux jeux.

### Étape 9 — Contrôles qualité finaux
`verifier_qualite` produit un bilan par jeu de données :

| Contrôle | TRAIN | TEST |
|---|---|---|
| Lignes (conservées) | 19 158 / 19 158 | 2 129 / 2 129 |
| Colonnes | 16 | 15 |
| Valeurs manquantes | 0 | 0 |
| Lignes / IDs dupliqués | 0 / 0 | 0 / 0 |
| `training_hours` ≥ 0 | ✅ | ✅ |
| `city_development_index` ∈ [0, 1] | ✅ | ✅ |

Contrôles supplémentaires dans le notebook : la colonne `target` et les `enrollee_id` sont
strictement inchangés après nettoyage.

### Étape 10 — Cohérence TRAIN / TEST
`verifier_coherence_train_test` :

- Colonnes identiques (hors `target`) : ✅
- IDs communs entre TRAIN et TEST : **0** (pas de chevauchement)
- `target` strictement binaire {0, 1} : ✅

### Étape 11 — Sauvegarde
`sauvegarder_donnees` écrit `aug_train_clean.csv` et `aug_test_clean.csv` dans
`data/processed/` — **uniquement après** que tous les contrôles soient validés.

### Étape 12 — Refactorisation modulaire
Le code du notebook a été transformé en fonctions dans `src/nettoyage_rh_modulaire.py`,
puis rejoué dans le notebook via `import nettoyage_rh_modulaire as nettoyage` et
`importlib.reload(...)` pour valider chaque fonction séparément. Les résultats sont
identiques à ceux de la version pas à pas.

---

## 🧩 Fonctions du module

| Fonction | Rôle |
|---|---|
| `charger_donnees(train_path, test_path)` | Lit les CSV sources |
| `examiner_donnees(df)` | Dimensions, types, tableau des valeurs manquantes |
| `verifier_doublons(df)` | Lignes et IDs dupliqués |
| `traiter_valeurs_manquantes_categorielles(df)` | `NaN` → `"Non renseigné"` |
| `convertir_variables(df)` | Crée `experience_num` et `last_new_job_num` |
| `calculer_medianes(train)` | Médianes calculées sur TRAIN uniquement |
| `traiter_valeurs_manquantes_numeriques(df, medianes)` | Imputation par médiane |
| `detecter_outliers(df)` | Comptage IQR (sans suppression) |
| `tracer_boxplots(df)` | Boxplots des variables numériques |
| `verifier_categories(df)` | Modalités et espaces superflus |
| `verifier_qualite(df)` | Bilan qualité final |
| `verifier_coherence_train_test(train, test)` | Colonnes, IDs, cible |
| `sauvegarder_donnees(train, test, dossier)` | Export des CSV nettoyés |
| `nettoyer_datasets(train, test)` | Pipeline complet (sans sauvegarde) |

---

## 🚀 Utilisation

```bash
git clone https://github.com/<votre-utilisateur>/rh-data-cleaning.git
cd rh-data-cleaning
pip install -r requirements.txt
# placer aug_train.csv et aug_test.csv dans data/raw/
```

```python
import sys; sys.path.append("src")
import nettoyage_rh_modulaire as nettoyage

train, test = nettoyage.charger_donnees()
train_clean, test_clean = nettoyage.nettoyer_datasets(train, test)

print(nettoyage.verifier_qualite(train_clean))
print(nettoyage.verifier_coherence_train_test(train_clean, test_clean))

nettoyage.sauvegarder_donnees(train_clean, test_clean)
```

---

## 🛣️ Pistes pour la suite
- Encodage des variables catégorielles (One-Hot, ordinal pour `education_level` / `company_size`).
- Regroupement des villes rares (`city` a 123 modalités).
- Gestion du déséquilibre de la cible (pondération des classes, SMOTE).
- Modélisation (Régression logistique, Random Forest, XGBoost) et évaluation (F1, ROC-AUC).

---

## 👩‍💻 Auteure
**Ibtissem Ayadi** — Étudiante ingénieure en IA & Data Engineering.
**Hiba belhaj ** — Étudiante ingénieure en IA & Data Engineering.


