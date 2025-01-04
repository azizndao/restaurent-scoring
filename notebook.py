# %% [markdown]
# # Analyse des Restaurants et Visualisations
#
# Ce notebook analyse les données de notation des restaurants et crée des visualisations pour mieux comprendre les tendances.
#
# L'analyse se concentre sur plusieurs aspects :
# - Les scores par type de cuisine
# - Les corrélations entre différentes notes
# - L'impact du budget sur les notes
# - La relation entre le nombre d'avis et les scores

# %%
# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du style des graphiques
plt.style.use('seaborn')
sns.set_palette("husl")
plt.matplotlib.inline()  # Version corrigée de %matplotlib inline

# %% [markdown]
# ## 1. Chargement et Préparation des Données
#
# Nous utilisons un fichier CSV 'Cuisine_rating.csv' qui contient les données suivantes :
# - Notes de la nourriture (Food Rating)
# - Notes du service (Service Rating)
# - Notes globales (Overall Rating)
# - Type de cuisine (Cuisines)
# - Budget

# %%
# Chargement des données
df = pd.read_csv('Cuisine_rating.csv')

# Affichage des premières lignes
print("Aperçu des données :")
display(df.head())

# Informations sur le dataset
print("\nInformations sur le dataset :")
display(df.info())

# %% [markdown]
# ## 2. Calcul des Scores
#
# La fonction `calculate_cuisine_scores` calcule un score pondéré pour chaque type de cuisine en prenant en compte :
#
# 1. Les différentes notes avec les pondérations suivantes :
#    - Note globale : 40%
#    - Note de la nourriture : 40%
#    - Note du service : 20%
#
# 2. Un facteur de popularité basé sur le nombre d'avis
#    - Utilisation de log(1 + nombre_avis) pour réduire l'impact des valeurs extrêmes
#    - Le facteur de popularité augmente le score de 0% à ~23% pour les restaurants les plus notés

# %%
def calculate_cuisine_scores(df):
    cuisine_scores = df.groupby('Cuisines').agg({
        'Food Rating': 'mean',
        'Service Rating': 'mean',
        'Overall Rating': 'mean'
    }).round(2)

    cuisine_counts = df['Cuisines'].value_counts()
    cuisine_scores['Number_of_Ratings'] = cuisine_counts

    cuisine_scores['Weighted_Score'] = (
        cuisine_scores['Overall Rating'] * 0.4 +
        cuisine_scores['Food Rating'] * 0.4 +
        cuisine_scores['Service Rating'] * 0.2
    ) * (1 + np.log1p(cuisine_scores['Number_of_Ratings']) / 10)

    return cuisine_scores.round(2)

# %% [markdown]
# ### Résultats des Scores
#
# Le tableau ci-dessous présente les scores calculés pour chaque type de cuisine, triés par score pondéré décroissant.
# Les scores prennent en compte à la fois la qualité (notes) et la popularité (nombre d'avis).

# %%
# Calcul des scores
cuisine_scores = calculate_cuisine_scores(df)
display(cuisine_scores.sort_values('Weighted_Score', ascending=False))

# %% [markdown]
# ## 3. Visualisations
#
# ### 3.1 Scores par Type de Cuisine
#
# Le graphique suivant montre les scores pondérés pour chaque type de cuisine.
# Les cuisines sont classées de la meilleure à la moins bien notée.

# %%
plt.figure(figsize=(12, 6))
cuisine_scores.sort_values('Weighted_Score', ascending=True).plot(
    kind='barh',
    y='Weighted_Score',
    title='Scores Pondérés par Type de Cuisine'
)
plt.xlabel('Score Pondéré')
plt.ylabel('Type de Cuisine')
plt.tight_layout()

# %% [markdown]
# ### 3.2 Carte Thermique des Corrélations
#
# Cette visualisation montre les corrélations entre les différentes notes :
# - Food Rating vs Service Rating
# - Overall Rating vs autres notes
# - Impact sur le score pondéré final

# %%
plt.figure(figsize=(10, 8))
correlation_matrix = cuisine_scores[['Food Rating', 'Service Rating', 'Overall Rating', 'Weighted_Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Corrélations entre les Différentes Notes')

# %% [markdown]
# ### 3.3 Analyse par Budget

# %%
budget_analysis = df.groupby('Budget').agg({
    'Food Rating': 'mean',
    'Service Rating': 'mean',
    'Overall Rating': 'mean'
}).round(2)

plt.figure(figsize=(10, 6))
budget_analysis.plot(
    kind='bar',
    title='Notes Moyennes par Niveau de Budget'
)
plt.xlabel('Niveau de Budget')
plt.ylabel('Note Moyenne')
plt.legend(title='Type de Note')
plt.tight_layout()

# %% [markdown]
# ### 3.4 Analyse des Relations

# %%
plt.figure(figsize=(10, 6))
plt.scatter(cuisine_scores['Number_of_Ratings'], cuisine_scores['Weighted_Score'])
plt.xlabel('Nombre d\'Avis')
plt.ylabel('Score Pondéré')
plt.title('Relation entre Nombre d\'Avis et Score Pondéré')

for i, cuisine in enumerate(cuisine_scores.index):
    plt.annotate(cuisine,
                (cuisine_scores['Number_of_Ratings'][i],
                 cuisine_scores['Weighted_Score'][i]))
plt.tight_layout()

# %% [markdown]
# ## 4. Analyses Supplémentaires

# %%
# Distribution des notes par type de cuisine
plt.figure(figsize=(15, 6))
sns.boxplot(x='Cuisines', y='Overall Rating', data=df)
plt.xticks(rotation=45)
plt.title('Distribution des Notes Globales par Type de Cuisine')
plt.tight_layout()


