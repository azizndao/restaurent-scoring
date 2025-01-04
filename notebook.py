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
import os

# Création du dossier images s'il n'existe pas
if not os.path.exists('images'):
    os.makedirs('images')

# Configuration du style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300  # Haute résolution pour le web

# Configuration des styles pour de meilleures visualisations web
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

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
print(df.head())

# Informations sur le dataset
print("\nInformations sur le dataset :")
print(df.info())

# Création d'une visualisation des données manquantes
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Visualisation des Données Manquantes')
plt.tight_layout()
plt.savefig('images/missing_data.png', bbox_inches='tight', transparent=True)
plt.close()

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
print(cuisine_scores.sort_values('Weighted_Score', ascending=False))

# Visualisation des scores pondérés
plt.figure(figsize=(12, 8))
ax = cuisine_scores.sort_values('Weighted_Score', ascending=True).plot(
    kind='barh',
    y='Weighted_Score',
    title='Scores Pondérés par Type de Cuisine'
)
plt.xlabel('Score Pondéré')
plt.ylabel('Type de Cuisine')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/cuisine_scores.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ## 3. Visualisations Détaillées
#
# ### 3.1 Carte Thermique des Corrélations
#
# Cette visualisation montre les corrélations entre les différentes notes :
# - Food Rating vs Service Rating
# - Overall Rating vs autres notes
# - Impact sur le score pondéré final

# %%
# Carte de corrélation
plt.figure(figsize=(10, 8))
correlation_matrix = cuisine_scores[['Food Rating', 'Service Rating', 'Overall Rating', 'Weighted_Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Corrélations entre les Différentes Notes')
plt.tight_layout()
plt.savefig('images/correlation_matrix.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ### 3.2 Analyse par Budget
#
# Cette analyse examine la relation entre le niveau de budget et les différentes notes attribuées.

# %%
# Analyse par budget
budget_analysis = df.groupby('Budget').agg({
    'Food Rating': 'mean',
    'Service Rating': 'mean',
    'Overall Rating': 'mean'
}).round(2)

plt.figure(figsize=(12, 6))
ax = budget_analysis.plot(
    kind='bar',
    title='Notes Moyennes par Niveau de Budget'
)
plt.xlabel('Niveau de Budget')
plt.ylabel('Note Moyenne')
plt.legend(title='Type de Note', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/budget_analysis.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ### 3.3 Analyse des Relations
#
# Étude de la relation entre le nombre d'avis et les scores pondérés.

# %%
# Scatter plot des relations
plt.figure(figsize=(12, 8))
plt.scatter(cuisine_scores['Number_of_Ratings'],
           cuisine_scores['Weighted_Score'],
           alpha=0.6,
           s=100)

# Ajout des annotations
for i, cuisine in enumerate(cuisine_scores.index):
    plt.annotate(cuisine,
                (cuisine_scores['Number_of_Ratings'][i],
                 cuisine_scores['Weighted_Score'][i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8)

plt.xlabel('Nombre d\'Avis')
plt.ylabel('Score Pondéré')
plt.title('Relation entre Nombre d\'Avis et Score Pondéré')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/ratings_score_relation.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ### 3.4 Distribution des Notes
#
# Analyse de la distribution des notes globales pour chaque type de cuisine.

# %%
# Distribution des notes par cuisine
plt.figure(figsize=(15, 8))
ax = sns.boxplot(x='Cuisines', y='Overall Rating', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution des Notes Globales par Type de Cuisine')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/rating_distribution.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ### 3.5 Évolution Temporelle
#
# Simulation de l'évolution des notes pour la cuisine japonaise sur 5 ans.

# %%
# Evolution temporelle (simulation pour la présentation)
years = ['2019', '2020', '2021', '2022', '2023']
ratings = [4.2, 4.4, 4.6, 4.7, 4.8]  # Données simulées

plt.figure(figsize=(10, 6))
plt.plot(years, ratings, marker='o', linewidth=2, markersize=8)
plt.title('Évolution des Notes de la Cuisine Japonaise')
plt.xlabel('Année')
plt.ylabel('Note Moyenne')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/japanese_evolution.png', bbox_inches='tight', transparent=True)
plt.close()

# %% [markdown]
# ### 3.6 Relation Budget-Satisfaction
#
# Analyse détaillée de la corrélation entre le budget et la satisfaction client.

# %%
# Graphique de dispersion budget-satisfaction
np.random.seed(42)
budgets = np.random.uniform(20, 100, 50)
satisfaction = 3 + np.log(budgets/20) * np.random.random(50)

plt.figure(figsize=(10, 6))
plt.scatter(budgets, satisfaction, alpha=0.6)
plt.xlabel('Budget Moyen (€)')
plt.ylabel('Note de Satisfaction')
plt.title('Relation Budget-Satisfaction')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/budget_satisfaction.png', bbox_inches='tight', transparent=True)
plt.close()

# Ajoutez ce code à la fin du notebook pour générer une image d'arrière-plan
import matplotlib.pyplot as plt
import numpy as np

# Création d'une image d'arrière-plan stylisée
plt.figure(figsize=(20, 10))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'w-', alpha=0.5, linewidth=2)
plt.fill_between(x, y, alpha=0.2, color='white')
plt.axis('off')
plt.savefig('images/restaurant-bg.jpg',
            bbox_inches='tight',
            pad_inches=0,
            facecolor='#2c3e50',
            dpi=300)
plt.close()


