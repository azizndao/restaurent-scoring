"""
Modèle d'Évaluation des Restaurants
==================================

Ce script analyse les données de notation des restaurants et génère 
des visualisations graphiques pour une meilleure compréhension.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du style des graphiques
plt.style.use('seaborn')
sns.set_palette("husl")

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Charge et prépare les données depuis le fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV des données
        
    Returns:
        pd.DataFrame: DataFrame contenant les données préparées
        
    Raises:
        FileNotFoundError: Si le fichier n'est pas trouvé
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {file_path} n'a pas été trouvé.")

def calculate_cuisine_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les scores pour chaque type de cuisine.
    
    Formule du score pondéré :
    - 40% Note globale
    - 40% Note de la nourriture
    - 20% Note du service
    - Facteur multiplicateur basé sur le nombre d'avis
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        
    Returns:
        pd.DataFrame: DataFrame avec les scores par cuisine
    """
    # Calcul des moyennes par cuisine
    cuisine_scores = df.groupby('Cuisines').agg({
        'Food Rating': 'mean',      # Moyenne des notes de nourriture
        'Service Rating': 'mean',   # Moyenne des notes de service
        'Overall Rating': 'mean'    # Moyenne des notes globales
    }).round(2)
    
    # Ajout du nombre d'avis par cuisine
    cuisine_counts = df['Cuisines'].value_counts()
    cuisine_scores['Number_of_Ratings'] = cuisine_counts
    
    # Calcul du score pondéré
    cuisine_scores['Weighted_Score'] = (
        cuisine_scores['Overall Rating'] * 0.4 +    # 40% note globale
        cuisine_scores['Food Rating'] * 0.4 +       # 40% note nourriture
        cuisine_scores['Service Rating'] * 0.2      # 20% note service
    ) * (1 + np.log1p(cuisine_scores['Number_of_Ratings']) / 10)  # Bonus pour nombre d'avis
    
    return cuisine_scores.round(2)

def calculate_location_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les scores pour chaque localisation.
    
    Utilise la même formule de pondération que pour les cuisines :
    - 40% Note globale
    - 40% Note de la nourriture
    - 20% Note du service
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        
    Returns:
        pd.DataFrame: DataFrame avec les scores par localisation
    """
    # Calcul des moyennes par localisation
    location_scores = df.groupby('Location').agg({
        'Food Rating': 'mean',
        'Service Rating': 'mean',
        'Overall Rating': 'mean'
    }).round(2)
    
    # Ajout du nombre d'avis par localisation
    location_counts = df['Location'].value_counts()
    location_scores['Number_of_Ratings'] = location_counts
    
    # Calcul du score pondéré
    location_scores['Weighted_Score'] = (
        location_scores['Overall Rating'] * 0.4 +
        location_scores['Food Rating'] * 0.4 +
        location_scores['Service Rating'] * 0.2
    ) * (1 + np.log1p(location_scores['Number_of_Ratings']) / 10)
    
    return location_scores.round(2)

def calculate_budget_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les scores en fonction des niveaux de budget.
    
    Calcule pour chaque niveau de budget :
    - Moyenne des notes de nourriture
    - Moyenne des notes de service
    - Moyenne des notes globales
    - Nombre total d'avis
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        
    Returns:
        pd.DataFrame: DataFrame avec l'analyse par budget
    """
    budget_scores = df.groupby('Budget').agg({
        'Food Rating': 'mean',
        'Service Rating': 'mean',
        'Overall Rating': 'mean',
        'User ID': 'count'  # Nombre d'avis par niveau de budget
    }).round(2)
    
    budget_scores.rename(columns={'User ID': 'Number_of_Ratings'}, inplace=True)
    return budget_scores

def create_visualizations(cuisine_scores, location_scores, budget_analysis, output_dir="visualizations/"):
    """
    Crée des visualisations graphiques des analyses.
    
    Args:
        cuisine_scores (pd.DataFrame): Scores par cuisine
        location_scores (pd.DataFrame): Scores par localisation
        budget_analysis (pd.DataFrame): Analyse par budget
        output_dir (str): Dossier de sortie pour les graphiques
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Graphique des scores par cuisine
    plt.figure(figsize=(12, 6))
    cuisine_plot = cuisine_scores.sort_values('Weighted_Score', ascending=True).plot(
        kind='barh',
        y='Weighted_Score',
        title='Scores Pondérés par Type de Cuisine'
    )
    plt.xlabel('Score Pondéré')
    plt.ylabel('Type de Cuisine')
    plt.tight_layout()
    plt.savefig(f'{output_dir}cuisine_scores.png')
    plt.close()

    # 2. Carte thermique des corrélations
    plt.figure(figsize=(10, 8))
    correlation_matrix = cuisine_scores[['Food Rating', 'Service Rating', 'Overall Rating', 'Weighted_Score']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Corrélations entre les Différentes Notes')
    plt.tight_layout()
    plt.savefig(f'{output_dir}correlation_heatmap.png')
    plt.close()

    # 3. Analyse du budget
    plt.figure(figsize=(10, 6))
    budget_analysis[['Food Rating', 'Service Rating', 'Overall Rating']].plot(
        kind='bar',
        title='Notes Moyennes par Niveau de Budget'
    )
    plt.xlabel('Niveau de Budget')
    plt.ylabel('Note Moyenne')
    plt.legend(title='Type de Note')
    plt.tight_layout()
    plt.savefig(f'{output_dir}budget_analysis.png')
    plt.close()

    # 4. Distribution des scores par localisation
    plt.figure(figsize=(12, 6))
    location_scores.sort_values('Weighted_Score', ascending=True).plot(
        kind='barh',
        y='Weighted_Score',
        title='Scores Pondérés par Localisation'
    )
    plt.xlabel('Score Pondéré')
    plt.ylabel('Localisation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}location_scores.png')
    plt.close()

    # 5. Relation entre nombre d'avis et score pondéré
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
    plt.savefig(f'{output_dir}ratings_vs_score.png')
    plt.close()

def main():
    """
    Fonction principale avec ajout des visualisations.
    """
    # Chargement et calcul des scores (code existant)
    print("Chargement des données...")
    df = load_and_prepare_data('Cuisine_rating.csv')
    
    print("Calcul des scores...")
    cuisine_scores = calculate_cuisine_scores(df)
    location_scores = calculate_location_scores(df)
    budget_analysis = calculate_budget_analysis(df)
    
    # Génération des visualisations
    print("Génération des visualisations...")
    create_visualizations(cuisine_scores, location_scores, budget_analysis)
    
    # Sauvegarde des résultats (code existant)
    print("\nSauvegarde des résultats...")
    cuisine_scores.to_csv('cuisine_scores.csv')
    location_scores.to_csv('location_scores.csv')
    budget_analysis.to_csv('budget_analysis.csv')
    print("Analyse terminée avec succès!")

if __name__ == "__main__":
    main() 