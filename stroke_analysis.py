import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (load_and_preprocess_data, generate_association_rules,
                  evaluate_rules, format_rule, print_rule_metrics)


def main():
    # Créer le dossier data s'il n'existe pas
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Veuillez placer le fichier 'healthcare-dataset-stroke-data.csv' dans le dossier 'data/'")
        return

    # Chemin vers le fichier de données
    data_path = 'data/healthcare-dataset-stroke-data.csv'
    
    if not os.path.exists(data_path):
        print(f"Le fichier {data_path} n'existe pas. Veuillez le placer dans le dossier 'data/'")
        return

    # Charger et prétraiter les données
    print("Chargement et prétraitement des données...")
    df = load_and_preprocess_data(data_path)
    
    # Définir les seuils avec des valeurs plus appropriées
    min_support = 0.001  # 0.1% des transactions
    min_confidence = 0.3  # 30% de confiance
    
    print(f"\nGénération des règles d'association avec:")
    print(f"Support minimum: {min_support}")
    print(f"Confiance minimum: {min_confidence}")
    
    # Générer les règles d'association
    rules = generate_association_rules(df, min_support, min_confidence)
    
    if rules.empty:
        print("\nAucune règle n'a été générée avec les seuils actuels.")
        print("Essayez de réduire les seuils de support et/ou de confiance.")
        return
    
    # Évaluer les règles
    evaluated_rules = evaluate_rules(rules)
    
    # Afficher les meilleures règles selon différentes métriques
    print("\nMeilleures règles par Lift (Indépendance):")
    for idx, rule in evaluated_rules['by_lift'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    print("\nMeilleures règles par Confiance (Fiabilité):")
    for idx, rule in evaluated_rules['by_confidence'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    print("\nMeilleures règles par Conviction (Dépendance):")
    for idx, rule in evaluated_rules['by_conviction'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    print("\nMeilleures règles par Jaccard (Similarité):")
    for idx, rule in evaluated_rules['by_jaccard'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    print("\nMeilleures règles par Certainty Factor (Certitude):")
    for idx, rule in evaluated_rules['by_certainty'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    print("\nMeilleures règles par Information Gain (Information):")
    for idx, rule in evaluated_rules['by_information'].head(5).iterrows():
        print("\nRègle:", format_rule(rule))
        print_rule_metrics(rule)
    
    # Visualisation des métriques
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='jaccard')
    plt.title('Relation entre Support, Confiance, Lift et Jaccard')
    plt.savefig('rule_metrics_visualization.png')
    plt.close()

if __name__ == "__main__":
    main() 