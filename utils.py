import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

VARIABLE_MAPPING = {
    # Type de travail 
    'private': 'work_type_Private',
    'sw': 'work_type_Self_employed',  
    'gw': 'work_type_Govt_job',
    'cw': 'work_type_children',

    # Résidence
    'ru': 'Residence_Urban', 
    'rr': 'Residence_Rural', 

    # Âge (discrétisé)
    'a1': 'age_<30',
    'a2': 'age_30-60',
    'a3': 'age_>60',

    # Mariage
    'my': 'ever_married_Yes',  
    'mn': 'ever_married_No',  

    # AVC
    's0': 'stroke_0',          
    's1': 'stroke_1',         

    # Hypertension
    'h0': 'hypertension_0',    
    'h1': 'hypertension_1',  

    # Maladie cardiaque
    'hd0': 'heart_disease_0',
    'hd1': 'heart_disease_1',

    # Tabagisme
    'sf': 'smoking_status_formerly_smoked', 
    'sn': 'smoking_status_never_smoked',    
    'ss': 'smoking_status_smokes',         
    'su': 'smoking_status_Unknown',         
    
    # Genre
    'gm': 'gender_Male',
    'gf': 'gender_Female',

    # BMI (discrétisé)
    'b1': 'bmi_<18.5',
    'b2': 'bmi_18.5-25',
    'b3': 'bmi_25-30',
    'b4': 'bmi_>30',
    
    # Glucose (discrétisé)
    'g1': 'glucose_<100',
    'g2': 'glucose_100-150',
    'g3': 'glucose_>150'
}




def load_and_preprocess_data(file_path):
    # 1. Chargement des données
    df = pd.read_csv(file_path)
    
    # 2. Gestion des valeurs manquantes
    print("Gestion des valeurs manquantes...")
    
    # Imputation numérique
    numeric_columns = ['bmi', 'avg_glucose_level', 'age']
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # Imputation catégorielle
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    # 3. Discrétisation des variables continues
    print("Discrétisation des variables continues...")
    
    df['age'] = pd.cut(df['age'], 
                      bins=[0, 30, 60, 100], 
                      labels=['age_<30', 'age_30-60', 'age_>60'])
    
    df['bmi'] = pd.cut(df['bmi'],
                      bins=[0, 18.5, 25, 30, 100],
                      labels=['bmi_<18.5', 'bmi_18.5-25', 'bmi_25-30', 'bmi_>30'])
    
    df['avg_glucose_level'] = pd.cut(df['avg_glucose_level'],
                                   bins=[0, 100, 150, 300],
                                   labels=['glucose_<100', 'glucose_100-150', 'glucose_>150'])
    
    # 4. Conversion des variables binaires en one-hot
    print("Conversion des variables binaires...")
    
    binary_columns = {
        'hypertension': ['hypertension_0', 'hypertension_1'],
        'heart_disease': ['heart_disease_0', 'heart_disease_1'],
        'stroke': ['stroke_0', 'stroke_1']
    }
    
    for col, new_cols in binary_columns.items():
        df[new_cols[0]] = (df[col] == 0).astype(int)
        df[new_cols[1]] = (df[col] == 1).astype(int)
    
    # 5. One-hot encoding pour les autres variables catégorielles
    print("Encodage des variables catégorielles...")
    
    df = pd.get_dummies(df, columns=[
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
        'age',
        'bmi',
        'avg_glucose_level'
    ], prefix_sep='_', drop_first=False)
    
    # 6. Uniformisation des noms de colonnes
    print("Uniformisation des noms de colonnes...")
    
    # Remplacement des espaces par des underscores
    df.columns = [col.replace(" ", "_") for col in df.columns]
    
    # 7. Renommage final selon VARIABLE_MAPPING
    print("Application du mapping...")
    
    # Création du mapping inverse avec noms normalisés
    reverse_mapping = {v.replace(" ", "_"): k for k, v in VARIABLE_MAPPING.items()}
    
    # Sélection uniquement des colonnes qui existent dans le mapping
    cols_to_keep = [col for col in df.columns if col in reverse_mapping]
    df = df[cols_to_keep]
    
    # Renommage final
    df.rename(columns=reverse_mapping, inplace=True)
    
    # 8. Vérification finale
    print("\nRésumé du prétraitement:")
    print(f"Shape final: {df.shape}")
    print(f"Colonnes finales: {list(df.columns)}")
    print(f"Exemple de données:\n{df.iloc[:3, :5]}")
    
    return df




def generate_association_rules(df, min_support=0.01, min_confidence=0.3):
    """
    Génère des règles d'association à partir du DataFrame prétraité
    """
    try:
        # Générer les itemsets fréquents
        frequent_itemsets = apriori(df.astype(bool), 
                                  min_support=min_support,
                                  use_colnames=True)
        
        # Générer les règles
        rules = association_rules(frequent_itemsets, 
                                metric="confidence",
                                min_threshold=min_confidence)
        
        return rules
    
    except Exception as e:
        print(f"Erreur lors de la génération des règles: {str(e)}")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur


def evaluate_rules(rules):
    """
    Évalue les règles d'association en utilisant différentes métriques
    selon les recommandations de l'article sur les mesures d'intérêt.
    """
    # 1. Métriques de base (Support, Confidence)
    rules['support'] = rules['support']
    rules['confidence'] = rules['confidence']
    
    # 2. Métriques d'indépendance et de dépendance
    rules['lift'] = rules['lift']  # Mesure d'indépendance
    rules['conviction'] = rules['conviction']  # Mesure de dépendance
    rules['leverage'] = rules['leverage']  # Mesure de différence
    
    # 3. Métriques supplémentaires recommandées par l'article
    # Jaccard similarity
    rules['jaccard'] = rules['support'] / (rules['antecedent support'] + rules['consequent support'] - rules['support'])
    
    # Certainty factor
    rules['certainty_factor'] = (rules['confidence'] - rules['consequent support']) / (1 - rules['consequent support'])
    
    # Information gain
    rules['information_gain'] = rules['support'] * np.log2(rules['lift'])
    
    # Trier les règles selon différentes métriques
    rules_by_lift = rules.sort_values('lift', ascending=False)
    rules_by_confidence = rules.sort_values('confidence', ascending=False)
    rules_by_conviction = rules.sort_values('conviction', ascending=False)
    rules_by_jaccard = rules.sort_values('jaccard', ascending=False)
    rules_by_certainty = rules.sort_values('certainty_factor', ascending=False)
    rules_by_information = rules.sort_values('information_gain', ascending=False)
    
    return {
        'by_lift': rules_by_lift,
        'by_confidence': rules_by_confidence,
        'by_conviction': rules_by_conviction,
        'by_jaccard': rules_by_jaccard,
        'by_certainty': rules_by_certainty,
        'by_information': rules_by_information
    }






def format_rule(rule):
  
    antecedents = list(rule['antecedents'])
    consequents = list(rule['consequents'])
    
    return f"{antecedents} => {consequents}"








def print_rule_metrics(rule):
    """
    Affiche les métriques d'une règle d'association.
    """
    print(f"Support: {rule['support']:.3f}")
    print(f"Confidence: {rule['confidence']:.3f}")
    print(f"Lift: {rule['lift']:.3f}")
    print(f"Conviction: {rule['conviction']:.3f}")
    print(f"Leverage: {rule['leverage']:.3f}")
    print(f"Jaccard: {rule['jaccard']:.3f}")
    print(f"Certainty Factor: {rule['certainty_factor']:.3f}")
    print(f"Information Gain: {rule['information_gain']:.3f}") 