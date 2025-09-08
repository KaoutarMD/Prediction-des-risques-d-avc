import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (load_and_preprocess_data, generate_association_rules,
                  evaluate_rules, format_rule, print_rule_metrics)

# Dictionnaire de correspondance des variables

VARIABLE_MAPPING = {
    # Type de travail 
    'pw': 'work_type_Private',
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



def format_rule_with_names(rule):
    """Version améliorée qui gère les noms longs"""
    def clean_name(name):
        return name.replace("_", " ").title()
    
    antecedents = [clean_name(VARIABLE_MAPPING.get(var, var)) for var in rule['antecedents']]
    consequents = [clean_name(VARIABLE_MAPPING.get(var, var)) for var in rule['consequents']]
    
    return f"{' + '.join(antecedents)} ⇒ {' + '.join(consequents)}"




st.set_page_config(page_title="Analyse des Règles d'Association - AVC", layout="wide")

st.title("Analyse des Règles d'Association pour les Données d'AVC")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres")
st.sidebar.markdown("""
### Aide sur les paramètres
- **Support Minimum** : Fréquence minimale des règles (0.001 = 0.1% des transactions)
- **Confiance Minimum** : Fiabilité minimale des règles (0.3 = 30% de confiance)
""")

# Ajuster les plages des sliders
min_support = st.sidebar.slider(
    "Support Minimum",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    help="Fréquence minimale des règles (0.001 = 0.1% des transactions)"
)

min_confidence = st.sidebar.slider(
    "Confiance Minimum",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Fiabilité minimale des règles (0.3 = 30% de confiance)"
)

# Afficher la légende des variables dans la barre latérale
st.sidebar.header("Légende des Variables")
for code, name in VARIABLE_MAPPING.items():
    st.sidebar.write(f"{code} : {name}")

# Chargement des données
@st.cache_data
def load_data():
    return load_and_preprocess_data('data/healthcare-dataset-stroke-data.csv')

try:
    df = load_data()
    
    # Afficher les statistiques de base
    st.header("Aperçu des Données")
    st.write(f"Nombre total d'observations : {len(df)}")
    st.write(f"Nombre de variables : {len(df.columns)}")
    
    # Générer les règles
    try:
        rules = generate_association_rules(df, min_support, min_confidence)
        
        if rules.empty:
            st.warning("""
            Aucune règle n'a été générée avec les seuils actuels.
            
            Suggestions :
            1. Réduisez le Support Minimum (actuellement {:.4f})
            2. Réduisez la Confiance Minimum (actuellement {:.2f})
            """.format(min_support, min_confidence))
        else:
            # Évaluer les règles
            evaluated_rules = evaluate_rules(rules)
            
            # Afficher le nombre de règles trouvées
            st.success(f"Nombre de règles trouvées : {len(rules)}")
            
            # Créer des onglets pour différentes métriques
            tabs = st.tabs(["Lift", "Confiance", "Conviction", "Jaccard", "Certainty Factor", "Information Gain"])
            
            metrics = ['by_lift', 'by_confidence', 'by_conviction', 'by_jaccard', 'by_certainty', 'by_information']
            metric_names = ["Lift (Indépendance)", "Confiance (Fiabilité)", "Conviction (Dépendance)", 
                           "Jaccard (Similarité)", "Certainty Factor (Certitude)", "Information Gain (Information)"]
            
            for tab, metric, name in zip(tabs, metrics, metric_names):
                with tab:
                    st.subheader(f"Meilleures règles par {name}")
                    for idx, rule in evaluated_rules[metric].head(5).iterrows():
                        with st.expander(f"Règle {idx+1}: {format_rule_with_names(rule)}"):
                            st.write(f"Support: {rule['support']:.3f}")
                            st.write(f"Confidence: {rule['confidence']:.3f}")
                            st.write(f"Lift: {rule['lift']:.3f}")
                            st.write(f"Conviction: {rule['conviction']:.3f}")
                            st.write(f"Leverage: {rule['leverage']:.3f}")
                            st.write(f"Jaccard: {rule['jaccard']:.3f}")
                            st.write(f"Certainty Factor: {rule['certainty_factor']:.3f}")
                            st.write(f"Information Gain: {rule['information_gain']:.3f}")
            
            # Visualisation
            st.header("Visualisation des Métriques")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='jaccard', ax=ax)
            plt.title('Relation entre Support, Confiance, Lift et Jaccard')
            st.pyplot(fig)
            plt.close()
    
    except Exception as e:
        st.error("""
        Aucune règle n'a été trouvée avec les paramètres actuels.
        
        Suggestions :
        1. Réduisez le Support Minimum (actuellement {:.4f})
        2. Réduisez la Confiance Minimum (actuellement {:.2f})
        
        Erreur technique : {}
        """.format(min_support, min_confidence, str(e)))

except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement des données : {str(e)}")
    st.info("Assurez-vous que le fichier 'healthcare-dataset-stroke-data.csv' est présent dans le dossier 'data/'") 