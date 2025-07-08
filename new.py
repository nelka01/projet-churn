import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Churn",
    page_icon="📊",
    layout="wide"
)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modele.joblib')
        return model
    except FileNotFoundError:
        st.error("Le fichier 'modele.joblib' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire que cette application.")
        return None

# Fonction pour faire la prédiction
def predict_churn(model, data):
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame([data])
        
        # Faire la prédiction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, probability
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {str(e)}")
        return None, None

# Interface utilisateur
def main():
    st.title("🔮 Prédiction de Churn Client")
    st.markdown("---")
    
    # Charger le modèle
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar pour les informations
    st.sidebar.header("ℹ️ Informations")
    st.sidebar.info(
        "Cette application utilise un modèle de Machine Learning "
        "pour prédire si un client va résilier (churn) ou non."
    )
    
    # Colonnes pour organiser les inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Informations Générales")
        region = st.selectbox("Région", options=list(range(1, 15)), index=0)
        tenure = st.number_input("Durée (TENURE)", min_value=0, max_value=1000, value=12)
        montant = st.number_input("Montant", min_value=0.0, value=1000.0, step=10.0)
        frequence_rech = st.number_input("Fréquence Recharge", min_value=0.0, value=5.0, step=0.1)
        revenue = st.number_input("Revenue", min_value=0.0, value=500.0, step=10.0)
        arpu_segment = st.number_input("ARPU Segment", min_value=0.0, value=100.0, step=1.0)
    
    with col2:
        st.subheader("📊 Données d'Usage")
        frequence = st.number_input("Fréquence", min_value=0.0, value=10.0, step=0.1)
        data_volume = st.number_input("Volume de Données", min_value=0.0, value=1000.0, step=10.0)
        on_net = st.number_input("On Net", min_value=0.0, value=50.0, step=1.0)
        orange = st.number_input("Orange", min_value=0.0, value=30.0, step=1.0)
        tigo = st.number_input("Tigo", min_value=0.0, value=20.0, step=1.0)
    
    with col3:
        st.subheader("🎯 Autres Métriques")
        mrg = st.selectbox("MRG", options=[0, 1], index=0)
        regularity = st.selectbox("Régularité", options=[0, 1], index=1)
        top_pack = st.selectbox("Top Pack", options=[0, 1], index=0)
        freq_top_pack = st.number_input("Fréquence Top Pack", min_value=0.0, value=2.0, step=0.1)
    
    # Bouton de prédiction
    st.markdown("---")
    if st.button("🚀 Prédire le Churn", type="primary"):
        # Préparer les données
        data = {
            'REGION': region,
            'TENURE': tenure,
            'MONTANT': montant,
            'FREQUENCE_RECH': frequence_rech,
            'REVENUE': revenue,
            'ARPU_SEGMENT': arpu_segment,
            'FREQUENCE': frequence,
            'DATA_VOLUME': data_volume,
            'ON_NET': on_net,
            'ORANGE': orange,
            'TIGO': tigo,
            'MRG': mrg,
            'REGULARITY': regularity,
            'TOP_PACK': top_pack,
            'FREQ_TOP_PACK': freq_top_pack
        }
        
        # Faire la prédiction
        prediction, probability = predict_churn(model, data)
        
        if prediction is not None:
            # Afficher les résultats
            st.markdown("---")
            st.subheader("📈 Résultats de la Prédiction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Le client va probablement résilier (CHURN)")
                else:
                    st.success("✅ Le client va probablement rester")
            
            with col2:
                if probability is not None:
                    churn_prob = probability[1] * 100
                    st.metric("Probabilité de Churn", f"{churn_prob:.1f}%")
                    
                    # Gauge chart pour la probabilité
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probabilité de Churn (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Section d'analyse des données
    st.markdown("---")
    st.subheader("📊 Analyse des Données Saisies")
    
    # Créer un DataFrame avec les données saisies
    current_data = {
        'REGION': region,
        'TENURE': tenure,
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'MRG': mrg,
        'REGULARITY': regularity,
        'TOP_PACK': top_pack,
        'FREQ_TOP_PACK': freq_top_pack
    }
    
    # Afficher un graphique radar des principales métriques
    metrics_normalized = {
        'Revenue': min(revenue / 1000, 1),
        'ARPU': min(arpu_segment / 200, 1),
        'Data Volume': min(data_volume / 2000, 1),
        'Tenure': min(tenure / 100, 1),
        'Frequence': min(frequence / 20, 1)
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics_normalized.values()),
        theta=list(metrics_normalized.keys()),
        fill='toself',
        name='Profil Client'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Profil Client (Valeurs Normalisées)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Application de Prédiction de Churn - Développée avec Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()