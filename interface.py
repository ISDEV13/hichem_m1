import streamlit as st
import requests
from loguru import logger


logger.add("logs/streamlit.log", rotation="500 MB", level="INFO")
st.title("💰 Prédiction du montant des prêts")

route = st.radio(
    "Choisissez la prédiction",
    ["Prédire", "Santé"]
)

if route =="Prédire":
    st.header("💸Formulaire de prédiction")
    with st.form("form_pret"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=30)
            taille = st.number_input("Taille (cm)", value=170.0)
            poids = st.number_input("Poids (kg)", value=70.0)

        with col2:
            sexe = st.selectbox("Sexe", ["H", "F"])
            sport_licence = st.selectbox("Licence sportive", ["oui", "non"])
            niveau_etude = st.selectbox("Niveau d'étude", ["bac", "bac+2", "bac+3", "master", "doctorat"])
            region = st.selectbox("Région", ["Normandie", "Occitanie", "Ile-de-France", "Autre"])
            smoker = st.selectbox("Fumeur", ["oui", "non"])
            nationalite = st.selectbox("Nationalité française", ["oui", "non"])
            revenu = st.number_input("Revenu mensuel (€)", value=1500.0)
            
        submit = st.form_submit_button("Prédire")
    
    if submit:
        
        data = {
            "age": age,
            "taille": taille,
            "poids": poids,
            "sexe": sexe,
            "sport_licence": sport_licence,
            "niveau_etude": niveau_etude,
            "region": region,
            "smoker": smoker,
            "nationalite_francaise": nationalite,
            "revenu_estime_mois": revenu
        }

        try:
            response = requests.post(
                "http://localhost:8000/predire_pret/",
                json=data
            )

            if response.status_code == 200:
                result = response.json()

                st.success(f"💸 Montant estimé : {result['prediction']} €")
                logger.info(f"Prédiction réussie : {result}")

            else:
                st.error("Erreur API")
                logger.error(response.text)

        except Exception as e:
            st.error("Impossible de contacter l'API")
            logger.error(str(e))
            
elif route == "Santé":
    st.header("✅ Vérification de la santé de l'API")
    try:
        response = requests.get("http://localhost:8000/health/")
        if response.status_code == 200:
            health = response.json()
            
            # Status général
            if health.get("status") == "ok":
                st.success("L'API est en bonne santé !")
            else:
                st.error("⚠️ L'API signale un problème général.")

            # Modèle chargé
            model_status = health.get("model_loaded")
            st.write(f"🔹 Modèle chargé : {'✅ Oui' if model_status else '❌ Non'}")
            
            # Routes disponibles
            st.write("🔹 Routes principales :")
            for route_name, status in health.get("routes", {}).items():
                st.write(f"- {route_name} : {'✅ OK' if status=='ok' else '❌ Problème'}")

            # Temps de réponse si disponible
            if "response_time_ms" in health:
                st.write(f"⏱ Temps de réponse : {health['response_time_ms']} ms")

            # Logs récents
            if "recent_errors" in health:
                st.write("⚠️ Erreurs récentes :")
                for e in health["recent_errors"]:
                    st.write(f"• {e}")

            logger.info("Vérification de santé réussie")
            
        else:
            st.error("L'API ne répond pas correctement.")
            logger.error(f"Vérification de santé échouée : {response.text}")
    except Exception as e:
        st.error("Impossible de contacter l'API pour la vérification de santé.")
        logger.error(f"Erreur lors de la vérification de santé : {str(e)}")