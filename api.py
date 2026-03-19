from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from loguru import logger
from joblib import load
from os.path import join
import pandas as pd
import numpy as np


app = FastAPI()


class FormulairePret(BaseModel):
    age: int
    taille: float
    poids: float
    revenu_estime_mois: float
    sexe: str
    sport_licence: str
    niveau_etude: str
    region: str
    smoker: str
    nationalité_francaise: str
    
logger.add("logs/predire_pret.log", rotation="500 MB", level="INFO")
try:
    preprocessor = joblib.load(join('models','preprocessor.pkl'))
    model = joblib.load(join('models','model_2024_08.pkl'))
    logger.info("Modèle et préprocesseur chargés avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle ou préprocesseur : {e}")

# --------------------------
# Fonction de prédiction
# --------------------------
def model_predict(model, X):
    """Fonction pour faire des prédictions avec le modèle de réseau de neurones."""
    y_pred = model.predict(X).flatten()
    return y_pred

@app.post("/predire_pret/")
async def predire_pret(formulaire: FormulairePret):
    try:
        data = formulaire.model_dump()
        df_new = pd.DataFrame([data])

        # Appliquer le préprocesseur
        X_new_processed = preprocessor.transform(df_new)

        # Faire la prédiction
        montant_pret_pred = model_predict(model, X_new_processed)[0]

        # Logger la prédiction
        logger.info(f"Données reçues : {data} | Prédiction : {montant_pret_pred}")

        # Retourner le résultat
        return {"prediction": float(montant_pret_pred)}

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")
    
@app.get("/health/")
async def health_check():
    health = {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "routes": {
            "predire_pret": "ok" if model is not None and preprocessor is not None else "problème",
            "health": "ok"
        }
    }
    return health
 