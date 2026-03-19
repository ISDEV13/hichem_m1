import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    return TestClient(app)



def test_sante(client):
    '''Test de la route de santé de l'API'''
    response = client.get("/health/")  

    assert response.status_code == 200
    assert "status" in response.json()



def test_prediction_success(client):
    '''Test de la route de prédiction avec des données valides 
    et vérification de la structure de la réponse'''
    payload = {
        "age": 30,
        "taille": 175.0,
        "poids": 70.0,
        "revenu_estime_mois": 2000.0,
        "sexe": "H",
        "sport_licence": "oui",
        "niveau_etude": "bac+3",
        "region": "Ile-de-France",
        "smoker": "non",
        "nationalite_francaise": "oui"
    }

    response = client.post("/predire_pret/", json=payload)
    print(response.json())
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))



def test_prediction_missing_field(client):
    '''Test de la route de prédiction avec des données manquantes et vérification que l'API retourne une erreur'''
    
    payload = {
        "age": 30  
    }

    response = client.post("/predire_pret/", json=payload)

    assert response.status_code == 422  



def test_prediction_invalid_values(client):
    '''Test de la route de prédiction avec des données invalides et vérification que l'API retourne une erreur'''
    payload = {
        "age": -10,
        "taille": -170,
        "poids": -70,
        "revenu_estime_mois": -2000,
        "sexe": "X",
        "sport_licence": "maybe",
        "niveau_etude": "aucun",
        "region": "inconnue",
        "smoker": "???",
        "nationalité_francaise": "non"
    }

    response = client.post("/predire_pret/", json=payload)


    assert response.status_code in [400, 422]


def test_not_found(client):
    '''Test d'une route qui n'existe pas pour vérifier que l'API retourne une erreur 404'''
    response = client.get("/route_qui_n_existe_pas")

    assert response.status_code == 404