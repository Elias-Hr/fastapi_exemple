# %%
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris

# Import pour entraîner un modèle sur le dataset Iris
from sklearn.linear_model import LogisticRegression

# %%
app = FastAPI(title="API de Prédiction ML avec FastAPI")


# Schéma de la requête via Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Schéma de la réponse
class PredictionResponse(BaseModel):
    prediction: str

# %%

# Chemin du fichier modèle
MODEL_PATH = "model.pkl"

# Charger le jeu de données Iris pour récupérer les noms de classes
iris_data = load_iris()
target_names = iris_data.target_names
# %%
# Charger ou entraîner et sauvegarder le modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = LogisticRegression(max_iter=200)
    model.fit(iris_data.data, iris_data.target)
    joblib.dump(model, MODEL_PATH)
    print("Modèle entraîné et sauvegardé dans", MODEL_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict(iris: IrisFeatures):
    """
    Prédit la classe d'Iris à partir de mesures fournies.
    """
    try:
        features = np.array(
            [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
        )
        prediction_class = model.predict(features)[0]
        prediction_label = target_names[prediction_class]
        return PredictionResponse(prediction=prediction_label)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
class ModelInfo(BaseModel):
    model: str

@app.get("/modelinfo", response_model=ModelInfo)
def info():
    """
    Renvoie des informations sur le modèle.
    """
    loaded_model = joblib.load(MODEL_PATH)
    ModelInfo = {"model": loaded_model}
    return ModelInfo

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction ML avec FastAPI"}
