from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join
import mlflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

mlflow.set_tracking_uri("file:./mlruns")  # Assurez-vous que le serveur MLflow est en cours d'exécution
# Chargement des datasets
df_old = pd.read_csv(join('data','df_old.csv'))
df_new = pd.read_csv(join('data','df_new.csv'))

# Charger le préprocesseur
preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# preprocesser les data
X, y, _ = preprocessing(df_old)
Xn, yn, _ = preprocessing(df_new)

# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y)
Xn_train, Xn_test, yn_train, yn_test = split(Xn, yn)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# # create a new model 
# model = create_nn_model(X_train.shape[1])

# # entraîner le modèle
# model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)
# draw_loss(hist)

# # sauvegarder le modèle
# joblib.dump(model, join('models','model_2024_08.pkl'))

# charger le modèle
model_base = joblib.load(join('models','model_2024_08.pkl'))

with mlflow.start_run(run_name="evaluation_model_base"):
    try:
        #%% predire sur les valeurs de train
        y_pred_train = model_predict(model_base, X_train)
        # mesurer les performances MSE, MAE et R²
        perf_train = evaluate_performance(y_train, y_pred_train)  
        
        mlflow.log_params({
        "dataset": "old",
        "type": "evaluation"
        })
        mlflow.log_metrics({
            "mse_train": perf_train["MSE"],
            "mae_train": perf_train["MAE"],
            "r2_train": perf_train["R2"]
        })
        print_data(perf_train, exp_name="Eval old model train")
        
          #%% predire sur les valeurs de train
        y_pred_test = model_predict(model_base, X_test)
        # mesurer les performances MSE, MAE et R²
        perf_test = evaluate_performance(y_test, y_pred_test)  

        mlflow.log_metrics({
            "mse_test": perf_test["MSE"],
            "mae_test": perf_test["MAE"],
            "r2_test": perf_test["R2"]
        })
        print_data(perf_train, exp_name="Eval old model train")
        print_data(perf_test, exp_name="Eval old model test")   
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
    
    
with mlflow.start_run(run_name="finetune_old_data"):
    try:
        
        model_1 = clone_model(model_base)
        model_1, hist = train_model(
        model_1,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        callbacks=[early_stop]
    )
        #%% predire sur les valeurs de train
        y_pred_train = model_predict(model_base, X_train)
        # mesurer les performances MSE, MAE et R²
        perf_train = evaluate_performance(y_train, y_pred_train)  
        
        #%% predire sur les valeurs de test
        y_pred_test = model_predict(model_base, X_test)
        # mesurer les performances MSE, MAE et R²
        perf_test = evaluate_performance(y_test, y_pred_test)  
        
        
        mlflow.log_params({
        "dataset": "old",
        "type": "finetune"
        })

        mlflow.log_metrics({
            "mse_train": perf_train["MSE"],
            "mae_train": perf_train["MAE"],
            "r2_train": perf_train["R2"]
        })
        
        
        mlflow.log_metrics({
            "mse_test": perf_test["MSE"],
            "mae_test": perf_test["MAE"],
            "r2_test": perf_test["R2"]
        })

        # Loss
        for step, loss in enumerate(hist.history["loss"]):
            mlflow.log_metric("loss", loss, step=step)

        for step, val_loss in enumerate(hist.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=step)

        mlflow.keras.log_model(model_1, "model 1")
        draw_loss(hist, "finetune_old")
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
    
   
with mlflow.start_run(run_name="finetune_model1_new_data"):
    try:
        
        model_1 = clone_model(model_base)
        model_1, hist = train_model(
        model_1,
        Xn_train,
        yn_train,
        X_val=Xn_test,
        y_val=yn_test,
        callbacks=[early_stop]
    )
        #%% predire sur les valeurs de train
        yn_pred_train = model_predict(model_base, Xn_train)
        # mesurer les performances MSE, MAE et R²
        perf_train = evaluate_performance(yn_train, y_pred_train)  
        
        #%% predire sur les valeurs de test
        yn_pred_test = model_predict(model_base, Xn_test)
        # mesurer les performances MSE, MAE et R²
        perf_test = evaluate_performance(yn_test, y_pred_test)  
        
        
        mlflow.log_params({
        "dataset": "new",
        "type": "finetune"
        })

        mlflow.log_metrics({
            "mse_train": perf_train["MSE"],
            "mae_train": perf_train["MAE"],
            "r2_train": perf_train["R2"]
        })
        
        
        mlflow.log_metrics({
            "mse_test": perf_test["MSE"],
            "mae_test": perf_test["MAE"],
            "r2_test": perf_test["R2"]
        })

        # Loss
        for step, loss in enumerate(hist.history["loss"]):
            mlflow.log_metric("loss", loss, step=step)

        for step, val_loss in enumerate(hist.history["val_loss"]):
            mlflow.log_metric("val_loss", val_loss, step=step)

        mlflow.keras.log_model(model_1, "model 1 new data")
        draw_loss(hist, "finetune_old")
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
    