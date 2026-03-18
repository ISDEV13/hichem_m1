import matplotlib.pyplot as plt

def print_data(dico, exp_name="exp 1"):
    """Affiche les métriques MSE, MAE et R² à partir d'un dictionnaire."""
    mse = dico["MSE"]
    mae = dico["MAE"]
    r2 = dico["R2"]
    print(f'{exp_name:=^60}')
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    print("="*60)


def draw_loss(history, title="Courbes de Loss et Val Loss"):
    """
    Affiche les courbes de loss et val_loss de l'historique d'entraînement.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title('Courbes de Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()    