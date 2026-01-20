from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .config import ModelConfig

def create_svm_model(config: ModelConfig):
    """Returns an SVM Classifier with predefined hyperparameters."""
    model = SVC(
        kernel=config.kernel,
        C=config.C,
        random_state=config.random_state
    )
    return model

def create_scaler():
    """Returns a StandardScaler for feature normalization."""
    return StandardScaler()
