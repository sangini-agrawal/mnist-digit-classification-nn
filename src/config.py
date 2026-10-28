"""Central configuration: paths and hyper-parameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SAMPLE_DIR = ROOT_DIR / "sample_digits"
LOCAL_DATA = ROOT_DIR / "data" / "mnist.npz"      # optional offline copy

MLP_PATH = MODELS_DIR / "mnist_mlp.keras"
CNN_PATH = MODELS_DIR / "mnist_cnn.keras"
METRICS_PATH = RESULTS_DIR / "metrics.json"

SEED = 42
VALIDATION_SIZE = 6000       # images held out from the 60,000 training images
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Dense neural network (main model, as in the project guide)
MLP_UNITS = (128, 64)
MLP_DROPOUT = 0.2
MLP_EPOCHS = 30
MLP_PATIENCE = 5

# Convolutional neural network (improved model)
CNN_FILTERS = (32, 64)
CNN_DROPOUT = 0.5
CNN_EPOCHS = 15
CNN_PATIENCE = 3

CLASS_NAMES = [str(d) for d in range(10)]
