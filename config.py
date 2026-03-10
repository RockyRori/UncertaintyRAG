from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

CORPUS_PATH = DATA_DIR / "corpus.json"
MINI_DATASET_PATH = DATA_DIR / "mini_dataset.json"
UTILITY_DATASET_PATH = DATA_DIR / "utility_dataset.json"

UTILITY_MODEL_PATH = SAVED_MODELS_DIR / "utility_mlp.pt"
TFIDF_VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_DIM = 256