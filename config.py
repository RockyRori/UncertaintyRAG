from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
OUTPUTS_DIR = ROOT_DIR / "outputs"

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

# -----------------------
# Phase 3: decision-aware retrieval
# -----------------------
INITIAL_TOP_K = 3
RETRIEVE_MORE_K = 2
MAX_RETRIEVAL_BUDGET = 2
MAX_DECISION_STEPS = 4

TAU_ANSWER = 0.75
TAU_RETRIEVE = 0.45
TAU_CONFLICT = 0.35

UNCERTAINTY_ALPHA = 0.5   # retrieval uncertainty
UNCERTAINTY_BETA = 0.3    # conflict uncertainty
UNCERTAINTY_GAMMA = 0.2   # stability uncertainty

# generator
GENERATOR_MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 32