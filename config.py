from pathlib import Path

# =========================================================
# Project root
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent

# =========================================================
# Directories
# =========================================================
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"

MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

OUTPUTS_DIR = ROOT_DIR / "outputs"

# =========================================================
# Dataset selection
# =========================================================
DATASET_NAME = "squad"
DATASET_VERSION = "v1.1"   # 当前先用 SQuAD 1.1，后续可切 v2.0

# =========================================================
# Raw dataset paths
# =========================================================
SQUAD_RAW_DIR = RAW_DATA_DIR / "squad"

SQUAD_TRAIN_V11_PATH = SQUAD_RAW_DIR / "train-v1.1.json"
SQUAD_DEV_V11_PATH = SQUAD_RAW_DIR / "dev-v1.1.json"

SQUAD_TRAIN_V20_PATH = SQUAD_RAW_DIR / "train-v2.0.json"
SQUAD_DEV_V20_PATH = SQUAD_RAW_DIR / "dev-v2.0.json"

# =========================================================
# Processed dataset paths
# =========================================================
DATASET_QA_PATH = PROCESSED_DATA_DIR / f"{DATASET_NAME}_{DATASET_VERSION}_qa.json"
DATASET_CORPUS_PATH = PROCESSED_DATA_DIR / f"{DATASET_NAME}_{DATASET_VERSION}_corpus.json"
DATASET_STATS_PATH = PROCESSED_DATA_DIR / f"{DATASET_NAME}_{DATASET_VERSION}_stats.json"
UTILITY_DATASET_PATH = PROCESSED_DATA_DIR / f"{DATASET_NAME}_{DATASET_VERSION}_utility.json"

# =========================================================
# Mini / sampled dataset paths
# =========================================================
MINI_DATASET_PATH = DATA_DIR / "mini_dataset.json"

# =========================================================
# Backward-compatible aliases
# =========================================================
CORPUS_PATH = DATASET_CORPUS_PATH

# =========================================================
# Saved model paths
# =========================================================
UTILITY_MODEL_PATH = SAVED_MODELS_DIR / "utility_mlp.pt"
TFIDF_VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"

# =========================================================
# Global random / training config
# =========================================================
RANDOM_SEED = 42
TEST_SIZE = 0.2

MAX_FEATURES = 5000
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.2
HIDDEN_DIM = 256

# =========================================================
# Retrieval config
# =========================================================
INITIAL_TOP_K = 3
RETRIEVE_MORE_K = 2
MAX_RETRIEVAL_BUDGET = 2
MAX_DECISION_STEPS = 4

# =========================================================
# Decision thresholds
# =========================================================
TAU_ANSWER = 0.75
TAU_RETRIEVE = 0.45
TAU_CONFLICT = 0.35

# =========================================================
# Uncertainty weighting
# =========================================================
UNCERTAINTY_ALPHA = 0.5
UNCERTAINTY_BETA = 0.3
UNCERTAINTY_GAMMA = 0.2

# =========================================================
# Generator config
# =========================================================
GENERATOR_MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 32

# =========================================================
# Utility dataset construction
# =========================================================
UTILITY_INCLUDE_SUPPORT_SCORE = True
UTILITY_POSITIVE_THRESHOLD = 0.5

# =========================================================
# SQuAD preprocessing config
# =========================================================
SQUAD_INCLUDE_TITLE_IN_PASSAGE = True
SQUAD_MAX_CONTEXT_CHARS = None  # 可设为整数做截断，比如 800

# =========================================================
# Default Phase 4A runtime settings
# =========================================================
DEFAULT_BUILD_MINI_SAMPLE_SIZE = 100
DEFAULT_BUILD_MINI_SPLIT = "train"

DEFAULT_UTILITY_MAX_QUESTIONS = 100
DEFAULT_UTILITY_TOP_K = 3
DEFAULT_UTILITY_SAVE_EVERY = 20

# =========================================================
# Threshold search config
# =========================================================
THRESHOLD_SEARCH_START = 0.05
THRESHOLD_SEARCH_END = 0.95
THRESHOLD_SEARCH_STEP = 0.05

# =========================================================
# Create necessary directories automatically
# =========================================================
for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEMO_DATA_DIR,
    MODELS_DIR,
    SAVED_MODELS_DIR,
    OUTPUTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)