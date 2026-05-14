"""
M2G-Net v2 — Central Configuration
All hyperparameters and column mappings are defined here.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10
DATA_PATH   = "data/raw/synthetic_rider_data.csv"

# ── Column Mapping ─────────────────────────────────────────────────────────────
INDIVIDUAL_VIEW_COLS = {
    "rider_role":   ["rider_type"],
    "rider_traits": ["gender", "age_group"],
}

CONTEXTUAL_VIEW_COLS = {
    "road_context": ["police_presence", "traffic_condition", "num_lanes", "has_signal"],
    "environment":  ["weather", "time_slot", "weekend"],
    "site_obs":     ["intersection_type"],   # observed infra features
}
SITE_ID_COL = "intersection_id"             # for learnable random intercept

ALL_VIEW_NAMES   = ["rider_role", "rider_traits", "road_context", "environment", "site"]
INDIVIDUAL_VIEWS = ["rider_role", "rider_traits"]
CONTEXTUAL_VIEWS = ["road_context", "environment", "site"]

TARGET_COLS = [
    "red_light_running",
    "no_turn_signal",
    "helmet_nonuse",
    "mobile_phone_use",
]
TASK_NAMES = TARGET_COLS
TASK_DISPLAY_NAMES = {
    "red_light_running": "Red-light running",
    "no_turn_signal": "Turn-signal non-use",
    "helmet_nonuse": "Helmet non-use",
    "mobile_phone_use": "Mobile phone visibility/use",
}

# Optional task-observation masks for partially observed labels.
# A mask column should be 1 when the corresponding task label is genuinely
# observable for that row, and 0 when the task is not applicable or missing.
# If these columns are absent in the CSV, the data pipeline defaults to all
# labels observed to preserve backward compatibility with the current
# synthetic proof-of-concept data.
TASK_MASK_COLS = {
    "red_light_running": "red_light_running_observed",
    "no_turn_signal": "no_turn_signal_observed",
    "helmet_nonuse": "helmet_nonuse_observed",
    "mobile_phone_use": "mobile_phone_use_observed",
}

# ── Encoder Dims ───────────────────────────────────────────────────────────────
EMBEDDING_DIM  = 8    # per-category embedding size inside each encoder
HIDDEN_DIM     = 32   # hidden layer size in encoder MLPs
VIEW_DIM       = 16   # d: output dimension of every view encoder
DROPOUT_RATE   = 0.30

# ── Cross-Level Interaction ────────────────────────────────────────────────────
INTERACTION_DIM = 16  # output dim of h_inter

# ── Gated Fusion ──────────────────────────────────────────────────────────────
NUM_GATE_INPUTS       = 6       # 5 views + 1 interaction term
TEMPERATURE_INIT      = 2.0     # annealed → 1.0 over training
TEMPERATURE_FINAL     = 1.0
GATE_PRIOR_WEIGHT     = 0.1     # convex weight for sparsemax uniform prior mixing

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE          = 1e-3
WEIGHT_DECAY           = 1e-4
BATCH_SIZE             = 64
MAX_EPOCHS             = 150
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_PATH        = "checkpoints/best_model.pt"

# ── Loss ──────────────────────────────────────────────────────────────────────
USE_UNCERTAINTY_WEIGHTING = True
USE_POS_WEIGHT             = False
USE_FOCAL_LOSS            = True
FOCAL_ALPHA               = 0.25
FOCAL_GAMMA               = 2.0

# ── Num Tasks / Views ─────────────────────────────────────────────────────────
NUM_TASKS = len(TASK_NAMES)
NUM_VIEWS = len(ALL_VIEW_NAMES)
