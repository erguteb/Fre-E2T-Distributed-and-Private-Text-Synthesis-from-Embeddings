from datasets import load_dataset
from tqdm import tqdm
import pickle, os

# ---- Config ----
CHECKPOINT_EVERY = 5_000   # save every N conversations
# PICKLE_TEXT = "texts.pkl"
# PICKLE_META = "metadata.pkl"
dataset_name = "taylor"
DATA_DIR = os.getenv("DATA_DIR", "./data")
PICKLE_TEXT = f"{DATA_DIR}/{dataset_name}/texts.pkl"
PICKLE_META = f"{DATA_DIR}/{dataset_name}/metadata.pkl"
START_FILE = f"{DATA_DIR}/{dataset_name}/start_idx.txt"
os.makedirs(os.path.dirname(PICKLE_TEXT), exist_ok=True)

# ---- Load dataset once ----
ds = load_dataset("TaylorAI/user_queries_dataset", split="train")

# ---- Resume or init ----
if os.path.exists(START_FILE):
    with open(START_FILE, "r") as f:
        start_idx = int(f.read().strip() or 0)
    with open(PICKLE_META, "rb") as f:
        all_meta_data = pickle.load(f)
    with open(PICKLE_TEXT, "rb") as f:
        all_text = pickle.load(f)
else:
    start_idx = 0
    all_meta_data, all_text = [], []

n_total = len(ds)
pbar = tqdm(range(start_idx, n_total), total=n_total - start_idx, desc="Queries")

def checkpoint(cur_idx):
    """Persist progress."""
    with open(PICKLE_TEXT, "wb") as f:
        pickle.dump(all_text, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PICKLE_META, "wb") as f:
        pickle.dump(all_meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(START_FILE, "w") as f:
        f.write(str(cur_idx))
for cov_idx in pbar:
    row = ds[cov_idx]

    # Fast skip if not English
    if row.get("lang") != "en":
        if (cov_idx + 1) % CHECKPOINT_EVERY == 0:
            checkpoint(cov_idx + 1)
        continue

    # Batch append for efficiency
    all_text.append(row["query"])
    all_meta_data.append({
        "language": row.get("lang"),
    })

    # Periodic checkpoint (and on last item below)
    if (cov_idx + 1) % CHECKPOINT_EVERY == 0:
        checkpoint(cov_idx + 1)

# Final save
checkpoint(n_total)
