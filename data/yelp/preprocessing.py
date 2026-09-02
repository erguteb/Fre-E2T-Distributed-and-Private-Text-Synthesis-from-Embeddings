from datasets import load_dataset
from tqdm import tqdm
import pickle, os

# ---- Config ----
CHECKPOINT_EVERY = 5_000   # save every N reviews
dataset_name = "yelp"
DATA_DIR = os.getenv("DATA_DIR", "./data")
PICKLE_TEXT = f"{DATA_DIR}/{dataset_name}/texts.pkl"
PICKLE_META = f"{DATA_DIR}/{dataset_name}/metadata.pkl"
START_FILE = f"{DATA_DIR}/{dataset_name}/start_idx.txt"
os.makedirs(os.path.dirname(PICKLE_TEXT), exist_ok=True)


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



def checkpoint(cur_idx):
    """Persist progress."""
    with open(PICKLE_TEXT, "wb") as f:
        pickle.dump(all_text, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PICKLE_META, "wb") as f:
        pickle.dump(all_meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(START_FILE, "w") as f:
        f.write(str(cur_idx))

# ---- Load dataset once ----
# Only the train split (650k reviews) is used; the test split is held out.
ds = load_dataset("Yelp/yelp_review_full", split="train")

n_total = len(ds)
pbar = tqdm(range(start_idx, n_total), total=n_total - start_idx, desc="Reviews")

for cov_idx in pbar:
    row = ds[cov_idx]
    all_text.append(row["text"])
    all_meta_data.append({
        "label": row.get("label")
    })

    # Periodic checkpoint (and on last item below)
    if (cov_idx + 1) % CHECKPOINT_EVERY == 0:
        checkpoint(cov_idx + 1)

# Final save
checkpoint(n_total)
print("saved to", PICKLE_TEXT, "and", PICKLE_META)
print(len(all_text), len(all_meta_data))
