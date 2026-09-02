import pickle
import math
import torch
import numpy as np

from service_embedding import EmbeddingService
from data.load_data import Dataset
import os
import vec2text
import argparse
from service_dp import DPWrapper
from tqdm import tqdm
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["HF_HUB_ENABLE_XET"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="taylor")
    parser.add_argument("--model_name", type=str, default="gtr-t5-base-no-norm")
    parser.add_argument("--setting", type=str, default="non_dp")
    parser.add_argument("--embedding_model", type=str, default="gtr-t5-base-no-norm")
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--random_projection_K", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    
    args = parser.parse_args()
    
    r = args.r
    random_projection_K = args.random_projection_K
    b = 2 * r / math.sqrt(random_projection_K)
    tau = args.tau
    batch_size = args.batch_size
        
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    model_name = args.model_name
    embedding_model = args.embedding_model
    dataset_name = args.dataset_name
    setting = args.setting
    
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    
    corrector_repeat = 4
    is_recompute = False
    corrector_noise_ratio = 0.01
    
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
    SYNTHETIC_DATA_DIR = os.getenv("SYNTHETIC_DATA_DIR", "./results_synthetic_data")
    bin_histogram_path = f"{RESULTS_DIR}/{dataset_name}_{model_name}_b{round(b, 4)}_{random_projection_K}_{random_seed}.pkl"
    
    if setting == "non_dp":
        SAVE_DIR = f"{SYNTHETIC_DATA_DIR}/{dataset_name}_{model_name}_b{round(b, 4)}_{random_projection_K}_{random_seed}_non_dp_{tau}_synthetic_data.pkl"
    elif setting == "dp":
        SAVE_DIR = f"{SYNTHETIC_DATA_DIR}/{dataset_name}_{model_name}_b{round(b, 4)}_{random_projection_K}_{random_seed}_dp_{args.noise_std}_{args.mask_prob}_{tau}_synthetic_data.pkl"
    else:
        raise ValueError(f"Invalid setting: {setting}")
    
    embedding_service = EmbeddingService(model_name=model_name, batch_size=256)     
    os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)

    if os.path.exists(SAVE_DIR) and is_recompute is False:
        with open(SAVE_DIR, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = {}
    
    with open(bin_histogram_path, "rb") as f:
        res = pickle.load(f)
        bin_averages = res["bin_averages"]
        bin_members = res["bin_members"]

    if setting == "non_dp":
        remaining_keys = [key for key in bin_members.keys() if len(bin_members[key]) >= tau and (key not in all_results or is_recompute is True)]
        centroids = bin_averages
    elif setting == "dp":
        dataset = Dataset(
            data_dir=DATA_DIR, dataset_name=dataset_name, embedding_model=embedding_model
        )
        text_embeddings = dataset.load_embeddings(embedding_service, recompute=False)
        
        dp_graph = DPWrapper(noise_std=args.noise_std, mask_prob=args.mask_prob, original_tau=tau)
        dp_graph.construct(bin_histogram_path, text_embeddings, is_recompute=is_recompute)
        remaining_keys = list(dp_graph.result['dp_bin_averages'].keys())
        centroids = dp_graph.result['dp_bin_averages']
        
    else:
        raise ValueError(f"Invalid setting: {setting}")

    if len(remaining_keys) == 0:
        return

    corrector = vec2text.load_pretrained_corrector("gtr-base")

    all_batches = [remaining_keys[i:i+batch_size] for i in range(0, len(remaining_keys), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(all_batches, desc="Processing batches")):
        batch_centroids = [centroids[k] for k in batch]
        batch_centroids = torch.stack(batch_centroids).cuda()
        
        batch_synthetic_data = vec2text.invert_embeddings(
            embeddings=batch_centroids,
            corrector=corrector,
            num_steps=5,
        )
        noisy_synthetic_data = []
        for repeat_idx in range(corrector_repeat):
            noise = torch.randn(batch_centroids.shape[0], batch_centroids.shape[1]).cuda() * corrector_noise_ratio
            noisy_embeddings = batch_centroids.clone() + noise
            noisy_items = vec2text.invert_embeddings(
                embeddings=noisy_embeddings,
                corrector=corrector,
                num_steps=5,
            )
            noisy_synthetic_data.append(noisy_items)
        batch_all_results = {}
        for idx, k in enumerate(batch):
            batch_all_results[k] = {
                "key": k,
                "centroid": batch_centroids[idx].cpu().tolist(),
                "num_members": len(bin_members[k]),
                "synthetic_data": batch_synthetic_data[idx],
                "noisy_synthetic_data": [noisy_data[idx] for noisy_data in noisy_synthetic_data],
                "original_text": bin_members[k]
            }
        all_results.update(batch_all_results)
                
        with open(SAVE_DIR, "wb") as f:
            pickle.dump(all_results, f)


if __name__ == "__main__":
    main()