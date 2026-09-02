from service_embedding import EmbeddingService
from data.load_data import Dataset
import numpy as np
import torch
from service_nn import binning_histogram
import os
import math
import pickle


from pathlib import Path


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="om")
    parser.add_argument("--model_name", type=str, default="gtr-t5-base-no-norm")
    parser.add_argument("--embedding_model", type=str, default="gtr-t5-base-no-norm")
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--random_projection_K", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--b", type=float, default=None)
    parser.add_argument("--c", type=float, default=None,
                        help="clipping range for the projected space; defaults to the max absolute projected value")
    args = parser.parse_args()
    c = args.c

    model_name = args.model_name
    embedding_model = args.embedding_model
    dataset_name = args.dataset_name
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

    embedding_service = EmbeddingService(model_name=model_name, batch_size=256)
    random_seed = args.random_seed
    tau = args.tau
    r = args.r
    random_projection_K = args.random_projection_K
    b = args.b
    if b is None:
        b = 2 * r / math.sqrt(random_projection_K)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    save_path = f"{RESULTS_DIR}/{dataset_name}_{model_name}_b{round(b, 4)}_{random_projection_K}_{random_seed}.pkl"
    if os.path.exists(save_path) is False:
        dataset = Dataset(
            data_dir=DATA_DIR,
            dataset_name=dataset_name,
            embedding_model=embedding_model,
        )
        embedding = dataset.load_embeddings(embedding_service, recompute=True)

        embedding = torch.from_numpy(embedding).to(device="cuda", dtype=torch.float32)
        random_projection_matrix = torch.randn(
            embedding.shape[1], random_projection_K
        ).to(device="cuda", dtype=torch.float32)
        embeddings_projected = (embedding @ random_projection_matrix) / math.sqrt(
            random_projection_K
        )
        if c is None:
            c = max(-embeddings_projected.min(), embeddings_projected.max())
        bin_indices, histogram, bin_averages, bin_members = binning_histogram(
            embeddings_projected, c=c, b=b, tau=tau, raw_matrix=embedding
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "bin_indices": bin_indices,
                    "histogram": histogram,
                    "bin_averages": bin_averages,
                    "bin_members": bin_members,
                },
                f,
            )

    else:
        print(f"skip {dataset_name} because the results are already computed")


if __name__ == "__main__":
    main()
