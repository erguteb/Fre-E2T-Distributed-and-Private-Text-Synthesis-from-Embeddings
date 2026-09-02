import pickle
import numpy as np
from data.load_data import Dataset
from service_embedding import EmbeddingService
import torch

import os


class DPWrapper:
    def __init__(self, noise_std, mask_prob, original_tau):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.original_tau = original_tau

    def construct(self, path, text_embeddings, is_recompute=False):
        dp_path = path.replace(".pkl", f"_dp_{self.noise_std}_{self.mask_prob}.pkl")

        if os.path.exists(dp_path) and not is_recompute:
            with open(dp_path, "rb") as f:
                self.result = pickle.load(f)
        else:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.result = pickle.load(f)
            else:
                raise FileNotFoundError(f"NonDP bin histogram not found: {path}")

            dp_histogram = {}
            dp_bin_averages = {}
            subsampled_avg = {}
            for key in self.result["bin_members"].keys():
                num_parties = len(self.result["bin_members"][key])
                mask = np.random.binomial(1, self.mask_prob, size=num_parties)
                noisy_count = mask.sum()
                if noisy_count > self.original_tau * self.mask_prob:
                    parties_after_mask = np.array(self.result["bin_members"][key])[
                        mask == 1
                    ]

                    subsampled_sum = text_embeddings[parties_after_mask].sum(axis=0)
                    noise = torch.randn(subsampled_sum.shape) * self.noise_std
                    noisy_sum = torch.from_numpy(subsampled_sum).float() + noise
                    dp_bin_averages[key] = noisy_sum / noisy_count
                    dp_histogram[key] = noisy_count
                    subsampled_avg[key] = subsampled_sum / noisy_count
            self.result["dp_histogram"] = dp_histogram
            self.result["dp_bin_averages"] = dp_bin_averages
            self.result["subsampled_avgerages"] = subsampled_avg
            with open(dp_path, "wb") as f:
                pickle.dump(self.result, f)
