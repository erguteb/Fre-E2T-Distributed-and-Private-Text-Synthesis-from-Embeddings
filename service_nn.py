import numpy as np
import torch
from tqdm import tqdm


def binning_histogram(
    X, c, b, tau=0, device="cuda", show_progress=True, raw_matrix=None
):
    N, K = X.shape
    X = X.to(device)

    if raw_matrix is not None:
        assert (
            raw_matrix.shape[0] == N
        ), "raw_matrix must have the same number of samples N"
        raw_matrix = raw_matrix.to(device)
        avg_dim = raw_matrix.shape[1]
    else:
        raw_matrix = X
        avg_dim = K

    X_clipped = torch.clamp(X, -c, c)

    starts = torch.rand(K, device=device) * b - c

    bin_indices = torch.floor((X_clipped - starts.unsqueeze(0)) / b).long()
    bin_indices = torch.clamp(bin_indices, min=0)

    N = bin_indices.shape[0]
    offset1 = torch.tensor(1469598103934665603, dtype=torch.long, device=device)
    prime1 = torch.tensor(1099511628211, dtype=torch.long, device=device)
    offset2 = torch.tensor(7809847782465536322, dtype=torch.long, device=device)
    prime2 = torch.tensor(6364136223846793005, dtype=torch.long, device=device)

    h1 = offset1.repeat(N)
    h2 = offset2.repeat(N)
    for j in range(K):
        col = bin_indices[:, j].to(torch.long)
        h1 = (h1 ^ col) * prime1
        h2 = (h2 ^ col) * prime2

    keys2 = torch.stack([h1, h2], dim=1)

    if show_progress:
        print("Step 4/5: Computing unique bins and aggregating...")
    unique_keys, inverse_indices, counts = torch.unique(
        keys2, dim=0, return_inverse=True, return_counts=True
    )

    bin_sums = torch.zeros(len(unique_keys), avg_dim, device=device)
    bin_sums.index_add_(0, inverse_indices, raw_matrix)

    mask = counts > tau
    filtered_group_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
    filtered_counts = counts[mask]
    filtered_sums = bin_sums[mask]

    filtered_avgs = filtered_sums / filtered_counts.unsqueeze(1)

    if show_progress:
        print(
            f"Step 5/6: Collecting member indices for {len(filtered_group_ids)} bins..."
        )

    sample_indices = torch.arange(N, device=device)

    bin_members_list = {}
    for i in range(len(filtered_group_ids)):
        gid = filtered_group_ids[i]
        mask_members = inverse_indices == gid
        members = sample_indices[mask_members].cpu().tolist()
        bin_members_list[i] = members

    if show_progress:
        print(f"Step 6/6: Converting {len(counts)} unique bins to dictionaries...")

    histogram = {}
    num_groups = counts.shape[0]
    if show_progress:
        iterator = tqdm(range(num_groups), total=num_groups, desc="Building histogram")
    else:
        iterator = range(num_groups)
    for gid in iterator:
        idx = torch.nonzero(inverse_indices == gid, as_tuple=False)[0, 0].item()
        key = tuple(bin_indices[idx].tolist())
        histogram[key] = counts[gid].item()

    bin_averages = {}
    bin_members = {}
    if show_progress and len(filtered_group_ids) > 0:
        iterator = tqdm(
            range(len(filtered_group_ids)),
            total=len(filtered_group_ids),
            desc="Building averages",
        )
    else:
        iterator = range(len(filtered_group_ids))
    for i in iterator:
        gid = filtered_group_ids[i].item()
        idx = torch.nonzero(inverse_indices == gid, as_tuple=False)[0, 0].item()
        key = tuple(bin_indices[idx].tolist())
        bin_averages[key] = filtered_avgs[i].cpu()
        bin_members[key] = bin_members_list[i]

    if show_progress:
        print("Done!")

    return bin_indices, histogram, bin_averages, bin_members
