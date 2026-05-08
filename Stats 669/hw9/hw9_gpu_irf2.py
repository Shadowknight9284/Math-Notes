import os
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

import torch
import torch.multiprocessing as mp


def create_M(grid_size: int = 25, device=None, dtype=torch.float64):
    xs, ys = torch.meshgrid(
        torch.arange(1, grid_size + 1, device=device, dtype=dtype),
        torch.arange(1, grid_size + 1, device=device, dtype=dtype),
        indexing='xy'
    )
    x = xs.reshape(-1)
    y = ys.reshape(-1)
    M = torch.stack([torch.ones_like(x), x, y, x * x, y * y, x * y], dim=1)
    return M


def create_D(grid_size: int = 25, device=None, dtype=torch.float64):
    xs, ys = torch.meshgrid(
        torch.arange(1, grid_size + 1, device=device, dtype=dtype),
        torch.arange(1, grid_size + 1, device=device, dtype=dtype),
        indexing='xy'
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    diff = coords[:, None, :] - coords[None, :, :]
    D = torch.sqrt((diff * diff).sum(dim=2))
    return D


def restricted_log_lik_gpu(kappa: torch.Tensor, z_vec: torch.Tensor, D: torch.Tensor, M: torch.Tensor):
    n = z_vec.numel()
    p = M.shape[1]

    kv = float(kappa.item())
    if 0 < kv < 2:
        Omega = -(D ** kv)
    elif 2 <= kv < 4:
        Omega = +(D ** kv)
    else:
        Omega = -(D ** kv)

    Omega = Omega.clone()
    idx = torch.arange(Omega.shape[0], device=Omega.device)
    Omega[idx, idx] += 1e-6

    try:
        L = torch.linalg.cholesky(Omega)
        Omega_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        return None

    MO = M.transpose(0, 1) @ Omega_inv
    MOM = MO @ M

    try:
        Lm = torch.linalg.cholesky(MOM)
        MOM_inv = torch.cholesky_inverse(Lm)
    except RuntimeError:
        return None

    beta_hat = MOM_inv @ MO @ z_vec
    residuals = z_vec - (M @ beta_hat)

    QF = (residuals.transpose(0, 1) @ Omega_inv @ residuals).squeeze()
    theta = QF / (n - p)
    if not torch.isfinite(theta) or theta <= 0:
        return None

    sign_o, logdet_o = torch.linalg.slogdet(Omega)
    sign_m, logdet_m = torch.linalg.slogdet(MOM)
    if sign_o <= 0 or sign_m <= 0:
        return None

    treml = 0.5 * logdet_o + 0.5 * logdet_m + ((n - p) / 2.0) * torch.log(QF)
    return treml


def optimize_block(z_vec_np: np.ndarray, D: torch.Tensor, M: torch.Tensor, lower=0.01, upper=5.99, steps=80):
    device = D.device
    z_vec = torch.tensor(z_vec_np.reshape(-1, 1), device=device, dtype=torch.float64)

    grid = torch.linspace(lower, upper, steps=steps, device=device, dtype=torch.float64)
    best_val = None
    best_kappa = None

    for kappa in grid:
        val = restricted_log_lik_gpu(kappa, z_vec, D, M)
        if val is None:
            continue
        if (best_val is None) or (val < best_val):
            best_val = val
            best_kappa = kappa

    if best_val is None:
        return np.nan, np.nan, False
    return round(float(best_kappa.item()), 2), float(best_val.item()), True


def read_center_aggregated_crop(tif_file: str, target_size: int = 2000):
    with rasterio.open(tif_file) as src:
        nr, nc = src.height, src.width
        row_factor = nr // target_size
        col_factor = nc // target_size
        fact = min(row_factor, col_factor)
        if fact < 1:
            fact = 1

        out_h = math.ceil(nr / fact)
        out_w = math.ceil(nc / fact)

        arr = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.average
        )

    nr_a, nc_a = arr.shape
    target = min(target_size, nr_a, nc_a)
    half = target // 2
    center_row = nr_a // 2
    center_col = nc_a // 2

    row_start = center_row - (half - 1)
    row_end = center_row + half
    col_start = center_col - (half - 1)
    col_end = center_col + half

    Z = arr[row_start - 1:row_end - 1 + 1, col_start - 1:col_end - 1 + 1]
    Z = Z[:target, :target]
    return Z


def build_blocks(Z: np.ndarray, block_size=50, thin_step=2):
    n_row_blocks = Z.shape[0] // block_size
    n_col_blocks = Z.shape[1] // block_size
    Z = Z[:n_row_blocks * block_size, :n_col_blocks * block_size]

    blocks = []
    meta = []
    for r in range(n_row_blocks):
        for c in range(n_col_blocks):
            rs = r * block_size
            re = (r + 1) * block_size
            cs = c * block_size
            ce = (c + 1) * block_size
            sub = Z[rs:re, cs:ce]
            sub = sub[::thin_step, ::thin_step]
            blocks.append(sub.reshape(-1).astype(np.float64))
            meta.append((r + 1, c + 1))
    return blocks, meta, n_row_blocks, n_col_blocks


def worker(rank: int, world_size: int, tif_file: str, out_csv: str, target_size: int, max_blocks: int | None):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    Z = read_center_aggregated_crop(tif_file, target_size=target_size)
    blocks, meta, nrb, ncb = build_blocks(Z)

    if max_blocks is not None:
        blocks = blocks[:max_blocks]
        meta = meta[:max_blocks]

    my_idx = list(range(rank, len(blocks), world_size))
    my_blocks = [blocks[i] for i in my_idx]
    my_meta = [meta[i] for i in my_idx]

    D = create_D(grid_size=25, device=device)
    M = create_M(grid_size=25, device=device)

    rows = []
    for (rb, cb), z in zip(my_meta, my_blocks):
        kappa_hat, reml_loglik, converged = optimize_block(z, D, M)
        rows.append({
            'row_block': rb,
            'col_block': cb,
            'kappa_hat': kappa_hat,
            'reml_loglik': reml_loglik,
            'converged': converged
        })

    part_path = out_csv.replace('.csv', f'.part{rank}.csv')
    pd.DataFrame(rows).to_csv(part_path, index=False)


def combine_parts(out_csv: str, world_size: int):
    dfs = []
    for rank in range(world_size):
        part = out_csv.replace('.csv', f'.part{rank}.csv')
        if os.path.exists(part):
            dfs.append(pd.read_csv(part))
    if not dfs:
        raise RuntimeError('No output parts found.')
    df = pd.concat(dfs, ignore_index=True).sort_values(['row_block', 'col_block'])
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif_file', type=str, default='data/USGS_13_n37w118_20260112.tif')
    parser.add_argument('--out_csv', type=str, default='output/hw9_reml_powerlaw_irf2_results_gpu.csv')
    parser.add_argument('--target_size', type=int, default=2000)
    parser.add_argument('--max_blocks', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. This script requires a GPU-enabled PyTorch install.')

    world_size = args.gpus or torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError('No GPUs detected.')

    mp.spawn(
        worker,
        args=(world_size, args.tif_file, args.out_csv, args.target_size, args.max_blocks),
        nprocs=world_size,
        join=True
    )

    df = combine_parts(args.out_csv, world_size)
    print(df[['kappa_hat', 'reml_loglik']].describe(include='all'))


if __name__ == '__main__':
    main()