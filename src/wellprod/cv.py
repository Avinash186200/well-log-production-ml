"""Cross-validation splits for wells: random K-fold, grouped by pad, spatial blocks, and
leave-field-out.

Wells on one pad share a surface location, crew and rock; wells a few km apart share geology.
Random K-fold puts pad-mates on both sides of a split, so a model (or a nearest-neighbour
baseline) can score well by looking at the answer next door. Grouped and spatial splits keep
every group whole on one side.

All splitters return a list of (train_idx, test_idx) integer arrays over positions 0..n-1 and
are deterministic for a given seed.
"""
from __future__ import annotations

import numpy as np

SCHEMES = ("random", "pad", "block", "field")


def kfold_random(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    return [(np.sort(np.concatenate(folds[:i] + folds[i + 1:])), np.sort(folds[i]))
            for i in range(k)]


def kfold_grouped(groups, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Whole groups per fold. Groups are shuffled with the seed, then each goes to the fold
    with the fewest wells so far (largest groups first), which keeps folds balanced."""
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) < k:
        raise ValueError(f"{len(uniq)} groups cannot fill {k} folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_of = np.empty(len(uniq), dtype=int)
    size = np.zeros(k, dtype=int)
    for gi in order:
        f = int(np.argmin(size))
        fold_of[gi] = f
        size[f] += counts[gi]
    well_fold = fold_of[inv]
    idx = np.arange(len(groups))
    return [(idx[well_fold != f], idx[well_fold == f]) for f in range(k)]


def spatial_blocks(x_km, y_km, block_km: float) -> np.ndarray:
    """Square map cells of side `block_km`; the cell label is the group."""
    bx = np.floor(np.asarray(x_km) / block_km).astype(int)
    by = np.floor(np.asarray(y_km) / block_km).astype(int)
    return np.array([f"{i}_{j}" for i, j in zip(bx, by)])


def pad_blocks(wells, block_km: float) -> np.ndarray:
    """Block label per well, taken from its pad's centroid, so a pad whose wells straddle a
    cell edge still lands in one block (and one fold)."""
    cen = wells.groupby("pad_id")[["x_km", "y_km"]].transform("mean")
    return spatial_blocks(cen["x_km"], cen["y_km"], block_km)


def make_splits(scheme: str, wells, k: int, seed: int, block_km: float):
    """Splits for one scheme over the rows of `wells` (needs pad_id, field_id, x_km, y_km)."""
    if scheme == "random":
        return kfold_random(len(wells), k, seed)
    if scheme == "pad":
        return kfold_grouped(wells["pad_id"].to_numpy(), k, seed)
    if scheme == "block":
        return kfold_grouped(pad_blocks(wells, block_km), k, seed)
    if scheme == "field":
        fields = wells["field_id"].to_numpy()
        idx = np.arange(len(wells))
        return [(idx[fields != f], idx[fields == f]) for f in np.unique(fields)]
    raise ValueError(f"unknown scheme {scheme!r}")


def nearest_train_distance(wells, splits) -> np.ndarray:
    """For every test well (pooled over folds), the distance in km to the nearest well in its
    training set. Random K-fold leaves pad-mates a few hundred metres away."""
    xy = wells[["x_km", "y_km"]].to_numpy()
    out = []
    for tr, te in splits:
        d = np.sqrt(((xy[te][:, None, :] - xy[tr][None, :, :]) ** 2).sum(-1))
        out.append(d.min(axis=1))
    return np.concatenate(out)
