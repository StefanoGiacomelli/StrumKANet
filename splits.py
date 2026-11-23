"""Partition utilities (pure Python): recording-disjoint, class-stratified splits for excerpt datasets.

The stored 2025 experiments used an unseeded excerpt-level ``random_split`` that is drawn again on every
``setup()`` call (see docs/PROTOCOL.md). The functions below provide the alternative used by the default-off options
``--split grouped`` and by ``main_recording_disjoint.py``: whole recordings are assigned to one part, so no recording
contributes excerpts to two parts, and the class proportions of every part follow the requested ratios.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence


def grouped_partition(groups: Sequence[str], labels: Optional[Sequence[int]], ratios: Sequence[float], seed: int) -> List[List[int]]:
    """Recording-disjoint, class-stratified partition of excerpt indices.

    Args:
        groups: recording identifier of every excerpt (``groups[i]`` belongs to excerpt ``i``).
        labels: class of every excerpt (constant within a recording) or ``None`` for no stratification.
        ratios: target share of every part (normalised to sum to 1), e.g. ``(0.8, 0.1, 0.1)`` or ``[0.2] * 5``.
        seed: seed of the (local) random generator; the result is deterministic and independent of the global RNG.

    Returns:
        One sorted list of excerpt indices per part. Greedy deficit assignment: within each class, recordings are
        shuffled (seeded), sorted by decreasing size, and every recording goes to the part with the largest remaining
        deficit (target excerpts of that class minus what the part already holds).
    """
    if len(groups) == 0:
        raise ValueError("no excerpts to partition")
    total = float(sum(ratios))
    shares = [r / total for r in ratios]
    members: Dict[str, List[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    stratum_of: Dict[str, object] = {}
    for group, indices in members.items():
        classes = {labels[i] for i in indices} if labels is not None else {0}
        if len(classes) != 1:
            raise ValueError(f"recording {group!r} has more than one label: {sorted(classes)}")
        stratum_of[group] = classes.pop()
    rng = random.Random(seed)
    parts: List[List[int]] = [[] for _ in shares]
    for stratum in sorted({s for s in stratum_of.values()}):
        stratum_groups = sorted(g for g in members if stratum_of[g] == stratum)       # deterministic base order
        rng.shuffle(stratum_groups)
        stratum_groups.sort(key=lambda g: -len(members[g]))                            # stable: random order among equal sizes
        stratum_total = sum(len(members[g]) for g in stratum_groups)
        held = [0.0] * len(shares)
        for group in stratum_groups:
            deficits = [share * stratum_total - held[p] for p, share in enumerate(shares)]
            target = max(range(len(shares)), key=lambda p: (deficits[p], -p))
            parts[target].extend(members[group])
            held[target] += len(members[group])
    return [sorted(part) for part in parts]


def grouped_kfold(groups: Sequence[str], labels: Optional[Sequence[int]], k: int, seed: int) -> List[List[int]]:
    """Stratified, recording-disjoint K-fold: ``k`` disjoint parts of (almost) equal size covering every excerpt once."""
    if k < 2:
        raise ValueError("k must be at least 2")
    return grouped_partition(groups, labels, [1.0 / k] * k, seed)


def mixed_group_kfold(groups: Sequence[str], labels: Sequence[int], k: int, seed: int) -> List[List[int]]:
    """Balanced K-fold for groups that may contain several classes (for example all recordings of one performer).

    Greedy assignment of whole groups, largest first (random order among equal sizes, seeded): every group goes to the fold where it
    increases the squared deviation of the fold's class-count vector from the target ``class total / k`` the least. For groups with a
    single class this is the largest-deficit rule of :func:`grouped_partition`, applied to all classes at once.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    classes = sorted(set(labels))
    members: Dict[str, List[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    counts = {g: [sum(1 for i in indices if labels[i] == c) for c in classes] for g, indices in members.items()}
    totals = [sum(1 for label in labels if label == c) for c in classes]
    targets = [total / k for total in totals]
    held = [[0.0] * len(classes) for _ in range(k)]
    order = sorted(members)                                                           # deterministic base order
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(members[g]))                                        # stable: random order among equal sizes
    parts: List[List[int]] = [[] for _ in range(k)]
    for group in order:
        count = counts[group]

        def increase(part: int) -> float:
            return sum(count[j] * (2 * (held[part][j] - targets[j]) + count[j]) / totals[j] ** 2 for j in range(len(classes)))

        target = min(range(k), key=lambda part: (increase(part), part))
        parts[target].extend(members[group])
        for j in range(len(classes)):
            held[target][j] += count[j]
    return [sorted(part) for part in parts]


def mixed_group_partition(groups: Sequence[str], labels: Sequence[int], ratios: Sequence[float], seed: int) -> List[List[int]]:
    """Two-part version of :func:`mixed_group_kfold` (``ratios`` such as ``(0.875, 0.125)``: the smaller share must be 1/k of the total)."""
    if len(ratios) != 2:
        raise ValueError("two parts are supported")
    k = round(sum(ratios) / min(ratios))
    if abs(min(ratios) / sum(ratios) - 1.0 / k) > 1e-9:
        raise ValueError("the smaller share must be 1/k of the total")
    folds = mixed_group_kfold(groups, labels, k, seed)
    held_out = set(folds[0])
    parts = [[i for i in range(len(groups)) if i not in held_out], sorted(held_out)]
    return parts if ratios[0] >= ratios[1] else parts[::-1]


def check_partition(parts: Sequence[Sequence[int]], groups: Sequence[str], n_excerpts: int, disjoint_groups: bool = True) -> Dict[str, object]:
    """Validate a partition (coverage, disjoint excerpts and, if requested, disjoint recordings) and summarise it."""
    flat = [i for part in parts for i in part]
    if sorted(flat) != list(range(n_excerpts)):
        raise ValueError("the parts do not cover every excerpt exactly once")
    recordings = [sorted({groups[i] for i in part}) for part in parts]
    shared = sorted(set.intersection(*[set(r) for r in recordings])) if len(parts) > 1 else []
    overlap = sorted({g for a in range(len(parts)) for b in range(a + 1, len(parts)) for g in set(recordings[a]) & set(recordings[b])})
    if disjoint_groups and overlap:
        raise ValueError(f"{len(overlap)} recordings appear in more than one part (e.g. {overlap[:3]})")
    return {"excerpts": [len(p) for p in parts], "recordings": [len(r) for r in recordings], "recordings_in_more_than_one_part": len(overlap),
            "recordings_in_every_part": len(shared)}


def class_counts(parts: Sequence[Sequence[int]], labels: Sequence[int]) -> List[Dict[int, int]]:
    out = []
    for part in parts:
        counts: Dict[int, int] = defaultdict(int)
        for i in part:
            counts[labels[i]] += 1
        out.append(dict(sorted(counts.items())))
    return out


def save_partition(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def load_partition(path: str) -> Dict[str, object]:
    with open(path) as handle:
        return json.load(handle)
