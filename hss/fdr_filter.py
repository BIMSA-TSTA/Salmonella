# -*- coding: utf-8 -*-
"""Benjamini-Hochberg FDR post-processing for HSS-SHAPi.


"""
import argparse
import os
import re

import numpy as np
import pandas as pd


def bh_qvalues(p):
    """Benjamini-Hochberg adjusted p values, capped at 1."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    q = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    m = len(pv)
    prev = 1.0
    for rank in range(m, 0, -1):
        j = order[rank - 1]
        val = pv[j] * m / rank
        if val < prev:
            prev = val
        q[idx[j]] = min(prev, 1.0)
    return q


def norm_gene(x):
    return str(x).split("~~~")[0].replace(" cluster", "").strip()


def add_key(df):
    if "Gene_A" in df.columns and "Gene_B" in df.columns:
        a = df["Gene_A"].map(norm_gene)
        b = df["Gene_B"].map(norm_gene)
        df = df.copy()
        df["_pair_key"] = [
            "||".join(sorted((x, y))) for x, y in zip(a, b)
        ]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="human, pig or chicken")
    ap.add_argument("--permutation-table", required=True,
                    help="per-pair permutation threshold CSV from permutation1000.py")
    ap.add_argument("--hss-table", default=None,
                    help="optional HSS candidate CSV from bootstrap.py")
    ap.add_argument("--out-dir", default="hss/fdr_out")
    ap.add_argument("--fdr", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = pd.read_csv(args.permutation_table)
    if "Permutation_Empirical_P" not in d.columns:
        raise SystemExit("Permutation_Empirical_P column not found")

    d["BH_q"] = bh_qvalues(d["Permutation_Empirical_P"].values)
    d["Pass_FDR"] = d["BH_q"] < args.fdr

    hss_n = 0
    edge_n = 0
    if args.hss_table:
        h = pd.read_csv(args.hss_table)
        d = add_key(d)
        h = add_key(h)
        if "Pair_ID" in d.columns and "Pair_ID" in h.columns:
            hss_ids = set(h["Pair_ID"].astype(int))
            d["HSS_candidate"] = d["Pair_ID"].astype(int).isin(hss_ids)
        elif "_pair_key" in d.columns and "_pair_key" in h.columns:
            hss_keys = set(h["_pair_key"])
            d["HSS_candidate"] = d["_pair_key"].isin(hss_keys)
        else:
            raise SystemExit("Could not match HSS table: need Pair_ID or Gene_A/Gene_B")
        d["Network_edge"] = d["HSS_candidate"] & d["Pass_FDR"]
        hss_n = int(d["HSS_candidate"].sum())
        edge_n = int(d["Network_edge"].sum())
    else:
        d["HSS_candidate"] = False
        d["Network_edge"] = False

    out_pairs = os.path.join(args.out_dir, f"{args.host}_pairwise_P_BHq.csv")
    d.to_csv(out_pairs, index=False)

    summary = pd.DataFrame([{
        "Host": "swine" if args.host == "pig" else args.host,
        "Tested_pairs": int(len(d)),
        "Permutation_P_lt_0.05": int((d["Permutation_Empirical_P"] < 0.05).sum()),
        "HSS_candidates": hss_n,
        "BH_FDR_lt_0.10": int((d["BH_q"] < args.fdr).sum()),
        "HSS_and_FDR_lt_0.10_network_edges": edge_n,
        "Min_BH_q": float(np.nanmin(d["BH_q"])),
    }])
    out_summary = os.path.join(args.out_dir, f"{args.host}_FDR_summary.csv")
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print("written:", out_pairs)
    print("written:", out_summary)


if __name__ == "__main__":
    main()
