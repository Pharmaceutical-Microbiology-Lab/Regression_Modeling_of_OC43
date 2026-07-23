import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

def add_qc_metrics(adata, use_layer='counts'):
    """
    Calculates standard QC metrics including mitochondrial and viral (OC43) genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_layer : str, optional
        The layer to use for QC calculation. If 'counts' and not present, 
        X will be copied to this layer. Default is 'counts'.
    """
    # 0. Ensure raw count layer exists
    if use_layer == 'counts' and 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()

    # 1. Define gene masks in .var
    # Mitochondrial genes (starts with 'MT-')
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')

    # OC43 Viral genes (starts with 'OC43_')
    # Check both index and 'gene_ids' column if available
    oc43_by_name = adata.var_names.str.startswith('OC43_')
    if 'gene_ids' in adata.var.columns:
        oc43_by_id = adata.var['gene_ids'].astype(str).str.startswith('OC43_')
        adata.var['oc43'] = oc43_by_name | oc43_by_id
    else:
        adata.var['oc43'] = oc43_by_name

    # 2. Calculate QC metrics using Scanpy
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'oc43'],
        inplace=True,
        log1p=False,
        layer=use_layer
    )

    # 3. Calculate viral-specific derived metrics
    # Viral fraction and per 10k normalized values
    adata.obs['viral_fraction'] = adata.obs['pct_counts_oc43'] / 100.0
    adata.obs['viral_per_10k']  = 1e4 * adata.obs['viral_fraction']
    adata.obs['y_log1p_per10k'] = np.log1p(adata.obs['viral_per_10k'])
    
    print(f"QC metrics added. Viral counts detected in {np.sum(adata.obs['total_counts_oc43'] > 0)} cells.")

    return adata 

def basic_qc_filter(adata, pct_mt_max=20.0, min_genes=2000, min_cells=3, 
                    drop_doublet=True, drop_unassigned=True):
    """
    Filters cells and genes based on quality control thresholds.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pct_mt_max : float
        Maximum allowed mitochondrial gene percentage.
    min_genes : int
        Minimum number of genes expressed per cell. default 2000
    min_cells : int
        Minimum number of cells expressing a gene.
    drop_doublet : bool
        Whether to remove cells labeled as 'doublet' in .obs['condition'].
    drop_unassigned : bool
        Whether to remove cells labeled as 'unassigned' in .obs['condition'].
    
    Returns
    -------
    AnnData : Filtered AnnData object.
    """

    # 1. Filter genes and cells using Scanpy defaults
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    # 2. Filtering based on custom metadata
    keep = np.ones(adata.n_obs, dtype=bool)

    # Mitochondrial percentage threshold
    if 'pct_counts_mt' in adata.obs:
        keep &= (adata.obs['pct_counts_mt'] < pct_mt_max).to_numpy()

    # Condition-based filtering (Doublets, Unassigned)
    if 'condition' in adata.obs:
        if drop_doublet:
            keep &= (adata.obs['condition'].astype(str) != 'doublet').to_numpy()
        if drop_unassigned:
            keep &= (adata.obs['condition'].astype(str) != 'unassigned').to_numpy()

    adata = adata[keep].copy()
    
    print(f"Filtering complete. Remaining cells: {adata.n_obs}, Genes: {adata.n_vars}")
    return adata

def normalize_log1p(adata, target_sum=1e4):
    """
    Performs total count normalization and log1p transformation.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    print(f"Normalization (target_sum={target_sum}) and log1p transformation applied.")
    return adata

def generate_pseudoreplicate_bulk(adata, group_col="infection_group", 
                                  min_cells_per_rep=100, 
                                  max_reps_per_group=3,
                                  min_samples_expressing_gene=3,
                                  random_state=777):
    """
    Creates pseudo-replicates for single-cell data by randomly splitting cells 
    within each group and summing their counts. Useful for DESeq2 analysis.
    """
    print(f"Generating pseudobulk for groups in: {group_col}")
    
    rng = np.random.default_rng(random_state)
    
    # Get raw counts
    if "counts" in adata.layers:
        X = adata.layers["counts"]
    elif adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    
    genes = adata.var_names.to_list()
    obs = adata.obs.copy()
    cell_pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)

    col_series_list = []
    meta_rows = []

    groups = obs.groupby(group_col).indices

    for grp, cell_names_idx in groups.items():
        cell_names = obs.index.take(list(cell_names_idx))
        n = len(cell_names)
        if n < min_cells_per_rep:
            print(f"Skipping group {grp}: Insufficient cells ({n})")
            continue

        # Decide K (number of replicates)
        K = min(int(n // min_cells_per_rep), max_reps_per_group)
        if K < 2: K = 2 # Minimum 2 reps for DESeq2
        
        pos = cell_pos.loc[cell_names].to_numpy()
        pos = rng.permutation(pos)
        splits = np.array_split(pos, K)

        for k, split_idx in enumerate(splits, start=1):
            if len(split_idx) == 0: continue

            # Sum counts across cells
            v = np.asarray(X[split_idx, :].sum(axis=0)).ravel()
            v = np.rint(v).astype(np.int64) 

            ps_id = f"{grp}_rep{k}"
            col_series_list.append(pd.Series(v, index=genes, name=ps_id))

            meta_rows.append({
                "pseudo_sample": ps_id,
                "infection_group": grp,
                "replicate": k,
                "n_cells": len(split_idx),
                "umi_sum": v.sum(),
            })

    pb_counts = pd.concat(col_series_list, axis=1)
    pb_coldata = pd.DataFrame(meta_rows).set_index("pseudo_sample")

    # Filter genes expressed in at least N pseudo-samples
    expressed_mask = (pb_counts > 0).sum(axis=1) >= min_samples_expressing_gene
    pb_counts = pb_counts.loc[expressed_mask]

    print(f"Completed: {pb_counts.shape[1]} samples, {pb_counts.shape[0]} genes.")
    return pb_counts, pb_coldata