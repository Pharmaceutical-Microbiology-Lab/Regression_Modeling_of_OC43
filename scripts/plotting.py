import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import learning_curve
import scanpy as sc
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.patches as mpatches
import shap
from adjustText import adjust_text

# ===================================================================
# [Global Project Style Settings]
# ===================================================================
# 논문 전체에서 공통으로 사용할 색상과 순서 정의
SC_COLORS = {
    'No infection': '#4A90E2', 
    'Low infection': '#F5A623', 
    'High infection': '#D0021B'
}
SC_ORDER = ['No infection', 'Low infection', 'High infection']

def set_publication_style():
    """기본 폰트 및 스타일 설정"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def plot_performance_scatter(y_true, y_pred, 
                             xlabel='Actual value', 
                             ylabel='Predicted value', 
                             title=None,
                             ax=None,
                             save_path=None):
    """
    Generates a publication-quality scatter plot: Actual vs Predicted.
    Automatically calculates and displays R2, RMSE, and MAE.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    xlabel : str, optional
        Label for the X-axis.
    ylabel : str, optional
        Label for the Y-axis.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure is created.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    def _truncate_value(n, decimals=3):
        if n is None: return 0
        factor = 10.0 ** decimals
        return np.trunc(n * factor) / factor
    
    # 1. 데이터 타입 정리 (Pandas Series 등이 들어와도 처리되도록)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 2. 성능 지표 계산
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))

    # 3. Figure 준비 (ax가 없으면 새로 생성)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    else:
        fig = ax.figure

    # 4. 산점도 (Scatter Plot)
    ax.scatter(y_true, y_pred, 
               s=20, alpha=0.4, color='#FF8C42', 
               edgecolors='none', label='Predictions',
               rasterized=True) # 점이 많을 경우 벡터 그래픽 용량 최적화

    # 5. 회귀선 (Regression Line)
    sns.regplot(x=y_true, y=y_pred, 
                scatter=False, color='#C41E3A', 
                line_kws={'linewidth': 2, 'label': 'Regression line'},
                ax=ax)

    # 6. 완전 일치선 (Identity Line, y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    # 여백을 약간 둠
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction')

    # 7. 라벨 및 텍스트 설정
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    # 통계 박스
    textstr = (f'$R^2$ = {_truncate_value(r2):.3f}\n'
               f'RMSE = {_truncate_value(rmse):.3f}\n'
               f'MAE = {_truncate_value(mae):.3f}\n'
               f'n = {len(y_true):,}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # 8. 스타일 및 범례
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
              shadow=False, framealpha=0.9, edgecolor='gray')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=10, 
                   width=1.2, length=5, direction='out')

    # 9. 저장 (옵션)
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), save_path=None):
    """
    Generates a learning curve plot to evaluate model fit and training-validation gap.

    Parameters
    ----------
    model : estimator object
        The machine learning model to evaluate.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    cv : int, optional
        Number of cross-validation folds. Default is 5.
    train_sizes : array-like, optional
        Relative or absolute numbers of training examples that will be used to generate the learning curve.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    
    print("Generating learning curve...")
    start_time = time.time()

    # Calculate learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    # Calculate statistics
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Fill between for standard deviation
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                    alpha=0.15, color='#1f77b4')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                    alpha=0.15, color='#ff7f0e')
    
    # Plot means
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', 
            linewidth=2, label='Training Score')
    ax.plot(train_sizes_abs, val_mean, 'o-', color='#ff7f0e', 
            linewidth=2, label='Cross-Validation Score')

    # Final metrics for annotation
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val

    # Labels and Style
    ax.set_xlabel('Number of Training Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('$R^2$ Score', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Metrics text box
    textstr = (f'Final Training $R^2$ = {final_train:.4f}\n'
               f'Final Validation $R^2$ = {final_val:.4f}\n'
               f'Training-Validation Gap = {gap:.4f}')
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', bbox=props, family='monospace')

    elapsed = time.time() - start_time
    print(f"Learning curve generated in {elapsed:.1f}s")
    print(f"Final Training R2: {final_train:.4f}")
    print(f"Final Validation R2: {final_val:.4f}")
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_gene_violin_overlay(adata, genes, group_key='infection_group', 
                             order=None, palette=None, figsize=None, 
                             save_path=None):
    """
    Plots an overlay of Violin and Box plots for multiple genes across groups.
    Ensures color mapping strictly follows the provided 'order' by converting 
    the palette dictionary into an ordered list.
    """
    
    # 1. Setup layout grid
    num_genes = len(genes)
    ncols = 3 if num_genes >= 3 else num_genes
    nrows = int(np.ceil(num_genes / ncols))
    
    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    axes = axes.flatten() if num_genes > 1 else [axes]

    # 2. Extract data
    plot_df = sc.get.obs_df(adata, keys=[group_key] + genes)

    plot_df[group_key] = pd.Categorical(plot_df[group_key], categories=order, ordered=True)
    
    if order is None:
        order = plot_df[group_key].unique().tolist()
        
    # 3. FIX: Convert dict palette to an ordered LIST based on the 'order' parameter
    # This bypasses any internal categorical ordering issues in Seaborn/Pandas
    if isinstance(palette, dict):
        ordered_colors = [palette[group] for group in order]
    else:
        ordered_colors = palette # Fallback if already a list or string

    # 4. Plotting loop
    for idx, gene in enumerate(genes):
        ax = axes[idx]
        
        # Layer 1: Violin Plot (Distribution)
        sns.violinplot(
            data=plot_df, x=group_key, y=gene, 
            order=order,      # Controls X-axis position
            palette=ordered_colors, # Controls actual colors via ordered list
            ax=ax,
            inner=None, 
            density_norm='width', 
            saturation=0.8,
            linewidth=0, 
            alpha=0.7, 
            legend=False,
            dodge=False 
        )
        
        # Layer 2: Box Plot (Summary Statistics)
        sns.boxplot(
            data=plot_df, x=group_key, y=gene, order=order,
            ax=ax, width=0.18, color='white', showfliers=False,
            linewidth=1.5,
            boxprops={'zorder': 2, 'edgecolor': 'black', 'alpha': 0.9},
            medianprops={'color': 'black', 'linewidth': 2.0},
            whiskerprops={'color': 'black', 'linewidth': 1.2},
            capprops={'color': 'black', 'linewidth': 1.2}
        )
        
        # --- Styling ---
        ax.set_title(gene, fontweight='bold', fontsize=20, pad=15) 
        ax.set_ylabel('Log Expression', fontsize=18)
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        sns.despine(ax=ax)
        
        # X-axis tick labels
        if idx < (nrows - 1) * ncols and num_genes > ncols:
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)
        else:
            short_labels = [label.split()[0] for label in order]
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(short_labels, fontsize=18)

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overlay plots successfully saved to {save_path}")

    return fig, axes

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc

def plot_umap_categorical(adata, color_var, seed, palette=None, title=None, 
                          figsize=(7, 6), save_path=None):
    """
    Plots a categorical UMAP (e.g., cell types, conditions) with a fixed seed 
    to ensure reproducibility and prevent occlusion bias.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing UMAP coordinates.
    color_var : str
        Column name in adata.obs to color by.
    palette : dict or list, optional
        Custom color mapping. If None, uses matplotlib's default cycle.
    title : str, optional
        Plot title. Defaults to color_var name.
    seed : int, optional
        Random seed for shuffling plotting order.
    figsize : tuple, optional
        Figure size (width, height).
    save_path : str, optional
        Path to save figure (without extension). Saves as PNG and PDF.

    Returns
    -------
    fig, ax : matplotlib objects
    """
    # 1. Validation
    if color_var not in adata.obs:
        raise ValueError(f"'{color_var}' not found in adata.obs")

    # 2. Prepare Data
    umap_coords = adata.obsm['X_umap']
    categories = adata.obs[color_var]
    
    # Ensure categorical dtype
    if not isinstance(categories.dtype, pd.CategoricalDtype):
        categories = categories.astype('category')
    
    cat_names = categories.cat.categories

    # 3. Handle Color Palette
    if palette is None:
        # Use default matplotlib property cycle
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_map = {cat: default_colors[i % len(default_colors)] for i, cat in enumerate(cat_names)}
    elif isinstance(palette, dict):
        color_map = palette
    else:
        # Assuming palette is a list
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cat_names)}

    # Map categories to actual colors for each cell
    # Fill NaN or unmapped values with grey
    cell_colors = categories.map(color_map).fillna('#d3d3d3').astype(str).values

    # 4. Shuffle Order for Reproducibility & Visualization
    # Shuffling prevents specific clusters from hiding behind others
    n_obs = adata.n_obs
    indices = np.arange(n_obs)
    
    # Set seed explicitly for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Reorder coordinates and colors
    coords_shuffled = umap_coords[indices]
    colors_shuffled = cell_colors[indices]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot all points at once using the shuffled order
    ax.scatter(coords_shuffled[:, 0], coords_shuffled[:, 1],
               c=colors_shuffled,
               s=4, 
               alpha=0.8, 
               edgecolors='none',
               rasterized=True) # Rasterized is crucial for heavy vector graphics (PDF)

    # 6. Styling
    ax.set_xlabel('UMAP1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP2', fontsize=12, fontweight='bold')
    
    final_title = title if title else color_var
    ax.set_title(final_title, fontsize=14, fontweight='bold', pad=12)

    # Remove ticks and spines for a clean look
    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # 7. Create Custom Legend
    # Since we plotted everything at once, we need to build the legend manually
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=cat,
               markerfacecolor=color_map.get(cat, '#d3d3d3'), markersize=8)
        for cat in cat_names
    ]
    
    ax.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5), 
              frameon=False, 
              fontsize=10, 
              title=color_var,
              title_fontsize=11)

    # 8. Saving
    if save_path:
        # Save layout tightly
        plt.tight_layout()
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
        print(f"Categorical UMAP saved to: {save_path}")

    return fig, ax


def plot_umap_continuous(adata, color_var, title=None, cmap=None, 
                         vmax_percentile=99, figsize=(7, 6), save_path=None):
    """
    Plots a continuous UMAP (e.g., gene expression, scores).
    Reproducibility is guaranteed by deterministic sorting.
    
    High values are plotted on top of low values to ensure signals are visible.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color_var : str
        Column name in adata.obs or gene name in adata.var_names.
    title : str, optional
        Plot title.
    cmap : str or Colormap, optional
        Matplotlib colormap. Default is a custom grey-yellow-red gradient.
    vmax_percentile : int, optional
        Percentile to determine max color value (robust to outliers).
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig, ax : matplotlib objects
    """
    # 1. Validation & Data Retrieval
    if color_var in adata.obs:
        values = adata.obs[color_var].values
    elif color_var in adata.var_names:
        # Check if X is sparse or dense and extract flattened array
        if hasattr(adata[:, color_var].X, "toarray"):
            values = adata[:, color_var].X.toarray().flatten()
        else:
            values = adata[:, color_var].X.flatten()
    else:
        raise ValueError(f"'{color_var}' not found in adata.obs or adata.var_names")

    umap_coords = adata.obsm['X_umap']

    # 2. Define Colormap
    if cmap is None:
        # Standard publication style: Grey (low) -> Yellow -> Red (high)
        colors = ['#E0E0E0', '#FFFF00', '#D0021B'] 
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors)

    # 3. Handle Outliers (Vmax)
    v_max = np.percentile(values, vmax_percentile)
    
    # 4. Sort indices by value (Ascending)
    # This ensures cells with high expression are drawn LAST (on top)
    # This is better than shuffling for feature plots.
    sort_indices = np.argsort(values)
    
    values_sorted = values[sort_indices]
    coords_sorted = umap_coords[sort_indices]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    scatter = ax.scatter(coords_sorted[:, 0], coords_sorted[:, 1],
                         c=values_sorted, 
                         s=4, 
                         alpha=0.8, 
                         cmap=cmap,
                         vmin=0, 
                         vmax=v_max,
                         edgecolors='none', 
                         rasterized=True) # Rasterized ensures smaller PDF size
    
    # 6. Styling
    ax.set_xlabel('UMAP1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP2', fontsize=12, fontweight='bold')
    
    final_title = title if title else color_var
    ax.set_title(final_title, fontsize=14, fontweight='bold', pad=12)

    sns.despine(ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # 7. Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% of OC43 counts', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    # Make colorbar outline cleaner
    cbar.outline.set_linewidth(1)

    # 8. Saving
    if save_path:
        plt.tight_layout()
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
        print(f"Continuous UMAP saved to: {save_path}")
        
    return fig, ax

def plot_gene_expression_series(adata, genes, group_key='infection_group', 
                                order=SC_ORDER,
                                ncols=3, figsize=None, save_path=None):
    """
    Plots gene expression trends across infection groups with error bars (SEM).
    Automatically assigns panel letters (a, b, c...) to each subplot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : list
        List of genes to plot.
    group_key : str
        The column in adata.obs to group by.
    order : list
        The order of groups on the X-axis.
    ncols : int
        Number of columns in the figure grid.
    figsize : tuple, optional
        Figure size. If None, it's calculated based on rows/cols.
    save_path : str, optional
        Path to save the figure.
    """
    
    # 1. Check gene availability
    available_genes = [g for g in genes if g in adata.var_names]
    missing_genes = [g for g in genes if g not in adata.var_names]
    
    if missing_genes:
        print(f"Warning: Genes not found in adata: {missing_genes}")
    if not available_genes:
        print("Error: No available genes to plot.")
        return None

    # 2. Calculate layout
    num_genes = len(available_genes)
    nrows = int(np.ceil(num_genes / ncols))
    if figsize is None:
        figsize = (4 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    axes = axes.flatten() if num_genes > 1 else [axes]

    # 3. Prepare Data (Mean and SEM)
    # Extracting all genes at once for efficiency
    gene_indices = [adata.var_names.get_loc(g) for g in available_genes]
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X[:, gene_indices].toarray()
    else:
        expr_matrix = adata.X[:, gene_indices]
        
    df = pd.DataFrame(expr_matrix, columns=available_genes)
    df[group_key] = adata.obs[group_key].values
    
    # Group and aggregate
    df_summary = df.groupby(group_key).agg(['mean', 'sem'])

    # 4. Plotting Loop
    x_pos = np.arange(len(order))
    short_labels = [label.split()[0] for label in order] # 'No infection' -> 'No'

    for idx, gene in enumerate(available_genes):
        ax = axes[idx]
        
        # Get stats for this gene
        means = [df_summary.loc[g, (gene, 'mean')] if g in df_summary.index else np.nan for g in order]
        sems = [df_summary.loc[g, (gene, 'sem')] if g in df_summary.index else np.nan for g in order]
        
        means = np.array(means)
        sems = np.array(sems)

        # Plot Styling
        # Error bars
        ax.errorbar(x_pos, means, yerr=sems, fmt='none', 
                    ecolor='black', capsize=5, capthick=1.5, 
                    linewidth=1.5, alpha=0.7, zorder=2)
        
        # Connection Line
        ax.plot(x_pos, means, '-', color='black', linewidth=2.5, zorder=3)
        
        # Data points (Publication style: white fill, black edge)
        ax.scatter(x_pos, means, s=120, color='white', 
                   edgecolors='black', linewidth=2.5, zorder=4)

        # Labels & Formatting

        ymin = 0
        ymax = means.max() + sems.max() * 4.0
        y_range = ymax - ymin
        ax.set_ylim(ymin - y_range * 0.02, ymax + y_range * 0.15)

        ax.set_xlim(-0.5, len(order) - 0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, fontsize=14)
        ax.set_ylabel('Log Expression', fontsize=10, fontweight='bold')
        ax.set_xlabel('Infection Group', fontsize=12, fontweight='bold')
        
        # Panel Letter and Title: (a) GeneName
        panel_letter = chr(97 + idx) # a, b, c...
        ax.set_title(f'({panel_letter}) {gene}', fontsize=12, fontweight='bold', loc='center', pad=10)

        # Axis Style
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', width=1.2, length=5)
        ax.tick_params(axis='y', labelsize=9)

        
        # Adjust Y-limit for markers
        if not np.all(np.isnan(means)):
            ymax = np.nanmax(means + sems)
            ax.set_ylim(bottom=0, top=ymax * 1.3)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"Gene expression plots saved to: {save_path}")

    return fig, axes

def plot_directed_paga_trajectory(adata_sc, adata_bulk=None, 
                                  sc_group_col='infection_group',
                                  sc_pseudotime_key='dpt_pseudotime',
                                  bulk_group_col='hpi', 
                                  sc_colors=SC_COLORS,
                                  sc_order=SC_ORDER,
                                  bulk_colors=None, 
                                  bulk_order=None, 
                                  paga_resolution=1.0,
                                  threshold=0.05,
                                  figsize=(12, 10),
                                  title="", 
                                  save_path=None):
    """
    Plots a directed PAGA trajectory on UMAP, highlighting the root (start) node 
    and optionally projecting bulk data points.

    Parameters
    ----------
    adata_sc : AnnData
        Single-cell AnnData with UMAP and pseudotime.
    adata_bulk : AnnData, optional
        Bulk AnnData projected onto the same UMAP space.
    sc_group_col : str
        Column in adata_sc.obs for background cell coloring.
    sc_pseudotime_key : str
        Column in adata_sc.obs containing pseudotime values.
    bulk_group_col : str
        Column in adata_bulk.obs for bulk point coloring.
    sc_colors : dict, optional
        Color map for single-cell groups.
    bulk_colors : dict, optional
        Color map for bulk groups.
    bulk_order : list, optional
        Order of bulk groups in the legend.
    paga_resolution : float
        Resolution for Leiden clustering used in PAGA.
    threshold : float
        Connectivity threshold for drawing edges between clusters.
    title : str
        Title for plot
    save_path : str, optional
        Path to save the figure (without extension).
    """
    
    print("Computing Directed PAGA Trajectory...")
    
    # 1. PAGA and Clustering (using a copy to protect original adata)
    adata_temp = adata_sc.copy()
    sc.tl.leiden(adata_temp, resolution=paga_resolution, key_added='leiden_paga')
    sc.tl.paga(adata_temp, groups='leiden_paga')
    
    # 2. Calculate Cluster Centroids and Average Pseudotime
    clusters = adata_temp.obs['leiden_paga'].unique().sort_values()
    umap_centroids = []
    cluster_pseudotime = []
    
    for clus in clusters:
        mask = adata_temp.obs['leiden_paga'] == clus
        centroid = np.mean(adata_temp.obsm['X_umap'][mask], axis=0)
        umap_centroids.append(centroid)
        avg_pt = np.mean(adata_temp.obs[sc_pseudotime_key][mask])
        cluster_pseudotime.append(avg_pt)
        
    umap_centroids = np.array(umap_centroids)
    cluster_pseudotime = np.array(cluster_pseudotime)
    
    # Identify Root (Cluster with minimum pseudotime)
    root_idx = np.argmin(cluster_pseudotime)
    root_coord = umap_centroids[root_idx]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # [Layer 1] Background: Single-cell Scatter
    if sc_colors is None:
        sc_colors = {cat: color for cat, color in zip(adata_sc.obs[sc_group_col].unique(), sns.color_palette("Set1"))}
    
    umap_sc = adata_sc.obsm['X_umap']
    for cat in sc_order:
        if cat in adata_sc.obs[sc_group_col].values:
            mask = (adata_sc.obs[sc_group_col] == cat)
            ax.scatter(umap_sc[mask, 0], umap_sc[mask, 1], 
                       c=sc_colors.get(cat, 'grey'),
                       s=20, alpha=0.15, edgecolors='none', 
                       rasterized=True, zorder=1, label=cat)

    # [Layer 2] Directed Edges (Arrows)
    connectivities = adata_temp.uns['paga']['connectivities'].todense()
    edge_color = '#333333'
    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if connectivities[i, j] > threshold:
                # Direction: Low Pseudotime -> High Pseudotime
                start_node, end_node = (i, j) if cluster_pseudotime[i] < cluster_pseudotime[j] else (j, i)
                p1, p2 = umap_centroids[start_node], umap_centroids[end_node]
                
                # Draw Line
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=edge_color, linewidth=2.5, alpha=0.4, zorder=4)
                # Draw Arrow Head
                ax.annotate('', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                            arrowprops=dict(arrowstyle='-|>', color=edge_color, lw=0, mutation_scale=25), 
                            zorder=5)

    # [Layer 3] Root Marker and "Start" Label
    ax.scatter(root_coord[0], root_coord[1], s=500, c='white', marker='D', 
               edgecolors='black', linewidth=3, zorder=20, label='Trajectory Start')
               
    ax.annotate('Start', xy=(root_coord[0], root_coord[1]), 
                xytext=(root_coord[0] - 1.5, root_coord[1] + 1.5),
                fontsize=16, fontweight='bold', color='black',
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                zorder=25)

    # [Layer 4] Foreground: Bulk Data (Stars)
    if adata_bulk is not None:
        umap_bulk = adata_bulk.obsm['X_umap']
        bulk_vals = adata_bulk.obs[bulk_group_col]
        groups_to_plot = bulk_order if bulk_order else np.unique(bulk_vals)
        
        for grp in groups_to_plot:
            if grp in bulk_vals.values:
                mask = (bulk_vals == grp)
                color = bulk_colors.get(grp, 'black') if bulk_colors else 'black'
                ax.scatter(umap_bulk[mask, 0], umap_bulk[mask, 1], c=[color], 
                           s=450, marker='*', edgecolors='black', linewidth=1.2, zorder=10, label=grp)

    # Final Styling
    ax.set_xlabel('UMAP1', fontweight='bold', fontsize=16)
    ax.set_ylabel('UMAP2', fontweight='bold', fontsize=16)
    ax.set_title(f"{title}", fontweight='bold', fontsize=20, pad=20)
    
    # Legend Handling
    handles, labels = ax.get_legend_handles_labels()
    # SC Legend
    sc_leg = ax.legend(handles[:len(adata_sc.obs[sc_group_col].unique())], 
                       labels[:len(adata_sc.obs[sc_group_col].unique())], 
                       title="Cell State (SC)", loc='upper left', bbox_to_anchor=(1.02, 1.0), 
                       fontsize=12, title_fontsize=14)
    for lh in sc_leg.legend_handles: lh.set_alpha(1); lh.set_sizes([100])
    ax.add_artist(sc_leg)
    
    # Bulk Legend (if exists)
    if adata_bulk is not None:
        bulk_leg = ax.legend(handles[-(len(groups_to_plot)):], labels[-(len(groups_to_plot)):], 
                             title="Timepoints (Bulk)", loc='upper left', bbox_to_anchor=(1.02, 0.75),
                             fontsize=12, title_fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(right=0.70)

    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"PAGA Trajectory plot saved to: {save_path}")
        
    return fig, ax

def _freedman_diaconis_bins(x):
    """Helper to calculate optimal bin width."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 2: return 50
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr == 0: return 50
    h = 2 * iqr * (x.size ** (-1/3))
    return max(int(np.ceil((x.max() - x.min()) / h)), 20)

def plot_distribution_hist(data, xlabel='Value', title=None, 
                           bins='auto', show_stats=True, save_path=None):
    """
    Plots a histogram with KDE and optional median/quartile lines.
    
    Parameters
    ----------
    data : array-like
        The data to plot.
    xlabel : str
        Label for the X-axis.
    title : str, optional
        Optional title or sample size info.
    bins : int or 'auto', optional
        Number of bins. If 'auto', uses Freedman-Diaconis rule.
    show_stats : bool
        If True, adds median and quartile lines.
    save_path : str, optional
        Path to save the figure.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    
    if bins == 'auto':
        bins = _freedman_diaconis_bins(data)
        
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    # Main plot
    sns.histplot(data, bins=bins, ax=ax, element="bars", 
                 edgecolor="white", stat="count", kde=True,
                 line_kws={"linewidth": 2, "alpha": 0.8})
    
    if show_stats:
        med = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        ax.axvline(med, color='#1f77b4', ls="--", lw=1.5, alpha=0.9, label="Median")
        ax.axvline(q1, color='#ff7f0e', ls=":", lw=1.5, alpha=0.9, label="Q1/Q3")
        ax.axvline(q3, color='#ff7f0e', ls=":", lw=1.5, alpha=0.9)
        
        ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1), 
                  edgecolor='gray', framealpha=0.9)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of cells", fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, loc='right', fontsize=10, color='gray')
    
    sns.despine(ax=ax)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        
    return fig, ax

def plot_split_axis_histogram(data, threshold_split, xlabel='Value', 
                              bins=150, top_ratio=0.25, figsize=(6, 4), 
                              save_path=None):
    """
    Plots a histogram with a broken Y-axis to visualize distributions 
    with extreme frequency differences.

    Parameters
    ----------
    data : array-like
        The data to plot.
    threshold_split : float
        The Y-axis value where the axis will be split.
    xlabel : str
        Label for the X-axis.
    bins : int or array-like
        Number of bins or bin edges.
    top_ratio : float
        The ratio of the top panel height relative to the whole figure.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure (without extension).
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    
    # 1. Calculate Histogram
    if isinstance(bins, int):
        bins = np.linspace(data.min(), data.max(), bins)
    counts, bin_edges = np.histogram(data, bins=bins)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_edges[1:] - bin_edges[:-1]

    # 2. Create Figure with Subplots
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=figsize, dpi=300,
        gridspec_kw={'height_ratios': [top_ratio, 1 - top_ratio], 'hspace': 0.08},
        sharex=True
    )

    # 3. Plot Bars in Both Panels
    color_bar = '#4E79A7'
    for ax in [ax_top, ax_bottom]:
        ax.bar(bin_centers, counts, width=bar_width, 
               edgecolor='white', linewidth=0.3, color=color_bar, align='center')

    # 4. Set Y-limits and Ticks
    # Top Panel: extreme frequencies
    ax_top.set_ylim(threshold_split + 1, counts.max() * 1.1)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Bottom Panel: main distribution
    ax_bottom.set_ylim(0, threshold_split)
    ax_bottom.spines['top'].set_visible(False)

    # Clean up y-ticks to avoid overlap
    yticks_top = ax_top.get_yticks()
    ax_top.set_yticks(yticks_top[yticks_top > threshold_split])
    
    yticks_bottom = ax_bottom.get_yticks()
    ax_bottom.set_yticks(yticks_bottom[yticks_bottom < threshold_split])

    # 5. Add Break Marks (the "//" symbol)
    d = 0.015  # size of the diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d*3, +d*3), **kwargs)        # Top-left break
    ax_top.plot((1-d, 1+d), (-d*3, +d*3), **kwargs)    # Top-right break

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1-d*1.5, 1+d*1.5), **kwargs)  # Bottom-left break
    ax_bottom.plot((1-d, 1+d), (1-d*1.5, 1+d*1.5), **kwargs) # Bottom-right break

    # 6. Statistical Lines (Added only to bottom for clarity)
    med = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    
    ax_bottom.axvline(med, color='#1f77b4', ls="--", lw=1.5, alpha=0.8, zorder=5)
    ax_bottom.axvline(q1, color='#ff7f0e', ls=":", lw=1.2, alpha=0.8, zorder=5)
    ax_bottom.axvline(q3, color='#ff7f0e', ls=":", lw=1.2, alpha=0.8, zorder=5)

    # 7. Labels and Legend
    ax_bottom.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    fig.text(0.04, 0.5, 'Number of cells', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    # Stats Text
    fig.text(0.98, 0.96, f'n = {len(data):,}', ha='right', va='top', fontsize=10, color='gray')

    # Custom Legend
    median_line = mpatches.Patch(color='#1f77b4', label='Median')
    q_line = mpatches.Patch(color='#ff7f0e', label='Q1/Q3')
    ax_bottom.legend(handles=[median_line, q_line], loc="upper right", frameon=True, fontsize=9)

    sns.despine(ax=ax_top, bottom=True)
    sns.despine(ax=ax_bottom, top=True)

    if save_path:
        plt.tight_layout(rect=[0.05, 0, 1, 1]) # Make room for y-label
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"Split-axis histogram saved to: {save_path}")

    return fig, (ax_top, ax_bottom)

def plot_shap_custom_features(shap_values, X_train, feature_names, target_features, 
                              plot_type="violin", figsize=(10, 6), save_path=None):
    """
    Plots a SHAP summary plot (violin/beeswarm) ONLY for a specific list of target features.
    
    Parameters
    ----------
    shap_values : np.array
        SHAP values calculated from the model.
    X_train : pd.DataFrame or np.array
        Training data used for SHAP calculation.
    feature_names : list
        Full list of feature names corresponding to columns in X_train.
    target_features : list
        List of specific feature names to visualize.
    plot_type : str, optional
        'violin' (density) or 'dot' (beeswarm). Default is 'violin'.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure.
    """
    
    # 1. Validate and Find Indices
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()
        
    target_indices = []
    found_features = []
    
    for feature in target_features:
        if feature in feature_names:
            target_indices.append(feature_names.index(feature))
            found_features.append(feature)
        else:
            print(f"⚠️ Warning: Feature '{feature}' not found in the model features.")
            
    if not target_indices:
        raise ValueError("No valid target features found to plot.")

    # 2. Slice Data (Extract only target columns)
    # Handle SHAP values slicing
    if isinstance(shap_values, list): # For multi-class models
        shap_values_subset = shap_values[0][:, target_indices] # Assuming class 0 or binary
    else:
        shap_values_subset = shap_values[:, target_indices]
        
    # Handle X matrix slicing
    if isinstance(X_train, pd.DataFrame):
        X_subset = X_train.iloc[:, target_indices].values
    else:
        X_subset = X_train[:, target_indices]

    # 3. Create Plot
    fig = plt.figure(figsize=figsize, dpi=300)
    
    # SHAP Summary Plot
    # show=False allows us to customize the plot with matplotlib afterwards
    shap.summary_plot(
        shap_values_subset, 
        X_subset,
        feature_names=found_features,
        plot_type=plot_type,
        show=False,
        max_display=len(found_features),
        plot_size=figsize,
        color_bar=True
    )
    
    # 4. Custom Styling (Publication Quality)
    ax = plt.gca()
    
    # Axis Labels
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=14, fontweight='bold')
    
    # Y-axis Feature Labels (Font settings)
    plt.yticks(fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    
    # Adjust Colorbar Label if it exists
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
        cbar_ax.tick_params(labelsize=10)

    plt.tight_layout()

    # 5. Save
    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"Custom SHAP plot saved to: {save_path}")
        
    return fig

def plot_shap_beeswarm(model, X_train, feature_names=None, top_k=20, save_path=None):
    """
    Generates a SHAP beeswarm plot for the top-k features.
    
    Parameters
    ----------
    model : object
        Trained machine learning model (e.g., CatBoost, XGBoost, Tree-based).
    X_train : pd.DataFrame or np.array
        Training data used for SHAP calculation.
    feature_names : list, optional
        List of feature names. If None, tries to use X_train.columns.
    top_k : int
        Number of top features to display.
    save_path : str, optional
        Path to save the figure.
    """
    
    # 1. Feature Names Handling
    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]
            
    # 2. SHAP Values Calculation
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # 3. Create Plot
    plt.figure(figsize=(8, 10), dpi=300)
    
    # Summary plot (Beeswarm)
    shap.summary_plot(
        shap_values, 
        X_train, 
        feature_names=feature_names,
        max_display=top_k,
        show=False,
        plot_size=(8, 10),
        color_bar=True
    )
    
    # 4. Customizing Style (Post-hoc)
    ax = plt.gca()
    fig = plt.gcf()
    
    # Fonts and Labels
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_k} SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
    
    # Tick parameters
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)
    
    # Adjust Colorbar Label if exists
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Feature value', fontsize=10, fontweight='bold')
        cbar_ax.tick_params(labelsize=9)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"SHAP beeswarm plot saved to: {save_path}")
        
    return fig


def plot_feature_importance(model, feature_names, importance_type='feature_importances_', 
                            top_k=20, save_path=None):
    """
    Plots feature importance from a trained model.
    Supports Scikit-learn style (.feature_importances_) and CatBoost (.get_feature_importance()).
    
    Parameters
    ----------
    model : object
        Trained model.
    feature_names : list or pd.Index
        Names of the features.
    importance_type : str
        Attribute name to retrieve importance (default: 'feature_importances_').
        For CatBoost, it handles 'get_feature_importance()' automatically.
    """
    
    # 1. Get Importance Values
    if hasattr(model, 'get_feature_importance'): # CatBoost specific
        importances = model.get_feature_importance()
    elif hasattr(model, importance_type): # Sklearn / XGBoost
        importances = getattr(model, importance_type)
    else:
        raise ValueError(f"Model does not have attribute '{importance_type}' or 'get_feature_importance'")

    # 2. Create DataFrame and Sort
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_k)
    
    # Reverse for horizontal bar plot (Top feature at top)
    fi_df = fi_df.sort_values(by='Importance', ascending=True)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300) # Height adjusts to K
    
    # Color gradient based on importance
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, len(fi_df)))
    
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # 4. Styling
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)
    
    # Add Grid
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True, bottom=False) # Remove left spine for cleaner look
    
    # Add Value Labels
    max_val = fi_df['Importance'].max()
    for rect in bars:
        width = rect.get_width()
        ax.text(width + max_val*0.02, rect.get_y() + rect.get_height()/2, 
                f'{width:.2f}', ha='left', va='center', fontsize=9, color='black')
        
    # Adjust X-limit for text space
    ax.set_xlim(0, max_val * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
        
    return fig, ax

def plot_pca_scatter(all_df, variance_df, title="PCA Analysis", 
                     group_col='group', sample_col='sample',
                     order=None, palette=None, figsize=(10, 8), save_path=None):
    """
    Plots a PCA scatter plot with sample labels and variance explained.
    """
    # 1. Prepare variance ratios
    vmap = variance_df.set_index('PC')['ratio']
    # Convert to percentage (0.1 -> 10.0 or 10.0 -> 10.0)
    pc1_ratio = vmap.loc['PC1'] * 100 if vmap.loc['PC1'] <= 1.0 else vmap.loc['PC1']
    pc2_ratio = vmap.loc['PC2'] * 100 if vmap.loc['PC2'] <= 1.0 else vmap.loc['PC2']

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 3. Scatter Plot
    # Apply standard SC_COLORS if palette is not provided
    scatter = sns.scatterplot(
        data=all_df, x='PC1', y='PC2', hue=group_col,
        palette=palette, s=150, ax=ax, edgecolor='black', linewidth=0.8
    )

    # 4. Annotate sample names with overlap adjustment
    texts = [
        ax.text(row['PC1'], row['PC2'], row[sample_col], fontsize=9)
        for _, row in all_df.iterrows()
    ]
    
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.6, alpha=0.7)
    )

    # 5. Styling
    ax.set_xlabel(f'PC1 ({pc1_ratio:.1f}%)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pc2_ratio:.1f}%)', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    sns.despine(ax=ax)

    # 6. Legend Formatting (e.g., handling m^3 units)
    handles, labels = ax.get_legend_handles_labels()
    # Replace units with LaTeX for better publication quality
    formatted_labels = [l.replace('m^3', r'm$^3$') for l in labels]
    
    ax.legend(handles, formatted_labels, loc='center left', 
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', bbox_inches='tight')
    
    return fig, ax