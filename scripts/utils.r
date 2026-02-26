#' Run Gene Ontology (BP) Enrichment Analysis
#'
#' This function performs GO Biological Process enrichment analysis using clusterProfiler,
#' saves the results as a CSV, and generates a publication-quality dotplot.
#'
#' @param csv_path Path to the input CSV file. Must contain a 'gene' column (Symbols).
#' @param plot_title Title for the generated dotplot.
#' @param out_dir_csv Directory to save the result CSV. Default is "./GO_results/".
#' @param out_dir_fig Directory to save the result Plot. Default is "./GO_plots/".
#' @param show_n Number of top categories to show in the plot. Default is 10.
#' @param p_cut p-value cutoff. Default is 0.05.
#' @param q_cut q-value cutoff. Default is 0.20.
#' @param universe_csv Optional path to a CSV file for the background gene set (universe).
#' @param org_db Organism database to use. Default is "org.Hs.eg.db" (Human).
#'
#' @return A list containing the enrichment results and output paths (invisibly).
#' @export
run_go_enrichment <- function(csv_path,
                              plot_title,
                              out_dir_csv = "./GO_results/",
                              out_dir_fig = "./GO_plots/",
                              show_n = 10,
                              p_cut = 0.05,
                              q_cut = 0.20,
                              universe_csv = NULL,
                              org_db = "org.Hs.eg.db") {
  
  # ---- Load Dependencies ----
  required_pkgs <- c("clusterProfiler", org_db, "AnnotationDbi", "enrichplot", "ggplot2", "tools")
  for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("Package '%s' is required but not installed.", pkg))
    }
  }
  
  # ---- Directory Preparation ----
  if (!file.exists(csv_path)) stop("Input CSV not found: ", csv_path)
  if (!dir.exists(out_dir_csv)) dir.create(out_dir_csv, recursive = TRUE)
  if (!dir.exists(out_dir_fig)) dir.create(out_dir_fig, recursive = TRUE)
  
  tag <- tools::file_path_sans_ext(basename(csv_path))
  message(sprintf("[*] Processing: %s", tag))
  
  # ---- Load Data ----
  df <- utils::read.csv(csv_path, check.names = FALSE)
  if (!("Gene_symbol" %in% colnames(df))) {
    stop("Input CSV must contain a 'gene' column with Gene Symbols.")
  }
  
  # ---- Internal Helper: Symbol to Entrez ID ----
  get_entrez <- function(symbols, db) {
    symbols <- unique(trimws(as.character(symbols)))
    symbols <- symbols[nzchar(symbols)]
    
    if (length(symbols) == 0) return(NULL)
    
    suppressMessages({
      map <- clusterProfiler::bitr(
        symbols, 
        fromType = "SYMBOL", 
        toType   = "ENTREZID", 
        OrgDb    = db
      )
    })
    return(unique(map$ENTREZID))
  }
  
  # ---- ID Mapping ----
  gene_list <- get_entrez(df$Gene_symbol, org_db)
  if (is.null(gene_list)) stop("No valid Gene Symbols found for mapping.")
  
  universe_list <- NULL
  if (!is.null(universe_csv)) {
    udf <- utils::read.csv(universe_csv, check.names = FALSE)
    if ("Gene_symbol" %in% colnames(udf)) {
      universe_list <- get_entrez(udf$Gene_symbol, org_db)
    }
  }
  
  message(sprintf("Mapped %d genes to Entrez IDs.", length(gene_list)))
  
  # ---- GO Enrichment (Biological Process) ----
  ego <- clusterProfiler::enrichGO(
    gene          = gene_list,
    universe      = universe_list,
    OrgDb         = org_db,
    keyType       = "ENTREZID",
    ont           = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff  = p_cut,
    qvalueCutoff  = q_cut,
    readable      = TRUE
  )
  
  ego_df <- as.data.frame(ego)
  if (nrow(ego_df) == 0) {
    message("No significant GO terms found.")
    return(invisible(NULL))
  }
  
  # ---- Save Results ----
  out_csv_path <- file.path(out_dir_csv, paste0(tag, "_GO_results.csv"))
  utils::write.csv(ego_df, out_csv_path, row.names = FALSE)
  
  # ---- Visualization (Publication Quality) ----
  p <- enrichplot::dotplot(ego, showCategory = show_n) +
    ggplot2::ggtitle(plot_title) +
    ggplot2::labs(color = "p.adjust", size = "Count") +
    ggplot2::theme_bw(base_size = 14) +
    ggplot2::theme(
      plot.title   = ggplot2::element_text(hjust = 0.5, face = "bold", size = 18),
      axis.title   = ggplot2::element_text(size = 14),
      axis.text    = ggplot2::element_text(color = "black"),
      legend.title = ggplot2::element_text(size = 12),
      legend.text  = ggplot2::element_text(size = 10),
      panel.grid.major = ggplot2::element_line(linewidth = 0.2, color = "grey90")
    ) +
    # Adding a border to dots for better visibility in papers
    ggplot2::geom_point(ggplot2::aes(size = Count, color = p.adjust), 
                        shape = 21, colour = "black", stroke = 0.5)
  
  out_fig_path <- file.path(out_dir_fig, sprintf("%s_Top%d_GO_dotplot.png", tag, show_n))
  ggplot2::ggsave(out_fig_path, plot = p, width = 8, height = 7, dpi = 300)
  
  message(sprintf("[+] Results saved to: %s", out_dir_csv))
  
  return(invisible(list(
    data = ego_df,
    plot = p,
    csv_path = out_csv_path,
    fig_path = out_fig_path
  )))
}


analyze_save <- function(dds_obj, condition , coefficient, control, save_path , lfc_cutoff = 0.58, padj_cutoff = 0.05) {

  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("Package 'tibble' needed for this function to work. Please install it.")
  }
  
  message(paste("Processing:", coefficient, "versus", control, "..."))
  
  res <- results(dds_obj, contrast = c(condition, coefficient, control))
  
  res_ordered <- res[order(res$padj), ]

  res_df <- res_ordered %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "Gene_symbol")
  
  sig_deg <- res_df %>%
    as.data.frame() %>%
    dplyr::filter(padj < padj_cutoff & abs(log2FoldChange) > lfc_cutoff)
  
  write.csv(res_df, file = paste0(save_path, "DEG_results_", coefficient, "_vs_", control, ".csv"), row.names = FALSE, col.names = TRUE)
  write.csv(sig_deg, file = paste0(save_path, "Sig_DEG_", coefficient, "_vs_", control, "_LFC", lfc_cutoff, ".csv"), row.names = FALSE, col.names = TRUE)
  
  message(paste("Done:", coefficient, "| Found", nrow(sig_deg), "DEGs"))
}