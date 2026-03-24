# bayesweb.R — R wrapper for the bayesweb Python package
# Requires: install.packages("reticulate") + pip install bayesweb
#
# Quick start:
#   source("bayesweb.R")
#   bw_init()
#   web <- bw_new("Mon territoire")
#   web <- bw_add_node(web, "A", "Etalement urbain", alpha=3, beta=1, zone="N")
#   web <- bw_add_node(web, "B", "Biodiversite",     alpha=2, beta=2, zone="P")
#   web <- bw_add_edge(web, "A", "B", weight=0.7)
#   res <- bw_run(web, target="A", kappa=5, sigma=0.9)
#   print(res$indicators)
#   bw_plot_propagation(res, web)

library(reticulate)
.bw <- new.env(parent=emptyenv())

bw_init <- function(python=NULL, condaenv=NULL, virtualenv=NULL) {
  if (!is.null(python))     use_python(python,     required=TRUE)
  if (!is.null(condaenv))   use_condaenv(condaenv, required=TRUE)
  if (!is.null(virtualenv)) use_virtualenv(virtualenv, required=TRUE)
  .bw$pkg  <- import("bayesweb")
  .bw$Zone <- .bw$pkg$Zone
  .bw$copy <- import("copy")
  invisible(NULL)
}
.chk <- function() if (is.null(.bw$pkg)) stop("Call bw_init() first.")

bw_new  <- function(name="BeliefWeb") { .chk(); .bw$pkg$BeliefWeb(name) }

bw_add_node <- function(web, node_id, label, alpha=1.0, beta=1.0, zone="I") {
  .chk()
  web$add_node(node_id, label, alpha=alpha, beta=beta, zone=.bw$Zone(zone))
  web
}
bw_add_edge           <- function(web,s,t,w) { web$add_edge(s,t,w); web }
bw_add_undirected_edge<- function(web,i,j,w) { web$add_undirected_edge(i,j,w); web }
bw_auto_partition     <- function(web)        { web$auto_partition(); web }

bw_run <- function(web, target, kappa=5.0, sigma=0.9, lam=0.9, max_iter=500L) {
  .chk()
  shock_info  <- py_to_r(.bw$pkg$shock_summary(web, target, kappa, sigma))
  wc          <- .bw$copy$deepcopy(web)
  res_list    <- .bw$pkg$apply_shock(wc, target, kappa, sigma, inplace=TRUE)
  updated_web <- res_list[[1]]; dl <- res_list[[2]]
  result      <- .bw$pkg$propagate(updated_web, target, dl, lam=lam, max_iter=as.integer(max_iter))
  list(
    indicators = as.data.frame(py_to_r(.bw$pkg$indicators_summary(web, result))),
    nodes      = py_to_r(result$to_dataframe()),
    shock      = shock_info,
    result     = result
  )
}

bw_scan <- function(web, target, kappa=5.0, sigma=0.9) {
  .chk()
  sc <- .bw$pkg$scan_tipping_point(web, target, kappa, sigma)
  list(eta_values=py_to_r(sc$eta_values), G_values=py_to_r(sc$G_values),
       eta_rho=py_to_r(sc$eta_rho_values), eta_critical=sc$eta_critical)
}

bw_plot_propagation <- function(run_result, web, top_n=15) {
  if (!requireNamespace("ggplot2",quietly=TRUE)) stop("install.packages('ggplot2')")
  library(ggplot2)
  df <- run_result$nodes
  df$zone  <- sapply(df$node_id, function(n) web$get_node(n)$zone$value)
  df$label <- sapply(df$node_id, function(n) substr(web$get_node(n)$label,1,40))
  df <- df[order(-df$delta_p),][seq_len(min(top_n,nrow(df))),]
  ggplot(df, aes(x=reorder(label,delta_p), y=delta_p, fill=zone)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=c(N="#7F77DD",I="#1D9E75",P="#888780"), name="Zone") +
    coord_flip() + labs(title="Belief shift |delta_p|", x=NULL, y="delta_p") +
    theme_minimal(base_size=12)
}

bw_plot_tipping <- function(scan_result) {
  if (!requireNamespace("ggplot2",quietly=TRUE)) stop("install.packages('ggplot2')")
  library(ggplot2)
  df <- data.frame(eta_rho=scan_result$eta_rho, G=scan_result$G_values)
  ggplot(df, aes(x=eta_rho,y=G)) + geom_line(colour="#7F77DD",linewidth=1.2) +
    geom_vline(xintercept=1.0,colour="#D85A30",linetype="dashed") +
    geom_hline(yintercept=1.0,colour="#888780",linetype="dotted") +
    labs(title="Tipping-point scan: G vs eta*rho(W)", x="eta * rho(W)", y="G") +
    theme_minimal(base_size=12)
}

bw_summary   <- function(web)       cat(web$summary(), "\n")
bw_from_json <- function(path)      { .chk(); .bw$pkg$BeliefWeb$from_json(path) }
bw_to_json   <- function(web, path) { web$to_json(path); invisible(path) }
