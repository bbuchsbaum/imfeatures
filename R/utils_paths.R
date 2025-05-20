.common_root <- function(paths) {
  norm_paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
  split_paths <- strsplit(norm_paths, "/")
  min_len <- min(sapply(split_paths, length))
  common <- split_paths[[1]][1:min_len]
  for (i in seq_len(min_len)) {
    segs <- sapply(split_paths, function(x) x[i])
    if (length(unique(segs)) > 1) {
      if (i == 1) return("/")
      common <- common[1:(i - 1)]
      break
    }
  }
  root <- paste(common, collapse = "/")
  if (root == "") root <- "/"
  root
}

.relative_to_root <- function(paths, root) {
  norm_paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
  root_norm <- normalizePath(root, winslash = "/", mustWork = TRUE)
  root_norm <- sub("/$", "", root_norm)
  sub(paste0("^", root_norm, "/"), "", norm_paths)
}
