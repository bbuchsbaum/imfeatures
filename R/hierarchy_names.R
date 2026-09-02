#' @keywords internal
.stage_output_name <- function(stage, index, recipe) {
  base <- if (nzchar(recipe$name_prefix)) {
    paste0(recipe$name_prefix, "_", stage)
  } else {
    stage
  }
  if (!is.null(recipe$decomposition) &&
    isTRUE(recipe$innovation_suffix) &&
    index > 1L) {
    paste0(base, "_innovation")
  } else {
    base
  }
}

#' @keywords internal
.component_colnames <- function(output_name, n) {
  if (n <= 0L) {
    return(character(0))
  }
  sprintf("%s::comp_%03d", output_name, seq_len(n))
}

#' @keywords internal
.channel_width <- function(n_channels) {
  max(3L, nchar(as.character(as.integer(n_channels))))
}

#' @keywords internal
.channel_token <- function(channel, n_channels) {
  sprintf("channel_%0*d", .channel_width(n_channels), as.integer(channel))
}

#' @keywords internal
.stage_feature_name <- function(stage, channel, suffix, n_channels) {
  paste(stage, .channel_token(channel, n_channels), suffix, sep = "::")
}
