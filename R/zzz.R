resmem <- NULL
PIL <- NULL
# resmodel is not explicitly handled by install/load, remove for now or handle separately if needed
# resmodel <- NULL
tv <- NULL
tv_data <- NULL # Keep these sub-module placeholders for now
tv_utils_storing <- NULL
tv_core_extraction <- NULL
# Potentially tv_core_rsa, tv_core_cka, tv_utils_alignment later


# Helper function (can be internal, not exported using #')
# Checks if reticulate can find a conda executable.
.detect_conda_present <- function(conda = "auto") {
  tryCatch({
    conda_exe <- reticulate::conda_binary(conda = conda)
    # Check if a path was returned and if that path actually exists
    # conda_binary itself might error if conda='auto' and none is found.
    return(!is.null(conda_exe) && nzchar(conda_exe) && file.exists(conda_exe))
  }, error = function(e) {
    # If conda_binary throws an error (e.g., conda not found), return FALSE
    return(FALSE)
  })
}
#' @describeIn imfeatures_config Deprecated alias.
#' @export
install_imfeatures_python <- function(...) {
  warning("install_imfeatures_python() is deprecated; use imfeatures_config().", call. = FALSE)
  imfeatures_config(...)
}

.onLoad <- function(libname, pkgname) {
  imfeatures_config()
  if (reticulate::py_module_available("PIL")) PIL <<- reticulate::import("PIL", delay_load = TRUE)
  if (reticulate::py_module_available("resmem")) resmem <<- reticulate::import("resmem", delay_load = TRUE)
  if (reticulate::py_module_available("thingsvision")) {
    tv <<- reticulate::import("thingsvision", delay_load = TRUE)
    tv_data <<- reticulate::import("thingsvision.utils.data", delay_load = TRUE)
    tv_utils_storing <<- reticulate::import("thingsvision.utils.storing", delay_load = TRUE)
    tv_core_extraction <<- reticulate::import("thingsvision.core.extraction", delay_load = TRUE)
  }
}
