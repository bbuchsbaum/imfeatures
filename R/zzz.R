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
    if (is.null(conda_exe) || !nzchar(conda_exe) || !file.exists(conda_exe)) {
      return(FALSE)
    }
    # Verify the conda executable actually runs (guards against broken shebangs)
    ok <- FALSE
    try({
      res <- suppressWarnings(system2(conda_exe, "--version", stdout = TRUE, stderr = TRUE))
      ok <- length(res) > 0
    }, silent = TRUE)
    return(isTRUE(ok))
  }, error = function(e) {
    # If conda_binary throws an error (e.g., conda not found), return FALSE
    return(FALSE)
  })
}
                           
#' @describeIn imfeatures_config Deprecated alias for imfeatures_config
#' @param ... Arguments passed to imfeatures_config
#' @export
install_imfeatures_python <- function(...) {
  warning("install_imfeatures_python() is deprecated; use imfeatures_config().", call. = FALSE)
  imfeatures_config(...)
}

.onLoad <- function(libname, pkgname) {
  # Check if user wants to skip Python setup (e.g., on HPC systems)
  skip_python <- Sys.getenv("IMFEATURES_SKIP_PYTHON", "FALSE")
  skip_python <- toupper(skip_python) %in% c("TRUE", "1", "YES")
  
  if (skip_python) {
    message("Python setup skipped (IMFEATURES_SKIP_PYTHON=TRUE). ",
            "Python-dependent features will not be available until manually configured.")
    return(invisible(NULL))
  }
  
  # Try to configure Python environment, but don't fail package loading
  tryCatch({
    # Check if user specified a custom Python path
    custom_python <- Sys.getenv("IMFEATURES_PYTHON_PATH", "")
    reticulate_python <- Sys.getenv("RETICULATE_PYTHON", "")
    if (nzchar(custom_python)) {
      reticulate::use_python(custom_python, required = FALSE)
      message("Using custom Python: ", custom_python)
    } else if (nzchar(reticulate_python)) {
      # Respect RETICULATE_PYTHON if provided (common on HPC)
      if (file.exists(reticulate_python)) {
        reticulate::use_python(reticulate_python, required = FALSE)
        message("Using RETICULATE_PYTHON: ", reticulate_python)
      } else {
        message("RETICULATE_PYTHON set but path not found: ", reticulate_python)
      }
    } else {
      # Only try auto-configuration if no custom path specified
      imfeatures_config()
    }
    
    # Try to import Python modules, but don't fail if unavailable
    if (reticulate::py_module_available("PIL")) {
      PIL <<- reticulate::import("PIL", delay_load = TRUE)
    }
    if (reticulate::py_module_available("resmem")) {
      resmem <<- reticulate::import("resmem", delay_load = TRUE)
    }
    if (reticulate::py_module_available("thingsvision")) {
      tv <<- reticulate::import("thingsvision", delay_load = TRUE)
      tv_data <<- reticulate::import("thingsvision.utils.data", delay_load = TRUE)
      tv_utils_storing <<- reticulate::import("thingsvision.utils.storing", delay_load = TRUE)
      tv_core_extraction <<- reticulate::import("thingsvision.core.extraction", delay_load = TRUE)
    }
  }, error = function(e) {
    message("Note: Python configuration failed during package loading. ",
            "This is normal on some systems (e.g., HPC). ",
            "Python features can be configured later using imfeatures_config() or use_existing_python().")
    message("To suppress this message, set environment variable: IMFEATURES_SKIP_PYTHON=TRUE")
  })
}
