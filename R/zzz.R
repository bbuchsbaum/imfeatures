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
  tryCatch(
    {
      conda_exe <- reticulate::conda_binary(conda = conda)
      if (is.null(conda_exe) || !nzchar(conda_exe) || !file.exists(conda_exe)) {
        return(FALSE)
      }
      # Verify the conda executable actually runs (guards against broken shebangs)
      ok <- FALSE
      try(
        {
          res <- suppressWarnings(system2(conda_exe, "--version", stdout = TRUE, stderr = TRUE))
          ok <- length(res) > 0
        },
        silent = TRUE
      )
      return(isTRUE(ok))
    },
    error = function(e) {
      # If conda_binary throws an error (e.g., conda not found), return FALSE
      return(FALSE)
    }
  )
}

#' @describeIn imfeatures_config Deprecated alias for imfeatures_config
#' @param ... Arguments passed to imfeatures_config
#' @export
install_imfeatures_python <- function(...) {
  warning("install_imfeatures_python() is deprecated; use imfeatures_config().", call. = FALSE)
  imfeatures_config(...)
}

.onLoad <- function(libname, pkgname) {
  skip_python <- Sys.getenv("IMFEATURES_SKIP_PYTHON", "FALSE")
  skip_python <- toupper(skip_python) %in% c("TRUE", "1", "YES")

  if (skip_python) {
    return(invisible(NULL))
  }

  pkg_env <- topenv(parent.frame())

  tryCatch(
    {
      custom_python <- Sys.getenv("IMFEATURES_PYTHON_PATH", "")
      reticulate_python <- Sys.getenv("RETICULATE_PYTHON", "")
      if (nzchar(custom_python)) {
        reticulate::use_python(custom_python, required = FALSE)
      } else if (nzchar(reticulate_python)) {
        if (file.exists(reticulate_python)) {
          reticulate::use_python(reticulate_python, required = FALSE)
        }
      } else {
        imfeatures_config()
      }

      if (reticulate::py_module_available("PIL")) {
        assign("PIL", reticulate::import("PIL", delay_load = TRUE), envir = pkg_env)
      }
      if (reticulate::py_module_available("resmem")) {
        assign("resmem", reticulate::import("resmem", delay_load = TRUE), envir = pkg_env)
      }
      if (reticulate::py_module_available("thingsvision")) {
        assign("tv", reticulate::import("thingsvision", delay_load = TRUE), envir = pkg_env)
        assign("tv_data", reticulate::import("thingsvision.utils.data", delay_load = TRUE), envir = pkg_env)
        assign("tv_utils_storing", reticulate::import("thingsvision.utils.storing", delay_load = TRUE), envir = pkg_env)
        assign("tv_core_extraction", reticulate::import("thingsvision.core.extraction", delay_load = TRUE), envir = pkg_env)
      }
    },
    error = function(e) {
      invisible(NULL)
    }
  )
}

.onAttach <- function(libname, pkgname) {
  skip_python <- Sys.getenv("IMFEATURES_SKIP_PYTHON", "FALSE")
  skip_python <- toupper(skip_python) %in% c("TRUE", "1", "YES")
  if (skip_python) {
    packageStartupMessage(
      "Python setup skipped (IMFEATURES_SKIP_PYTHON=TRUE). ",
      "Python-dependent features will not be available until manually configured."
    )
  }
}
