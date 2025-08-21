#' Configure the Python environment for imfeatures
#'
#' Sets up or activates the Python environment required by the package.
#' The environment is detected once per R session and cached.  Users can
#' force a reinstall with `reset = TRUE`.
#'
#' @param envname Name of the environment to create/use. Defaults to
#'   "r-imfeatures".
#' @param method Installation method. Options: "auto", "conda", "virtualenv", 
#'   or "existing" (to use current Python without modification).
#' @param reset Force reinstallation of the environment.
#' @param create Whether to create the environment if it doesn't exist.
#'   Set to FALSE on HPC systems where you want to use a pre-configured environment.
#' @return Invisible path to the configured Python binary.
#' @export
imfeatures_config <- local({
  cached <- NULL
  function(envname = "r-imfeatures",
           method = c("auto", "conda", "virtualenv", "existing"),
           reset = FALSE,
           create = TRUE) {
    if (!is.null(cached) && !reset) {
      cli_msg <- crayon::green(paste0("Using cached Python env: ", cached))
      message(cli_msg)
      return(invisible(cached))
    }
    method <- match.arg(method)
    
    # Handle existing Python environment
    if (method == "existing") {
      # Use current Python configuration without modification
      cfg <- reticulate::py_config()$python
      if (is.null(cfg) || !file.exists(cfg)) {
        stop("No existing Python configuration found. Please set up Python first or use a different method.")
      }
      cached <<- cfg
      message(crayon::green(paste0("Using existing Python: ", cfg)))
      return(invisible(cached))
    }
    
    if (method == "auto") {
      method <- if (.detect_conda_present()) "conda" else "virtualenv"
    }
    if (reset) {
      if (method == "conda") {
        try(reticulate::conda_remove(envname, conda = "auto"), silent = TRUE)
      } else {
        try(reticulate::virtualenv_remove(envname, confirm = FALSE), silent = TRUE)
      }
    }
    exists <- FALSE
    if (method == "conda") {
      envs <- try(reticulate::conda_list()$name, silent = TRUE)
      exists <- !inherits(envs, "try-error") && envname %in% envs
    } else {
      root <- reticulate::virtualenv_root()
      exists <- dir.exists(file.path(root, envname))
    }
    if (!exists && create) {
      message(crayon::yellow(paste0("Creating Python env '", envname, "'")))
      req_file <- system.file("requirements.txt", package = "imfeatures")
      if (!file.exists(req_file)) {
        # If requirements.txt doesn't exist, install basic packages
        packages <- c("Pillow", "numpy")
        reticulate::py_install(envname = envname,
                               method = method,
                               packages = packages)
      } else {
        reticulate::py_install(envname = envname,
                               method = method,
                               packages = character(),
                               requirements = req_file)
      }
    } else if (!exists && !create) {
      stop("Environment '", envname, "' does not exist and create = FALSE")
    }
    
    tryCatch({
      if (method == "conda") {
        reticulate::use_condaenv(envname, required = TRUE)
        cfg <- reticulate::conda_python(envname)
      } else {
        reticulate::use_virtualenv(envname, required = TRUE)
        cfg <- reticulate::virtualenv_python(envname)
      }
      cached <<- cfg
    }, error = function(e) {
      stop("Failed to configure Python environment: ", e$message)
    })
    
    # Check backend availability
    backend <- "CPU"
    if (reticulate::py_module_available("torch")) {
      tryCatch({
        torch <- reticulate::import("torch")
        if (torch$cuda$is_available()) {
          device <- try(torch$cuda$get_device_name(0L), silent = TRUE)
          backend <- paste0("CUDA (", ifelse(inherits(device, "try-error"), "GPU", device), ")")
        }
      }, error = function(e) {
        # Torch import might fail, stay with CPU
      })
    }
    msg <- paste0("Backend: ", backend)
    if (backend == "CPU") message(crayon::blue(msg)) else message(crayon::green(msg))
    invisible(cached)
  }
})

#' Use existing Python installation for imfeatures
#'
#' Configures imfeatures to use an existing Python installation without
#' attempting to create or modify environments. This is particularly useful
#' on HPC systems where Python environments are pre-configured.
#'
#' @param python_path Optional path to Python executable. If NULL, uses the
#'   current reticulate Python configuration.
#' @param check_modules Whether to check for required Python modules and
#'   provide informative messages about missing dependencies.
#' @param force If TRUE, prioritize system Python over any existing virtualenvs.
#'   Useful on HPC systems where module-loaded Python should be used.
#'
#' @return Invisible path to the configured Python binary.
#'
#' @details
#' This function is designed for HPC and other restricted environments where:
#' - Conda/virtualenv creation may not be allowed
#' - Python modules are installed via module load systems
#' - Custom Python paths need to be specified
#'
#' @examples
#' \dontrun{
#' # On HPC after loading Python module
#' module load python/3.9
#' use_existing_python()
#'
#' # With specific Python path
#' use_existing_python("/usr/local/bin/python3")
#'
#' # Skip module checking
#' use_existing_python(check_modules = FALSE)
#' }
#'
#' @export
use_existing_python <- function(python_path = NULL, check_modules = TRUE, force = FALSE) {
  if (!is.null(python_path)) {
    if (!file.exists(python_path)) {
      stop("Python executable not found at: ", python_path)
    }
    reticulate::use_python(python_path, required = TRUE)
    message("Configured to use Python at: ", python_path)
  } else {
    # Try to get current Python configuration, handling errors gracefully
    cfg <- tryCatch({
      # First check if RETICULATE_PYTHON is set
      env_python <- Sys.getenv("RETICULATE_PYTHON", "")
      if (nzchar(env_python) && file.exists(env_python)) {
        reticulate::use_python(env_python, required = TRUE)
        return(list(python = env_python))
      }
      
      # If force=TRUE or no Python configured yet, prioritize system Python
      if (force || !reticulate::py_available()) {
        # Look for system Python first (module-loaded or system-wide)
        python_candidates <- c(
          Sys.which("python3"),
          Sys.which("python"),
          "/usr/bin/python3",
          "/usr/bin/python",
          "/usr/local/bin/python3",
          "/usr/local/bin/python"
        )
        python_candidates <- python_candidates[nzchar(python_candidates) & file.exists(python_candidates)]
        
        if (length(python_candidates) > 0) {
          reticulate::use_python(python_candidates[1], required = TRUE)
          return(list(python = python_candidates[1]))
        }
      }
      
      # Try py_config, but suppress broken virtualenv errors
      suppressWarnings({
        reticulate::py_config()
      })
    }, error = function(e) {
      # If py_config fails, try to find Python on the system
      python_candidates <- c(
        Sys.which("python3"),
        Sys.which("python"),
        "/usr/bin/python3",
        "/usr/bin/python",
        "/usr/local/bin/python3",
        "/usr/local/bin/python"
      )
      python_candidates <- python_candidates[nzchar(python_candidates) & file.exists(python_candidates)]
      
      if (length(python_candidates) > 0) {
        reticulate::use_python(python_candidates[1], required = TRUE)
        list(python = python_candidates[1])
      } else {
        NULL
      }
    })
    
    if (is.null(cfg) || is.null(cfg$python)) {
      stop("No Python configuration found. Please specify python_path or ensure Python is available in PATH.")
    }
    message("Using existing Python configuration: ", cfg$python)
  }
  
  if (check_modules) {
    # Check for required and optional modules
    required_modules <- c("PIL" = "Pillow", "numpy" = "numpy")
    optional_modules <- c("thingsvision" = "thingsvision", 
                         "torch" = "torch",
                         "torchvision" = "torchvision",
                         "resmem" = "resmem")
    
    message("\nChecking Python modules:")
    
    # Check required modules
    missing_required <- character()
    for (mod in names(required_modules)) {
      if (reticulate::py_module_available(mod)) {
        message("  ✓ ", mod, " (", required_modules[mod], ")")
      } else {
        message("  ✗ ", mod, " (", required_modules[mod], ") - REQUIRED")
        missing_required <- c(missing_required, required_modules[mod])
      }
    }
    
    # Check optional modules
    missing_optional <- character()
    for (mod in names(optional_modules)) {
      if (reticulate::py_module_available(mod)) {
        message("  ✓ ", mod, " (", optional_modules[mod], ")")
      } else {
        message("  ○ ", mod, " (", optional_modules[mod], ") - optional")
        missing_optional <- c(missing_optional, optional_modules[mod])
      }
    }
    
    if (length(missing_required) > 0) {
      message("\nMissing REQUIRED modules. Install with:")
      message("  pip install ", paste(missing_required, collapse = " "))
      warning("Some core features may not work without required modules.")
    }
    
    if (length(missing_optional) > 0) {
      message("\nOptional modules not found. For full functionality, install with:")
      message("  pip install ", paste(missing_optional, collapse = " "))
    }
  }
  
  # Import available modules
  tryCatch({
    if (reticulate::py_module_available("PIL")) {
      assign("PIL", reticulate::import("PIL", delay_load = TRUE), envir = parent.frame())
    }
    if (reticulate::py_module_available("resmem")) {
      assign("resmem", reticulate::import("resmem", delay_load = TRUE), envir = parent.frame())
    }
    if (reticulate::py_module_available("thingsvision")) {
      assign("tv", reticulate::import("thingsvision", delay_load = TRUE), envir = parent.frame())
      assign("tv_data", reticulate::import("thingsvision.utils.data", delay_load = TRUE), envir = parent.frame())
      assign("tv_utils_storing", reticulate::import("thingsvision.utils.storing", delay_load = TRUE), envir = parent.frame())
      assign("tv_core_extraction", reticulate::import("thingsvision.core.extraction", delay_load = TRUE), envir = parent.frame())
    }
  }, error = function(e) {
    message("Note: Some Python modules could not be imported: ", e$message)
  })
  
  invisible(reticulate::py_config()$python)
}
