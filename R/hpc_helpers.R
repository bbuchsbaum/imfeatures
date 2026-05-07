#' Configure Python for HPC systems
#'
#' Helper function specifically designed for HPC systems where Python
#' is loaded via module systems or custom installations.
#'
#' @param module_cmd Optional module load command to run (e.g., "module load python/3.9")
#' @param python_cmd Python command to use (default: "python3")
#' @param install_deps Whether to attempt installing missing dependencies
#' @param pip_user Whether to use --user flag for pip installs. Automatically
#'   disabled when targeting a virtualenv/venv, where --user is not allowed.
#'
#' @return Invisible path to configured Python
#'
#' @details
#' This function is tailored for HPC environments and will:
#' 1. Run the module load command if provided
#' 2. Find and configure the appropriate Python
#' 3. Check for required modules
#' 4. Optionally install missing dependencies with pip --user
#'
#' @examples
#' \dontrun{
#' # Load Python module and configure
#' configure_hpc_python("module load python/3.9")
#'
#' # Use specific Python without module load
#' configure_hpc_python(python_cmd = "/apps/python/3.9/bin/python3")
#'
#' # Auto-install missing dependencies
#' configure_hpc_python("module load python/3.9", install_deps = TRUE)
#' }
#'
#' @export
configure_hpc_python <- function(module_cmd = NULL,
                                 python_cmd = "python3",
                                 install_deps = FALSE,
                                 pip_user = TRUE) {
  # Run module load command if provided
  if (!is.null(module_cmd) && nzchar(module_cmd)) {
    message("Running: ", module_cmd)
    system(module_cmd)
  }

  # Find Python executable
  python_path <- Sys.which(python_cmd)
  if (!nzchar(python_path)) {
    # Try common HPC Python locations
    hpc_paths <- c(
      paste0("/apps/python/", python_cmd),
      paste0("/software/python/", python_cmd),
      paste0("/usr/local/bin/", python_cmd),
      paste0("/opt/python/", python_cmd)
    )
    python_path <- hpc_paths[file.exists(hpc_paths)][1]

    if (is.na(python_path)) {
      stop(
        "Python executable '", python_cmd, "' not found. ",
        "Please load the appropriate module or specify the full path."
      )
    }
  }

  # Configure Python
  message("Configuring Python: ", python_path)
  reticulate::use_python(python_path, required = TRUE)

  # Detect if this Python is a virtualenv/venv; in that case, --user installs are invalid
  in_venv <- FALSE
  in_venv <- tryCatch(
    {
      reticulate::py_eval(
        "import sys; (getattr(sys, 'real_prefix', None) is not None) or (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix)",
        convert = TRUE
      )
    },
    error = function(e) FALSE
  )
  if (!isTRUE(in_venv)) {
    venv_root <- normalizePath(file.path(dirname(python_path), ".."), mustWork = FALSE)
    in_venv <- file.exists(file.path(venv_root, "pyvenv.cfg"))
  }

  # Check modules
  required <- c("Pillow", "numpy")
  optional <- c("thingsvision", "torch", "torchvision", "resmem")

  missing_required <- character()
  missing_optional <- character()

  for (pkg in required) {
    mod_name <- if (pkg == "Pillow") "PIL" else pkg
    if (!reticulate::py_module_available(mod_name)) {
      missing_required <- c(missing_required, pkg)
    }
  }

  for (pkg in optional) {
    if (!reticulate::py_module_available(pkg)) {
      missing_optional <- c(missing_optional, pkg)
    }
  }

  # Detect Python version and provide guidance for known incompatibilities
  py_ver <- tryCatch(
    {
      reticulate::py_eval("import sys; f'{sys.version_info[0]}.{sys.version_info[1]}'", convert = TRUE)
    },
    error = function(e) NA_character_
  )
  if (is.character(py_ver) && nzchar(py_ver)) {
    parts <- strsplit(py_ver, "\\.")[[1]]
    if (length(parts) >= 2) {
      maj <- suppressWarnings(as.integer(parts[1]))
      min <- suppressWarnings(as.integer(parts[2]))
      if (!is.na(maj) && !is.na(min) && (maj > 3 || (maj == 3 && min >= 11))) {
        if ("thingsvision" %in% missing_optional) {
          message(
            "\nNote: Python ", py_ver, " detected. The 'thingsvision' package currently pins 'numba' to a version that ",
            "does not support Python 3.11+. Prefer Python 3.9 or 3.10 when installing 'thingsvision'."
          )
        }
      }
    }
  }

  # Report status
  if (length(missing_required) > 0) {
    message("\nMissing REQUIRED packages: ", paste(missing_required, collapse = ", "))

    if (install_deps) {
      message("Installing required packages...")
      use_user <- isTRUE(pip_user) && !isTRUE(in_venv)
      pip_cmd <- paste0(
        python_path, " -m pip install ",
        if (use_user) "--user " else "",
        paste(missing_required, collapse = " ")
      )
      message("Running: ", pip_cmd)
      system(pip_cmd)
    } else {
      message("To install, run:")
      use_user <- isTRUE(pip_user) && !isTRUE(in_venv)
      message(
        "  ", python_path, " -m pip install ",
        if (use_user) "--user " else "",
        paste(missing_required, collapse = " ")
      )
    }
  } else {
    message("[OK] All required packages are installed")
  }

  if (length(missing_optional) > 0) {
    message("\nOptional packages not found: ", paste(missing_optional, collapse = ", "))
    if (install_deps && length(missing_required) == 0) {
      message("To install optional packages, run:")
      use_user <- isTRUE(pip_user) && !isTRUE(in_venv)
      message(
        "  ", python_path, " -m pip install ",
        if (use_user) "--user " else "",
        paste(missing_optional, collapse = " ")
      )
    }
  }

  # Import modules
  tryCatch(
    {
      pkg_env <- asNamespace("imfeatures")
      if (reticulate::py_module_available("PIL")) {
        tryCatch(
          {
            unlockBinding("PIL", pkg_env)
            assign("PIL", reticulate::import("PIL", delay_load = TRUE), envir = pkg_env)
            lockBinding("PIL", pkg_env)
          },
          error = function(e) invisible(NULL)
        )
      }
      if (reticulate::py_module_available("resmem")) {
        tryCatch(
          {
            unlockBinding("resmem", pkg_env)
            assign("resmem", reticulate::import("resmem", delay_load = TRUE), envir = pkg_env)
            lockBinding("resmem", pkg_env)
          },
          error = function(e) invisible(NULL)
        )
      }
      if (reticulate::py_module_available("thingsvision")) {
        # Unlock and update bindings
        tryCatch(
          {
            unlockBinding("tv", pkg_env)
            assign("tv", reticulate::import("thingsvision", delay_load = TRUE), envir = pkg_env)
            lockBinding("tv", pkg_env)

            unlockBinding("tv_data", pkg_env)
            assign("tv_data", reticulate::import("thingsvision.utils.data", delay_load = TRUE), envir = pkg_env)
            lockBinding("tv_data", pkg_env)

            unlockBinding("tv_utils_storing", pkg_env)
            assign("tv_utils_storing", reticulate::import("thingsvision.utils.storing", delay_load = TRUE), envir = pkg_env)
            lockBinding("tv_utils_storing", pkg_env)

            unlockBinding("tv_core_extraction", pkg_env)
            assign("tv_core_extraction", reticulate::import("thingsvision.core.extraction", delay_load = TRUE), envir = pkg_env)
            lockBinding("tv_core_extraction", pkg_env)
          },
          error = function(e) {
            # Ignore binding errors
          }
        )
      }
      message("\n[OK] Python configured successfully for imfeatures")
    },
    error = function(e) {
      message("\nNote: Some modules could not be imported: ", e$message)
    }
  )

  invisible(python_path)
}

#' Install Python dependencies for imfeatures
#'
#' Convenience function to install required and optional Python packages
#' for imfeatures using pip. If the target `python_cmd` belongs to a
#' virtualenv/venv, the `--user` flag is automatically disabled as it is
#' not supported inside virtual environments.
#'
#' @param python_cmd Python command or path (default: "python3")
#' @param user Whether to use --user flag (default: TRUE for HPC). Ignored when
#'   `python_cmd` is a virtualenv/venv.
#' @param optional Whether to install optional packages (default: FALSE)
#' @param upgrade Whether to upgrade existing packages (default: FALSE)
#'
#' @examples
#' \dontrun{
#' # Install only required packages
#' install_python_deps()
#'
#' # Install all packages including optional
#' install_python_deps(optional = TRUE)
#'
#' # Use specific Python
#' install_python_deps(python_cmd = "/apps/python/3.9/bin/python3")
#' }
#'
#' @export
install_python_deps <- function(python_cmd = "python3",
                                user = TRUE,
                                optional = FALSE,
                                upgrade = FALSE) {
  # Find Python
  if (!file.exists(python_cmd)) {
    python_cmd <- Sys.which(python_cmd)
    if (!nzchar(python_cmd)) {
      stop("Python executable not found. Please specify the full path or ensure Python is in PATH.")
    }
  }

  message("Using Python: ", python_cmd)

  # Detect if this Python is a virtualenv/venv; in that case, --user installs are invalid
  in_venv <- FALSE
  venv_root <- normalizePath(file.path(dirname(python_cmd), ".."), mustWork = FALSE)
  if (file.exists(file.path(venv_root, "pyvenv.cfg"))) in_venv <- TRUE

  # Build package list
  packages <- c("Pillow", "numpy")
  if (optional) {
    # Add all optional packages including thingsvision dependencies
    packages <- c(
      packages,
      "thingsvision",
      "torch",
      "torchvision",
      "tensorflow", # or tensorflow-cpu if available
      "torchtyping",
      "scipy",
      "scikit-learn",
      "pandas",
      "matplotlib",
      "h5py",
      "open-clip-torch",
      "resmem"
    )
  }

  # Build pip command
  pip_args <- character()
  use_user <- isTRUE(user) && !isTRUE(in_venv)
  if (use_user) pip_args <- c(pip_args, "--user")
  if (upgrade) pip_args <- c(pip_args, "--upgrade")

  pip_cmd <- paste(
    python_cmd, "-m pip install",
    paste(pip_args, collapse = " "),
    paste(packages, collapse = " ")
  )

  message("Installing packages with:")
  message("  ", pip_cmd)

  result <- system(pip_cmd)

  if (result == 0) {
    message("\n[OK] Installation completed successfully")
    message("You can now use: library(imfeatures); use_existing_python()")
  } else {
    warning("Installation may have failed. Please check the output above.")
  }

  invisible(result)
}
