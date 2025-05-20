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

#' Install Python dependencies for imfeatures
#'
#' This function creates a dedicated Python environment (using Conda or venv)
#' named "r-imfeatures" and installs the necessary Python packages required
#' by the imfeatures R package, namely Pillow, and optionally thingsvision and resmem.
#'
#' @param envname The name for the Python environment. Defaults to "r-imfeatures".
#'        If using venv, this can also be a path.
#' @param method The method to create the environment, either "conda" or "virtualenv".
#'        Defaults to "conda" if available, otherwise "virtualenv".
#' @param python_version The Python version to install in the environment (e.g., "3.9").
#'        Defaults to "3.9". Ensure this version is compatible with dependencies.
#' @param install_thingsvision Logical. Install the 'thingsvision' library? Defaults to TRUE.
#' @param install_resmem Logical. Install the 'resmem' library? Defaults to TRUE.
#' @param force_create Logical. If TRUE, an existing environment with the same name
#'        will be removed before creating the new one. Use with caution! Defaults to FALSE.
#' @param conda_path Path to the conda executable. Defaults to reticulate's auto-detection.
#'
#' @details
#' This function requires either Miniconda/Anaconda (for method="conda") or a
#' standard Python installation (for method="virtualenv") to be available.
#' Installation can take a significant amount of time depending on the packages selected.
#'
#' After successful installation, it's recommended to restart your R session.
#' The package will attempt to automatically configure reticulate to use the
#' "r-imfeatures" environment upon loading.
#'
#' @return Invisibly returns the path or name of the created environment.
#' @export
#' @import reticulate
install_imfeatures_python <- function(envname = "r-imfeatures",
                                      method = ifelse(.detect_conda_present(), "conda", "virtualenv"),
                                      python_version = "3.9",
                                      install_thingsvision = TRUE,
                                      install_resmem = TRUE,
                                      force_create = FALSE,
                                      conda_path = "auto") {

  message("Starting imfeatures Python setup using method: ", method)

  # --- Argument Checks ---
  stopifnot(method %in% c("conda", "virtualenv"))
  stopifnot(is.logical(install_thingsvision), is.logical(install_resmem), is.logical(force_create))
  stopifnot(is.character(envname), length(envname) == 1)
  stopifnot(is.character(python_version), length(python_version) == 1)

  # --- Availability Checks ---
  if (method == "conda" && !.detect_conda_present(conda = conda_path)) {
    stop("Conda installation not found or specified conda_path ('", conda_path, "') is invalid. ",
         "Please install Miniconda/Anaconda or specify the correct path.")
  }
  if (method == "virtualenv") {
      py_bin <- tryCatch(reticulate::py_exe(), error = function(e) NULL) # Use reticulate's finding mechanism
      if (is.null(py_bin) || !nzchar(py_bin)) {
         # Try system python as fallback
         python_create_bin <- Sys.which("python3")
         if (!nzchar(python_create_bin)) python_create_bin <- Sys.which("python")
         if (!nzchar(python_create_bin)) {
            stop("Python executable not found by reticulate or system path. Cannot create virtualenv. Please ensure Python is installed and accessible.")
         }
         message("Using system Python for virtualenv creation: ", python_create_bin)
      } else {
         message("Using reticulate's detected Python for virtualenv: ", py_bin)
      }
   }


  # --- Handle Existing Environment ---
  env_exists <- FALSE
  env_path <- NULL # Store path if found

  if (method == "conda") {
    conda_envs <- tryCatch(conda_list(conda = conda_path), error = function(e) NULL)
    if (!is.null(conda_envs) && envname %in% conda_envs$name) {
      env_exists <- TRUE
      env_path <- conda_envs$python[conda_envs$name == envname]
      message("Conda environment '", envname, "' already exists at ", env_path)
    }
  } else { # virtualenv
    potential_path <- if (grepl(.Platform$file.sep, envname, fixed=TRUE)) {
        envname # Treat as full path
    } else {
        file.path(reticulate::virtualenv_root(), envname) # Default location
    }
    py_suffix <- ifelse(.Platform$OS.type == "windows", "python.exe", "python")
    py_loc <- file.path(potential_path, ifelse(.Platform$OS.type == "windows", "Scripts", "bin"), py_suffix)

    if (dir.exists(potential_path) && file.exists(py_loc)) { # Check dir exists too
       env_exists <- TRUE
       env_path <- potential_path # Store the validated path
       message("Virtualenv '", envname, "' already exists at '", env_path, "'.")
    }
  }

  created_now <- FALSE # Flag if we are creating it in this run

  if (env_exists) {
    if (force_create) {
      warning("Existing environment '", envname, "' (located at '", env_path, "') will be removed because force_create = TRUE.")
      tryCatch({
         if (method == "conda") {
           conda_remove(envname = envname, conda = conda_path)
         } else {
           virtualenv_remove(envname = env_path, confirm = FALSE)
         }
         env_exists <- FALSE # Proceed with creation
         message("Existing environment removed.")
      }, error = function(e){
         stop("Failed to remove existing environment '", envname, "':\n", e$message)
      })
    } else {
      message("Using existing environment '", envname, "'. ",
              "To recreate, set force_create = TRUE or remove it manually. ",
              "Will attempt to install missing requested packages.")
      # Continue to install step even if env exists without force_create
    }
  }

  # --- Create Environment if it doesn't exist ---
  if (!env_exists) {
     message("Creating Python environment '", envname, "' with Python ", python_version, "...")
     tryCatch({
       if (method == "conda") {
         conda_create(envname, python_version = python_version, conda = conda_path)
       } else {
         # Determine python binary to use for creation
         python_create_bin <- reticulate::py_exe()
         if (is.null(python_create_bin) || !nzchar(python_create_bin)){
            python_create_bin <- Sys.which("python3")
            if (!nzchar(python_create_bin)) python_create_bin <- Sys.which("python")
         }
         if (is.null(python_create_bin) || !nzchar(python_create_bin)) {
            stop("Could not find python executable to create virtualenv.")
         }
         message("Creating venv using: ", python_create_bin)
         # Note: virtualenv_create itself doesn't take 'version', relies on the 'python' binary's version.
         # We might need a check here if python_version doesn't match python_create_bin
         virtualenv_create(envname = envname, python = python_create_bin)
       }
       message("Environment '", envname, "' created successfully.")
       created_now <- TRUE
     }, error = function(e) {
       stop("Failed to create environment '", envname, "':\n", e$message)
     })
  }

  # --- Install Packages ---
  # Use the created/existing environment for installation
  install_target_env <- envname # For conda/venv name
  if(method == "virtualenv" && !is.null(env_path)) {
      install_target_env <- env_path # Use path for venv installs if known
  }

  message("Installing/updating Python packages into '", envname, "'. This may take a while...")

  # Core dependencies
  packages_to_install <- c("pip", "Pillow") # Pillow provides PIL

  # Optional dependencies
  if (install_thingsvision) {
     # Note: thingsvision might have complex dependencies (torch, etc.)
     # Consider adding specific channel recommendations for conda if needed
     packages_to_install <- c(packages_to_install, "thingsvision")
     # Add optional CLIP/DreamSim back here if desired, maybe as separate args
  }
  if (install_resmem) {
     # Assuming resmem is available via pip
     packages_to_install <- c(packages_to_install, "resmem")
  }

  # Remove duplicates just in case
  packages_to_install <- unique(packages_to_install)

  message("Attempting to install: ", paste(packages_to_install, collapse=", "))

  tryCatch({
    if (method == "conda") {
      # Install using conda where possible, fall back to pip for others
      # This needs refinement - determine which packages are best from conda vs pip
      # Simple approach: Install all via pip within conda env for now
      conda_install(envname = install_target_env,
                    packages = packages_to_install,
                    pip = TRUE,
                    conda = conda_path,
                    pip_options = "--upgrade") # Use upgrade to get latest/install missing
    } else { # virtualenv
      virtualenv_install(envname = install_target_env,
                         packages = packages_to_install,
                         ignore_installed = FALSE, # Don't ignore if already there, try upgrading
                         pip_options = "--upgrade")
    }
    message("Required Python packages installed/updated successfully in '", envname, "'.")
  }, error = function(e) {
    warning("An error occurred during package installation in '", envname, "':\n", e$message,
            "\nThe environment exists, but package installation may be incomplete.",
            "\nYou may need to activate the environment manually and run pip/conda install commands yourself.")
    # Don't stop, let user try to fix manually
  })

  message("\nInstallation/Update process complete!")
  message("Restart your R session to ensure the package loads with the correct environment.")
  message("The 'imfeatures' package will attempt to use the '", envname, "' environment automatically.")

  return(invisible(envname)) # Return the name/path used
}

# Also update the .onLoad check if using it
.onLoad <- function(libname, pkgname) {
  # Define the default environment name
  .imf_env_name <- "r-imfeatures"
  env_configured <- FALSE
  env_path_msg <- .imf_env_name # For messages
  final_status_msg <- ""

  # Determine expected paths first
  expected_conda_path <- tryCatch(reticulate::conda_python(.imf_env_name, conda = "auto"), error = function(e) NULL)
  expected_venv_path_base <- tryCatch(reticulate:::virtualenv_path(.imf_env_name), error = function(e) NULL)
  expected_venv_path <- NULL
  if (!is.null(expected_venv_path_base)) {
      py_suffix <- ifelse(.Platform$OS.type == "windows", "python.exe", "python")
      py_bin <- ifelse(.Platform$OS.type == "windows", "Scripts", "bin")
      expected_venv_path <- file.path(expected_venv_path_base, py_bin, py_suffix)
      # Double check venv path actually exists
      if (!file.exists(expected_venv_path)) expected_venv_path <- NULL
  }
  if (!is.null(expected_conda_path) && !file.exists(expected_conda_path)) expected_conda_path <- NULL

  # 1. Check RETICULATE_PYTHON environment variable
  reticulate_python_env <- Sys.getenv("RETICULATE_PYTHON")
  if (nzchar(reticulate_python_env)) {
      is_expected_env <- FALSE
      if (!is.null(expected_conda_path) && reticulate_python_env == expected_conda_path) is_expected_env <- TRUE
      if (!is_expected_env && !is.null(expected_venv_path) && reticulate_python_env == expected_venv_path) is_expected_env <- TRUE

      if (is_expected_env) {
          final_status_msg <- paste("Reticulate already configured via RETICULATE_PYTHON to use:", reticulate_python_env)
          env_configured <- TRUE
          env_path_msg <- reticulate_python_env
      } else {
          final_status_msg <- paste(
              "RETICULATE_PYTHON is set to:", reticulate_python_env, "\n",
              "This does not match the expected 'r-imfeatures' environment path.\n",
              "imfeatures will not attempt automatic configuration."
          )
          # Do not proceed with auto-detection if RETICULATE_PYTHON is set but wrong
      }
  }

  # 2. If RETICULATE_PYTHON didn't configure it, check if reticulate is already initialized
  if (!env_configured && !nzchar(Sys.getenv("RETICULATE_PYTHON"))) { # Only proceed if RETICULATE_PYTHON was not set
      if (reticulate::py_available(initialize = FALSE)) {
          # Reticulate is already initialized - check if it's the right one
          current_config <- reticulate::py_config()
          current_python <- current_config$python

          is_expected_env <- FALSE
          if (!is.null(expected_conda_path) && current_python == expected_conda_path) is_expected_env <- TRUE
          if (!is_expected_env && !is.null(expected_venv_path) && current_python == expected_venv_path) is_expected_env <- TRUE

          if (is_expected_env) {
              final_status_msg <- paste("Reticulate already initialized with the expected 'r-imfeatures' environment:", current_python)
              env_configured <- TRUE
              env_path_msg <- current_python
          } else {
              final_status_msg <- paste(
                  "Reticulate is already initialized with a different Python environment:\n",
                  current_python, "\n",
                  "Cannot automatically switch to 'r-imfeatures'.\n",
                  "Please restart R and ensure 'imfeatures' is loaded before other packages using reticulate,",
                  "or manually use reticulate::use_python('[path_to_r-imfeatures_python]') in a clean session."
              )
              env_configured <- FALSE # Explicitly mark as not configured for imfeatures
          }
      } else {
          # 3. Reticulate is not initialized - try to initialize it with r-imfeatures using use_python()
          initialized_successfully <- FALSE
          # Try Conda path first if available
          if (!is.null(expected_conda_path)) {
              tryCatch({
                  reticulate::use_python(expected_conda_path, required = TRUE)
                  final_status_msg <- paste("Successfully configured reticulate to use Conda environment 'r-imfeatures':", expected_conda_path)
                  env_configured <- TRUE
                  env_path_msg <- expected_conda_path
                  initialized_successfully <- TRUE
              }, error = function(e) {
                  warning("Attempted to initialize reticulate with conda 'r-imfeatures' (", expected_conda_path, ") but failed: ", e$message)
              })
          }

          # If Conda failed or wasn't found, try Venv path if available
          if (!initialized_successfully && !is.null(expected_venv_path)) {
               tryCatch({
                  reticulate::use_python(expected_venv_path, required = TRUE)
                  final_status_msg <- paste("Successfully configured reticulate to use virtualenv 'r-imfeatures':", expected_venv_path)
                  env_configured <- TRUE
                  env_path_msg <- expected_venv_path
                  initialized_successfully <- TRUE
               }, error = function(e) {
                  warning("Attempted to initialize reticulate with virtualenv 'r-imfeatures' (", expected_venv_path, ") but failed: ", e$message)
               })
          }

          if (!initialized_successfully && final_status_msg == "") { # If neither worked and no message set yet
              final_status_msg <- paste("Default '", .imf_env_name, "' Python environment (conda or venv) not found or inaccessible.")
          }
      }
  }

  # Print the final status message determined above
  if (nzchar(final_status_msg)) {
      packageStartupMessage(final_status_msg)
  }

  # Attempt to import modules only if environment configuration seems successful
  if (env_configured) {
    import_error <- NULL
    tryCatch({
      PIL <<- reticulate::import("PIL", delay_load = TRUE)
      resmem <<- reticulate::import("resmem", delay_load = TRUE)
      tv <<- reticulate::import("thingsvision", delay_load = TRUE)
      packageStartupMessage("imfeatures: PIL, resmem, thingsvision modules queued for delayed loading from: ", env_path_msg)
    }, error = function(e) {
      import_error <<- e
    })

    if (!is.null(import_error)) {
       warning("Python environment ('", env_path_msg, "') configured, but failed to import required modules.\n",
              "Error: ", import_error$message, "\nCheck package installation (Pillow, resmem, thingsvision) in this environment.\n",
              "Python features may fail. Consider running install_imfeatures_python() and restarting R.")
       PIL <<- NULL; resmem <<- NULL; tv <<- NULL; tv_data <<- NULL; tv_utils_storing <<- NULL; tv_core_extraction <<- NULL
    }
  } else {
    # Environment not configured - print guidance if not already printed
    if (!nzchar(final_status_msg) || (!grepl("Cannot automatically switch", final_status_msg) && !grepl("not found or inaccessible", final_status_msg))) { # Avoid redundant message
        packageStartupMessage("Python features of 'imfeatures' require the '", .imf_env_name, "' environment.")
        packageStartupMessage("Please run install_imfeatures_python() to create/update it, then restart R.")
    }
    # Assign NULL placeholders
    PIL <<- NULL; resmem <<- NULL; tv <<- NULL; tv_data <<- NULL; tv_utils_storing <<- NULL; tv_core_extraction <<- NULL
  }
}