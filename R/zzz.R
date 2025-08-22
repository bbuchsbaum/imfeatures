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
#' @importFrom reticulate conda_create conda_install py_install virtualenv_create virtualenv_install use_condaenv use_virtualenv
install_imfeatures_python <- function(envname = "r-imfeatures",
                                      method = ifelse(.detect_conda_present(), "conda", "virtualenv"),
                                      python_version = "3.9",
                                      install_thingsvision = TRUE,
                                      install_resmem = TRUE,
                                      force_create = FALSE,
                                      conda_path = "auto") {

  message("Starting imfeatures Python setup using method: ", method)

  # --- Argument Checks ---
  method <- match.arg(method, c("conda", "virtualenv"))
  assert_scalar(envname, "character")
  assert_scalar(python_version, "character")
  assert_scalar(install_thingsvision, "logical")
  assert_scalar(install_resmem, "logical")
  assert_scalar(force_create, "logical")

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
    if (nzchar(custom_python)) {
      reticulate::use_python(custom_python, required = FALSE)
      message("Using custom Python: ", custom_python)
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
