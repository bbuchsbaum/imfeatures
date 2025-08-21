#' Configure Python for HPC systems
#'
#' Helper function specifically designed for HPC systems where Python
#' is loaded via module systems or custom installations.
#'
#' @param module_cmd Optional module load command to run (e.g., "module load python/3.9")
#' @param python_cmd Python command to use (default: "python3")
#' @param install_deps Whether to attempt installing missing dependencies
#' @param pip_user Whether to use --user flag for pip installs
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
      stop("Python executable '", python_cmd, "' not found. ",
           "Please load the appropriate module or specify the full path.")
    }
  }
  
  # Configure Python
  message("Configuring Python: ", python_path)
  reticulate::use_python(python_path, required = TRUE)
  
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
  
  # Report status
  if (length(missing_required) > 0) {
    message("\nMissing REQUIRED packages: ", paste(missing_required, collapse = ", "))
    
    if (install_deps) {
      message("Installing required packages...")
      pip_cmd <- paste0(python_path, " -m pip install ",
                       if (pip_user) "--user " else "",
                       paste(missing_required, collapse = " "))
      message("Running: ", pip_cmd)
      system(pip_cmd)
    } else {
      message("To install, run:")
      message("  ", python_path, " -m pip install ",
             if (pip_user) "--user " else "",
             paste(missing_required, collapse = " "))
    }
  } else {
    message("✓ All required packages are installed")
  }
  
  if (length(missing_optional) > 0) {
    message("\nOptional packages not found: ", paste(missing_optional, collapse = ", "))
    if (install_deps && length(missing_required) == 0) {
      message("To install optional packages, run:")
      message("  ", python_path, " -m pip install ",
             if (pip_user) "--user " else "",
             paste(missing_optional, collapse = " "))
    }
  }
  
  # Import modules
  tryCatch({
    if (reticulate::py_module_available("PIL")) {
      assign("PIL", reticulate::import("PIL", delay_load = TRUE), envir = .GlobalEnv)
    }
    if (reticulate::py_module_available("resmem")) {
      assign("resmem", reticulate::import("resmem", delay_load = TRUE), envir = .GlobalEnv)
    }
    if (reticulate::py_module_available("thingsvision")) {
      assign("tv", reticulate::import("thingsvision", delay_load = TRUE), envir = .GlobalEnv)
      assign("tv_data", reticulate::import("thingsvision.utils.data", delay_load = TRUE), envir = .GlobalEnv)
      assign("tv_utils_storing", reticulate::import("thingsvision.utils.storing", delay_load = TRUE), envir = .GlobalEnv)
      assign("tv_core_extraction", reticulate::import("thingsvision.core.extraction", delay_load = TRUE), envir = .GlobalEnv)
    }
    message("\n✓ Python configured successfully for imfeatures")
  }, error = function(e) {
    message("\nNote: Some modules could not be imported: ", e$message)
  })
  
  invisible(python_path)
}

#' Install Python dependencies for imfeatures
#'
#' Convenience function to install required and optional Python packages
#' for imfeatures using pip.
#'
#' @param python_cmd Python command or path (default: "python3")
#' @param user Whether to use --user flag (default: TRUE for HPC)
#' @param optional Whether to install optional packages (default: FALSE)
#' @param upgrade Whether to upgrade existing packages (default: FALSE)
#' @param no_deps Whether to use --no-deps flag to avoid dependency conflicts (default: FALSE)
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
                               upgrade = FALSE,
                               no_deps = FALSE) {
  
  # Find Python
  if (!file.exists(python_cmd)) {
    python_cmd <- Sys.which(python_cmd)
    if (!nzchar(python_cmd)) {
      stop("Python executable not found. Please specify the full path or ensure Python is in PATH.")
    }
  }
  
  message("Using Python: ", python_cmd)
  
  # Build package list
  packages <- c("Pillow", "numpy")
  if (optional) {
    packages <- c(packages, "thingsvision", "torch", "torchvision", "resmem")
  }
  
  # Build pip command
  pip_args <- character()
  if (user) pip_args <- c(pip_args, "--user")
  if (upgrade) pip_args <- c(pip_args, "--upgrade")
  if (no_deps) pip_args <- c(pip_args, "--no-deps")
  
  pip_cmd <- paste(python_cmd, "-m pip install",
                  paste(pip_args, collapse = " "),
                  paste(packages, collapse = " "))
  
  message("Installing packages with:")
  message("  ", pip_cmd)
  
  result <- system(pip_cmd)
  
  if (result == 0) {
    message("\n✓ Installation completed successfully")
    message("You can now use: library(imfeatures); use_existing_python()")
  } else {
    warning("Installation may have failed. Please check the output above.")
  }
  
  invisible(result)
}

#' Install thingsvision on HPC systems with problematic dependencies
#'
#' Special installer for HPC systems (like Compute Canada) where certain
#' dependency wheels may be corrupted or incompatible. This function installs
#' thingsvision without dependencies, then selectively installs only the
#' required dependencies.
#'
#' @param python_cmd Python command or path (default: "python3")
#' @param user Whether to use --user flag (default: TRUE for HPC)
#'
#' @details
#' This function is specifically designed for HPC systems where:
#' - TensorFlow wheels may be corrupted (e.g., Compute Canada)
#' - System-wide installations of PyTorch are preferred
#' - Dependencies need to be carefully managed
#'
#' The function will:
#' 1. Install thingsvision without dependencies
#' 2. Install only the PyTorch-related dependencies we actually need
#' 3. Skip problematic packages like tensorflow
#'
#' @examples
#' \dontrun{
#' # On Compute Canada / HPC system
#' install_thingsvision_hpc()
#' 
#' # Then configure Python
#' use_existing_python()
#' }
#'
#' @export
install_thingsvision_hpc <- function(python_cmd = "python3", user = TRUE) {
  
  # Find Python
  if (!file.exists(python_cmd)) {
    python_cmd <- Sys.which(python_cmd)
    if (!nzchar(python_cmd)) {
      stop("Python executable not found. Please specify the full path or ensure Python is in PATH.")
    }
  }
  
  message("Using Python: ", python_cmd)
  message("\nInstalling thingsvision for HPC (avoiding problematic dependencies)...")
  
  user_flag <- if (user) "--user" else ""
  
  # Step 1: Install thingsvision without dependencies
  message("\n1. Installing thingsvision without dependencies...")
  cmd1 <- paste(python_cmd, "-m pip install", user_flag, "--no-deps thingsvision")
  message("Running: ", cmd1)
  result1 <- system(cmd1)
  
  if (result1 != 0) {
    stop("Failed to install thingsvision. Please check your Python environment.")
  }
  
  # Step 2: Install core dependencies (avoiding tensorflow)
  message("\n2. Installing core dependencies...")
  core_deps <- c(
    "torch",
    "torchvision", 
    "torchtyping",  # Required by thingsvision
    "numpy",
    "Pillow",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "h5py",
    "open-clip-torch",
    "ftfy",
    "regex",
    "safetensors"
  )
  
  # Check which are already installed
  missing_deps <- character()
  for (dep in core_deps) {
    check_cmd <- paste(python_cmd, "-c \"import", gsub("-", "_", dep), "\"", "2>/dev/null")
    if (system(check_cmd, ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
      missing_deps <- c(missing_deps, dep)
    }
  }
  
  if (length(missing_deps) > 0) {
    message("Installing missing dependencies: ", paste(missing_deps, collapse = ", "))
    cmd2 <- paste(python_cmd, "-m pip install", user_flag, paste(missing_deps, collapse = " "))
    message("Running: ", cmd2)
    result2 <- system(cmd2)
    
    if (result2 != 0) {
      message("\nSome dependencies may have failed to install.")
      message("This is often OK on HPC systems where packages are pre-installed.")
      message("Continuing...")
    }
  } else {
    message("All core dependencies are already installed.")
  }
  
  # Step 3: Handle tensorflow requirement (create dummy if needed)
  message("\n3. Handling tensorflow requirement...")
  tf_check <- paste(python_cmd, "-c \"import tensorflow\"", "2>/dev/null")
  if (system(tf_check, ignore.stdout = TRUE, ignore.stderr = TRUE) != 0) {
    message("TensorFlow not found. Creating dummy module to satisfy import...")
    # Create a minimal dummy tensorflow module
    tf_dir <- paste0("~/.local/lib/python3.10/site-packages/tensorflow")
    system(paste("mkdir -p", tf_dir), ignore.stderr = TRUE)
    system(paste0("echo '__version__ = \"2.0.0\"' > ", tf_dir, "/__init__.py"), ignore.stderr = TRUE)
    system(paste0("echo 'class keras: pass' >> ", tf_dir, "/__init__.py"), ignore.stderr = TRUE)
    message("Dummy tensorflow module created.")
  }
  
  # Step 4: Verify installation
  message("\n4. Verifying installation...")
  verify_cmd <- paste(python_cmd, "-c \"import thingsvision; print('thingsvision version:', thingsvision.__version__)\"")
  result3 <- system(verify_cmd)
  
  if (result3 == 0) {
    message("\n✓ thingsvision installed successfully!")
    message("\nYou can now use:")
    message("  library(imfeatures)")
    message("  use_existing_python()")
    message("  list_module_names('resnet50')")
  } else {
    message("\n⚠ thingsvision import failed. You may need to install additional dependencies manually.")
    message("Try running:")
    message("  ", python_cmd, " -c \"import thingsvision\"")
    message("to see what dependencies are missing.")
  }
  
  invisible(result3 == 0)
}