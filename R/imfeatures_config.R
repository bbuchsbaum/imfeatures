#' Configure the Python environment for imfeatures
#'
#' Sets up or activates the Python environment required by the package.
#' The environment is detected once per R session and cached.  Users can
#' force a reinstall with `reset = TRUE`.
#'
#' @param envname Name of the environment to create/use. Defaults to
#'   "r-imfeatures".
#' @param method Installation method passed to `reticulate::py_install`.
#'   Use "auto", "conda" or "virtualenv".
#' @param reset Force reinstallation of the environment.
#' @return Invisible path to the configured Python binary.
#' @export
imfeatures_config <- local({
  cached <- NULL
  function(envname = "r-imfeatures",
           method = c("auto", "conda", "virtualenv"),
           reset = FALSE) {
    if (!is.null(cached) && !reset) {
      cli_msg <- crayon::green(paste0("Using cached Python env: ", cached))
      message(cli_msg)
      return(invisible(cached))
    }
    method <- match.arg(method)
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
    if (!exists) {
      message(crayon::yellow(paste0("Creating Python env '", envname, "'")))
      reticulate::py_install(envname = envname,
                             method = method,
                             packages = character(),
                             requirements = system.file("requirements.txt", package = "imfeatures"))
    }
    if (method == "conda") {
      reticulate::use_condaenv(envname, required = TRUE)
      cfg <- reticulate::conda_python(envname)
    } else {
      reticulate::use_virtualenv(envname, required = TRUE)
      cfg <- reticulate::virtualenv_python(envname)
    }
    cached <<- cfg
    backend <- "CPU"
    if (reticulate::py_module_available("torch")) {
      torch <- reticulate::import("torch")
      if (torch$cuda$is_available()) {
        device <- try(torch$cuda$get_device_name(0L), silent = TRUE)
        backend <- paste0("CUDA (", ifelse(inherits(device, "try-error"), "GPU", device), ")")
      }
    }
    msg <- paste0("Backend: ", backend)
    if (backend == "CPU") message(crayon::blue(msg)) else message(crayon::green(msg))
    invisible(cached)
  }
})
