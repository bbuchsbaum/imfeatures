with_mocked_bindings <- function(..., .env = parent.frame()) {
  dots <- substitute(list(...))
  code <- dots[[length(dots)]]
  mock_exprs <- dots[-length(dots)]
  mocks <- lapply(mock_exprs, eval, envir = parent.frame())
  names(mocks) <- names(mock_exprs)

  # Debug print
  cat("Mocking the following functions:\n")
  print(names(mocks))
  
  restore <- list()

  apply_mock <- function(name, value) {
    pkg <- NULL
    target_env <- .env
    obj <- name
    
    # Debug print
    cat("Processing mock:", name, "\n")
    
    # Handle :: notation (package::function)
    if (grepl("::", name, fixed = TRUE)) {
      pkg <- sub("::.*", "", name)
      obj <- sub(".*::", "", name)
      cat("  Package:", pkg, "Function:", obj, "\n")
      if (pkg == "base") {
        target_env <- baseenv()
      } else {
        target_env <- getNamespace(pkg)
      }
    }

    if (grepl("\\$", obj)) {
      pieces <- strsplit(obj, "\\$", fixed = TRUE)[[1]]
      container_name <- pieces[[1]]
      field_name <- pieces[[2]]
      cat("  Container:", container_name, "Field:", field_name, "\n")
      container <- get(container_name, envir = target_env)
      restore[[name]] <<- container[[field_name]]
      container[[field_name]] <- value
      assign(container_name, container, envir = target_env)
    } else {
      cat("  Getting function:", obj, "from env\n")
      restore[[name]] <<- get(obj, envir = target_env)
      assign(obj, value, envir = target_env)
    }
  }

  for (nm in names(mocks)) {
    tryCatch({
      apply_mock(nm, mocks[[nm]])
    }, error = function(e) {
      cat("ERROR in apply_mock for", nm, ":", e$message, "\n")
    })
  }

  on.exit({
    for (nm in rev(names(restore))) {
      pkg <- NULL
      target_env <- .env
      obj <- nm
      if (grepl("::", nm)) {
        pkg <- sub("::.*", "", nm)
        obj <- sub(".*::", "", nm)
        target_env <- getNamespace(pkg)
      }
      if (grepl("\\$", obj)) {
        pieces <- strsplit(obj, "\\$", fixed = TRUE)[[1]]
        container_name <- pieces[[1]]
        field_name <- pieces[[2]]
        container <- get(container_name, envir = target_env)
        container[[field_name]] <- restore[[nm]]
        assign(container_name, container, envir = target_env)
      } else {
        assign(obj, restore[[nm]], envir = target_env)
      }
    }
  }, add = TRUE)

  eval(code, envir = parent.frame())
}
