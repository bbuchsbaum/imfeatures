resmem <- NULL
PIL <- NULL
resmodel <- NULL

#' @import reticulate
.onLoad <- function(libname, pkgname) {
  resmem <<- reticulate::import("resmem", delay_load=TRUE)
  PIL <<- reticulate::import("PIL", delay_load=TRUE)
  resmodel <<- resmem$ResMem(pretrained=TRUE)
}
