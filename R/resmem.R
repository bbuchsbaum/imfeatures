
#resmem <<- reticulate::import("resmem")
#PIL <<- reticulate::import("PIL")
resmodel <- NULL

cacheEnv <- new.env()

memorability <- function(im) {
  if (is.null(cacheEnv$resmodel)) {
    cacheEnv$resmodel <- resmem$ResMem(pretrained=TRUE)
  }
  img = PIL$Image$open(im) # This loads your image into memory
  img = img$convert('RGB')
  cacheEnv$resmodel$eval()
  image_x = resmem$transformer(img)
  prediction = cacheEnv$resmodel(image_x$view(-1L, 3L, 227L, 227L))
  prediction$detach()$numpy()[1,1]
}
