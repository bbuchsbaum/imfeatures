
#resmem <<- reticulate::import("resmem")
#PIL <<- reticulate::import("PIL")
resmodel <- NULL

memorability <- function(im) {
  if (is.null(resmodel)) {
    resmodel <<- resmem$ResMem(pretrained=TRUE)
  }
  img = PIL$Image$open(im) # This loads your image into memory
  img = img$convert('RGB')
  resmodel$eval()
  image_x = resmem$transformer(img)
  prediction = resmodel(image_x$view(-1L, 3L, 227L, 227L))
}
