
# @import ggimage
im_scatter <- function(dframe, xvar="x", yvar="y", imagename="image") {
  ggplot(dframe, aes(x,y)) + geom_image(aes(image=image)) + theme_bw()
}

#' @import rgl
im_scatter3d <- function(dframe, imagename="image", radius=1) {
  tfiles <- vector(nrow(dframe), mode="list")
  for (i in 1:nrow(dframe)) {
    tmpF <- tempfile(fileext=".png")
    im <- load.image(dframe$image[i])
    save.image(im,tmpF)
    tfiles[[i]] <- tmpF
  }
  rgl.open()
  par3d(windowRect=c(0,0,700,700),zoom=.6)

  for (i in 1:nrow(dframe)) {
    rgl.sprites(dframe$x[i],dframe$y[i],dframe$z[i],radius=radius,
                texture=tfiles[[i]], shininess=50, specular="black", lit=FALSE)
    #rgl.spheres(consred[i,1],consred[i,2],consred[i,3],radius=2,
    #            texture=inames[i], shininess=50, specular="black", lit=FALSE)
  }

  par3d(windowRect=c(0,0,700,700),zoom=.6)
  plot3d(dframe$x, dframe$y, dframe$z, type="n")

  for (i in 1:nrow(dframe)) {
    rgl.sprites(dframe$x[i],dframe$y[i],dframe$z[i],radius=radius,
                texture=tfiles[[i]], shininess=50, specular="black", lit=FALSE)
    #rgl.spheres(consred[i,1],consred[i,2],consred[i,3],radius=2,
    #            texture=inames[i], shininess=50, specular="black", lit=FALSE)
  }

}
