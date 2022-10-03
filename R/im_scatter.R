
#' a 2D scatterplot with images displayed at each location
#'
#' @import ggimage
#' @import ggplot2
#' @param dframe a `data.frame` containing x and y coordinates and path to image file.
#' @param dframe xvar the name of the variable containing the x coordinates
#' @param dframe yvar the name of the variable containing the y coordinates
#' @param imagename the name of the image variable in `dframe`
#' @export
im_scatter <- function(dframe, xvar="x", yvar="y", imagename="image") {
  ggplot(dframe, aes_string(xvar,yvar)) + geom_image(aes_string(image=imagename)) + theme_bw()
}

#' a 3D scatterplot with images displayed at each location
#'
#' @import rgl
#' @param dframe a `data.frame` containing x, y, z coordinates and path to image file.
#' @param imagename the name of the image variable in `dframe`
#' @param radius the radius of the image sprite
#' @param width width of device in pixels
#' @param height height of device in pixels
#' @export
#' @importFrom imager load.image save.image
im_scatter3d <- function(dframe, imagename="image", radius=1, width=700, height=700, bgcol="white") {
  tfiles <- vector(nrow(dframe), mode="list")
  for (i in 1:nrow(dframe)) {
    tmpF <- tempfile(fileext=".png")
    im <- imager::load.image(dframe[[imagename]][i])
    imager::save.image(im,tmpF)
    tfiles[[i]] <- tmpF
  }
  rgl.open()
  rgl::rgl.bg(color=bgcol)
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
