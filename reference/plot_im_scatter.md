# a 2D scatterplot with images displayed at each location

a 2D scatterplot with images displayed at each location

a 3D scatterplot with images displayed at each location

## Usage

``` r
plot_im_scatter(dframe, xvar = "x", yvar = "y", imagename = "image")

im_scatter(dframe, xvar = "x", yvar = "y", imagename = "image")

plot_im_scatter3d(
  dframe,
  imagename = "image",
  radius = 1,
  width = 700,
  height = 700,
  bgcol = "white"
)

im_scatter3d(
  dframe,
  imagename = "image",
  radius = 1,
  width = 700,
  height = 700,
  bgcol = "white"
)
```

## Arguments

- dframe:

  a \`data.frame\` containing x, y, z coordinates and path to image
  file.

- xvar:

  the name of the variable containing the x coordinates

- yvar:

  the name of the variable containing the y coordinates

- imagename:

  the name of the image variable in \`dframe\`

- radius:

  the radius of the image sprite

- width:

  width of device in pixels

- height:

  height of device in pixels

- bgcol:

  background color for the 3D plot (default: "white")
