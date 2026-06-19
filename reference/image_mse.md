# Multiscale entropy for images

Compute the entropy of the Hue, Saturation and Value components of an
image across multiple blur scales.

## Usage

``` r
image_mse(im, sf = c(100, 50, 8, 4, 0), bins = 16L)
```

## Arguments

- im:

  image of type \`cimg\` from \`imager\` package. The function expects
  an image with three colour channels. If a single-channel image is
  supplied it will be converted with
  [`add.colour()`](https://rdrr.io/pkg/imager/man/add.colour.html), and
  an error is thrown if the result does not contain exactly three
  channels.

- sf:

  vector smoothing factors indicating the scales for entropy computation

- bins:

  number of bins for computing information

## Value

A named numeric vector with the mean multiscale entropy for the `H`, `S`
and `V` channels.

## Examples

``` r
if (FALSE) { # \dontrun{
# Load an example image
library(imager)
img <- load.example("parrots")

# Compute multiscale entropy with default parameters
mse_values <- image_mse(img)
print(mse_values)  # H, S, V entropy values

# Use custom smoothing factors for different scales
mse_custom <- image_mse(
  im = img,
  sf = c(200, 100, 50, 25, 0),  # More smoothing levels
  bins = 32L  # More bins for finer discretization
)

# Process a grayscale image (will be converted to color)
gray_img <- grayscale(img)
mse_gray <- image_mse(gray_img)
# Note: H and S channels will be 0 for grayscale images

# Batch process multiple images
img_files <- list.files("path/to/images", pattern = "\\.jpg$", full.names = TRUE)
mse_results <- lapply(img_files, function(f) {
  img <- load.image(f)
  image_mse(img, sf = c(50, 25, 0))
})

# Convert to data frame for analysis
mse_df <- do.call(rbind, mse_results)
colMeans(mse_df)  # Average entropy across images
} # }
```
