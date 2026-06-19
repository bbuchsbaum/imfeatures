# Calculate Edge Entropy Features from Images

This function calculates first-order and pairwise edge entropy features
from an image, which can be used for analyzing texture and structural
complexity in images.

## Usage

``` r
compute_edge_entropy(
  image,
  max_pixels = 120000L,
  maxdiag = 500L,
  gabor_bins = 24L,
  filter_length = 31L,
  circ_bins = 48L,
  ranges = list(c(20, 80), c(80, 160), c(160, 240)),
  use_cpp = TRUE,
  verbose = FALSE
)

edge_entropy(
  image,
  max_pixels = 120000L,
  maxdiag = 500L,
  gabor_bins = 24L,
  filter_length = 31L,
  circ_bins = 48L,
  ranges = list(c(20, 80), c(80, 160), c(160, 240)),
  use_cpp = TRUE,
  verbose = FALSE
)
```

## Arguments

- image:

  Either a file path to an image (character string) or a numeric matrix
  representing a grayscale image. If a file path is provided, the image
  will be loaded and converted to grayscale.

- max_pixels:

  Integer. The maximum number of pixels allowed in the processed image.
  Larger images will be resized. Only used when `image` is a file path.
  Set to NULL to disable resizing. Defaults to 300\*400.

- maxdiag:

  Integer. Maximum diagonal distance for pairwise entropy calculations.
  Defaults to 500.

- gabor_bins:

  Integer. Number of orientation bins for Gabor filter bank. Defaults to
  24.

- filter_length:

  Integer. Size of the Gabor filters (must be odd). Defaults to 31.

- circ_bins:

  Integer. Number of circular bins for directional statistics. Defaults
  to 48.

- ranges:

  List of integer vectors. Each vector should contain two elements
  specifying the start and end indices for grouping pairwise entropy at
  different distance ranges. Defaults to list(c(20,80), c(80,160),
  c(160,240)).

- use_cpp:

  Logical. Whether to use the C++ implementation (generally faster).
  Defaults to TRUE. If FALSE, uses the pure R implementation.

- verbose:

  Logical. If TRUE, print detailed progress messages.

## Value

A data frame with the following columns:

- im:

  Image identifier (file path or "matrix" for matrix input)

- entropy:

  First-order entropy value

- pentropy_20_80:

  Pairwise entropy for distance range 20-80

- pentropy_80_160:

  Pairwise entropy for distance range 80-160

- pentropy_160_240:

  Pairwise entropy for distance range 160-240

- complex_before:

  Image complexity measure before thresholding

## Details

Edge entropy measures quantify the distribution and organization of
oriented edges in images. The method applies a bank of Gabor filters at
different orientations and measures both first-order entropy
(distribution of dominant orientations) and pairwise entropy (how
orientation relationships vary with distance and direction).

The C++ implementation is substantially faster for larger images but
requires the same inputs. It automatically converts the ranges list to
the required format for the C++ function.

## Examples

``` r
# \donttest{
# Example 1: Using a matrix with default parameters
img_matrix <- matrix(runif(100*100), nrow=100)
result <- compute_edge_entropy(img_matrix)
print(result$entropy)  # First-order orientation entropy
#> [1] 4.581683
print(result$pentropy_20_80)  # Pairwise entropy for distance range 20-80
#> [1] 4.581904

# Example 2: Load and process an image file
library(imager)
#> Loading required package: magrittr
#> 
#> Attaching package: ‘imager’
#> The following object is masked from ‘package:magrittr’:
#> 
#>     add
#> The following objects are masked from ‘package:stats’:
#> 
#>     convolve, spectrum
#> The following object is masked from ‘package:graphics’:
#> 
#>     frame
#> The following object is masked from ‘package:base’:
#> 
#>     save.image
img <- load.example("coins")
gray_img <- grayscale(img)
#> Warning: Image appears to already be in grayscale mode
img_array <- as.array(gray_img)[,,1,1]

# Compute edge entropy with custom parameters
result_custom <- compute_edge_entropy(
  image = img_array,
  gabor_bins = 32L,  # More orientation bins
  filter_length = 41L,  # Larger filter for coarser features
  circ_bins = 64L,  # More angular bins
  use_cpp = TRUE  # Use fast C++ implementation
)

# Example 3: Compare R and C++ implementations
small_img <- matrix(runif(50*50), nrow=50)
result_r <- compute_edge_entropy(small_img, use_cpp = FALSE, verbose = TRUE)
#> [R] filter_bank: First Gabor filter range: [-0.599173185452427, 0.599173185452427], Sum = -8.80221613286648e-17
#> [R] run_filterbank: Input image range: [2.5581568479538e-05, 0.998988696141168], Mean = 0.506527143447753
#> [R] run_filterbank: First convolution result (conv) range: [-3.41757949879452, 2.96181634218024], Sum = -125.034634143302
#> [R] run_filterbank: Range resp_val (after border zero): [0, 1.95445907452918]
#> [R] do_counting: Range resp_val (input): [0, 1.95445907452918]
#> [R] do_counting: Calculated cutoff (k-th=2500): 1.10975810535899
#> [R] do_counting: Range resp_val (after cutoff): [0, 1.95445907452918]
#> [R] do_counting: Range counts cube: [0, 55.9581081643101], Sum = 106112.855109236
#> [R] do_statistics: counts_sum range: [1e-05, 127.439600519988], Sum = 106113.095109236
#> [R] do_statistics: normalized_counts range: [0, 0.999998340887017], Sum = 2597.99799676068
#> [R] do_statistics: shannon matrix range: [0, 4.35800809998248]
#> [R] edge_entropy (matrix): Range [20, 80]: rowmeans mean = 2.362952
#> [R] edge_entropy (matrix): Range [80, 160]: rowmeans mean = NA
#> [R] edge_entropy (matrix): Range [160, 240]: rowmeans mean = NA
#> [R] edge_entropy (matrix): Calculated fo = 4.55069384061516
#> [R] edge_entropy (matrix): final_shannon[1] = 2.36295151946193
#> [R] edge_entropy (matrix): final_shannon[2] = NA
#> [R] edge_entropy (matrix): final_shannon[3] = NA
result_cpp <- compute_edge_entropy(small_img, use_cpp = TRUE, verbose = FALSE)

# Results should be very similar
all.equal(result_r$entropy, result_cpp$entropy, tolerance = 1e-10)
#> [1] TRUE

result_ranges <- compute_edge_entropy(
  image = img_array,
  ranges = list(c(10,50), c(50,100), c(100,200), c(200,400)),
  maxdiag = 600L
)
# }
if (FALSE) { # \dontrun{
img_files <- list.files("path/to/images", pattern = "\\.jpg$", full.names = TRUE)
entropy_results <- lapply(img_files[1:5], function(f) {
  img <- imager::load.image(f)
  gray <- imager::grayscale(img)
  arr <- as.array(gray)[,,1,1]
  compute_edge_entropy(arr, max_pixels = 100000L)
})
} # }
```
