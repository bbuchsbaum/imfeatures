# extract features from intermediate layers

extract features from intermediate layers

## Usage

``` r
extract_features(
  impath,
  layers,
  model = NULL,
  target_size = c(224, 224),
  spatial_pooling = "none"
)

im_features(
  impath,
  layers,
  model = NULL,
  target_size = c(224, 224),
  spatial_pooling = "none"
)
```

## Arguments

- impath:

  path to image file

- layers:

  the layer indices

- model:

  the Keras model

- target_size:

  the target image dimensions (height, width) appropriate for the model

- spatial_pooling:

  How to reduce 4-D feature maps (`1 x H x W x C`, as produced by
  convolutional layers) over height and width. One of:

  `"none"`

  :   (Default) No pooling. The full map is returned as a 4-D array.

  `"avg"`

  :   Global average pooling: the mean over all positions, giving a
      vector of length `C`.

  `"max"`

  :   Global max pooling: the maximum over all positions, giving a
      vector of length `C`.

  `"resize_HxW"`

  :   Resize the map to `H` by `W` and flatten, for example
      `"resize_3x3"` or `"resize_7x7"`. Any positive `H` and `W` are
      accepted. This keeps coarse spatial layout and gives a vector of
      length `H * W * C`.

  See the Spatial pooling section for details.

## Value

A tibble with columns `image`, `layer` and a list-column `feature`. The
tibble inherits class `imfeatures_feature_tbl` for dplyr compatibility.

## Spatial pooling

Pooling trades spatial detail for size. For VGG-16 `block5_conv3` (a 14
x 14 map with 512 channels), `"none"` keeps 100,352 values,
`"resize_3x3"` keeps 4,608, and `"avg"` or `"max"` keep 512.

`"resize_HxW"` shrinks a map by area averaging: each output cell is the
mean of the input positions it covers, with partial overlap weighted by
area. Averaging the cells of a resized map therefore reproduces `"avg"`.
If the requested size is larger than the map in either direction,
bilinear interpolation is used instead. Resizing runs in TensorFlow. If
TensorFlow is unavailable, a warning is raised and the unpooled map is
returned.

A resized or unpooled map is flattened with height varying fastest, then
width, then channel. To recover the array from a flattened
`"resize_HxW"` vector `v`, use `array(v, dim = c(H, W, C))`.

`spatial_pooling` only affects 4-D outputs. Dense layers such as `fc1`
have no spatial axes and are returned as they are.

These string options serve `extract_features()`,
[`compute_feature_similarity()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_feature_similarity.md)
and
[`extract_vgg_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_vgg_features.md).
For per-stage pooling, and for grid, DCT and radial pooling, use
[`extract_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md)
with the pooler objects described in
[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md).

## See also

[`vignette("spatial-pooling", package = "imfeatures")`](https://bbuchsbaum.github.io/imfeatures/articles/spatial-pooling.md)
for a comparison of all pooling options.

## Examples

``` r
if (FALSE) { # \dontrun{
# Extract features from a single image using default VGG16 model
img_path <- system.file("extdata", "example.jpg", package = "imfeatures")

# Extract features from multiple layers
features <- extract_features(
  impath = img_path,
  layers = c(3, 5, 7), # conv1_2, conv2_1, conv2_2
  model = NULL, # Uses default VGG16
  target_size = c(224, 224)
)

# Extract features with spatial pooling
features_pooled <- extract_features(
  impath = img_path,
  layers = c(10, 12), # Later convolutional layers
  spatial_pooling = "avg" # Global average pooling
)

# Extract features with spatial resizing
features_resized <- extract_features(
  impath = img_path,
  layers = c(10),
  spatial_pooling = "resize_7x7" # Resize spatial dimensions to 7x7
)

# Access the extracted features
layer3_features <- features$feature[[1]] # Features from layer 3
dim(layer3_features) # Check dimensions
} # }
```
