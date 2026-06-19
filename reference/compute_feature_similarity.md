# Compute similarity matrix for a set of image using feature vectors from a Keras 3 model

Compute similarity matrix for a set of image using feature vectors from
a Keras 3 model

## Usage

``` r
compute_feature_similarity(
  impaths,
  layers,
  model = NULL,
  target_size = c(224, 224),
  spatial_pooling = "none",
  metric = "cosine",
  lowmem = TRUE,
  cache_size = 2048 * 2048^2,
  subsamp_prop = 1
)

im_feature_sim(
  impaths,
  layers,
  model = NULL,
  target_size = c(224, 224),
  spatial_pooling = "none",
  metric = "cosine",
  lowmem = TRUE,
  cache_size = 2048 * 2048^2,
  subsamp_prop = 1
)
```

## Arguments

- impaths:

  paths to image files (vector of file paths)

- layers:

  the layer indices

- model:

  the Keras model

- target_size:

  the target image dimensions for appropriate for model

- spatial_pooling:

  A character string specifying the type of spatial processing to apply
  to 4D feature maps (see `extract_features` for details)

- metric:

  the similarity metric to use, default is 'cosine' (see `proxy` package
  for allowable metrics)

- lowmem:

  logical, if TRUE use memory-efficient computation (default: TRUE)

- cache_size:

  maximum cache size in bytes for memoization (default: 2048 \* 2048^2)

- subsamp_prop:

  proportion of features to subsample (0 to 1, default: 1 for no
  subsampling)

## Value

A list of similarity matrices, one for each layer

## Examples

``` r
if (FALSE) { # \dontrun{
# Create a vector of image paths
img_dir <- system.file("extdata", package = "imfeatures")
img_paths <- list.files(img_dir, pattern = "\\.jpg$", full.names = TRUE)

# Compute similarity matrix using features from specific layers
sim_matrix <- compute_feature_similarity(
  impaths = img_paths,
  layers = c(10, 15),  # Two VGG16 layers
  model = NULL,  # Use default VGG16
  target_size = c(224, 224),
  metric = "cosine"
)

# Access similarity matrices for each layer
layer10_sim <- sim_matrix$layer_10
layer15_sim <- sim_matrix$layer_15

# Compute similarity with spatial pooling for efficiency
sim_pooled <- compute_feature_similarity(
  impaths = img_paths,
  layers = c(12),
  spatial_pooling = "avg",  # Average pool spatial dimensions
  metric = "cosine",
  lowmem = TRUE  # Memory-efficient computation
)

# Use subsampling for very large feature vectors
sim_subsampled <- compute_feature_similarity(
  impaths = img_paths,
  layers = c(10),
  subsamp_prop = 0.5,  # Use 50% of features
  metric = "euclidean"
)

# Visualize similarity matrix
heatmap(sim_matrix$layer_10, symm = TRUE)
} # }
```
