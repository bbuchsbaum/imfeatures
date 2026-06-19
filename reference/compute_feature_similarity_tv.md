# Compute Similarity Matrix using thingsvision Features

Calculates the pairwise similarity between a set of images based on
features extracted from specified model layers using the
\`thingsvision\` backend.

## Usage

``` r
compute_feature_similarity_tv(
  impaths,
  model_name,
  source = "torchvision",
  module_names = NULL,
  metric = "cosine",
  flatten_acts = TRUE,
  device = "cuda",
  pretrained = TRUE,
  model_parameters = NULL,
  batch_size = 32L,
  temp_out_dir = tempdir()
)

im_feature_sim_tv(
  impaths,
  model_name,
  source = "torchvision",
  module_names = NULL,
  metric = "cosine",
  flatten_acts = TRUE,
  device = "cuda",
  pretrained = TRUE,
  model_parameters = NULL,
  batch_size = 32L,
  temp_out_dir = tempdir()
)
```

## Arguments

- impaths:

  Character vector. A vector of full file paths to the images. The order
  determines the rows/columns of the output similarity matrices. Images
  can reside in different directories and will be processed relative to
  their computed common root.

- model_name:

  Character string. The name of the \`thingsvision\` model architecture
  (e.g., \`"resnet50"\`, \`"clip"\`). Must be a non-empty string.

- source:

  Character string. The source library of the model. Defaults to
  \`"torchvision"\`. Other options include \`"timm"\`, \`"ssl"\`,
  \`"custom"\`.

- module_names:

  Character vector. The specific layer/module names within the model
  from which to extract features for similarity calculation. If NULL
  (default), automatically selects an appropriate layer based on the
  model architecture. Use \`show_model(tv_get_extractor(model_name,
  source))\` to find valid names.

- metric:

  Character string. The similarity metric to use. Defaults to "cosine".
  Common options include "cosine", "correlation", "Euclidean",
  "Manhattan". Use \`proxy::pr_DB\$get_entry_names()\` to see all
  available metrics.

- flatten_acts:

  Logical. Should activations from the specified \`module_names\` be
  flattened into vectors before calculating similarity? This is almost
  always required for standard similarity metrics like cosine or
  correlation. Defaults to TRUE. Setting to FALSE will likely cause
  errors unless the metric can handle multi-dimensional arrays and the
  chosen layer output is suitable.

- device:

  Character string. The compute device ("cpu", "cuda", "cuda:0").
  Defaults to "cuda".

- pretrained:

  Logical. Use pretrained model weights? Defaults to TRUE.

- model_parameters:

  Named list (optional). Additional parameters for specific models
  (e.g., \`list(variant = "ViT-B/32")\` for CLIP). Defaults to NULL.

- batch_size:

  Integer. Batch size for feature extraction. Defaults to 32.

- temp_out_dir:

  Character string. Temporary directory for internal file list used
  during feature extraction. Defaults to \`tempdir()\`.

## Value

A named list of similarity matrices. Elements are named according to the
provided \`module_names\`. If a name is repeated, a numeric suffix
("\_1", "\_2", ...) is appended to keep names unique. Each element
contains a square similarity matrix (n_images x n_images).

## Details

This function streamlines the process of calculating representational
similarity matrices (RSMs) using features from the \`thingsvision\`
ecosystem.

**Workflow:**

1.  It iterates through each \`module_name\` provided.

2.  For each module, it calls
    [`im_features_tv`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features_tv.md)
    to extract features for all images specified in \`impaths\`. The
    \`flatten_acts\` parameter is crucial here to ensure features are in
    a suitable format (usually 2D matrix) for standard similarity
    calculation.

3.  It then calculates the full pairwise similarity matrix for the
    extracted features using the specified \`metric\` via the \`proxy\`
    package (or \`coop\` for optimized cosine).

4.  Rownames and colnames of the similarity matrices are set based on
    the image basenames.

**Memory Considerations:** This function extracts features for \*all\*
images for a given module \*before\* calculating the similarity matrix
for that module. This is generally efficient if the features for all
images fit into memory. It does \*not\* currently implement the
pair-by-pair extraction (\`lowmem=TRUE\`) strategy found in the original
\`im_feature_sim\` function, as the primary bottleneck is often feature
extraction itself when using large models. If memory issues arise during
the similarity calculation step (after feature extraction), consider
using metrics optimized for memory or processing subsets of images. The
\`output_dir\` option in \`im_features_tv\` can handle cases where the
features \*themselves\* don't fit in memory during extraction.

**Prerequisites:** Requires a correctly configured Python environment
with \`thingsvision\` installed. Use
[`imfeatures_config`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md)
and configure \`reticulate\` before use.

## See also

[`im_features_tv`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features_tv.md),
[`imfeatures_config`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md),
[`tv_get_extractor`](https://bbuchsbaum.github.io/imfeatures/reference/tv_get_extractor.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# --- Prerequisites ---
# imfeatures_config()
library(imfeatures)
library(reticulate)
tryCatch({
  use_condaenv("r-thingsvision", required = TRUE)
  tv <- import("thingsvision")
}, error = function(e) message("Python env 'r-thingsvision' not found."))

# --- Example Usage ---
# Create dummy image files
image_dir <- file.path(tempdir(), "sim_test_images")
dir.create(image_dir, showWarnings = FALSE)
png(file.path(image_dir, "cat.png")); plot(1:5); dev.off()
png(file.path(image_dir, "dog.png")); plot(rnorm(50)); dev.off()
png(file.path(image_dir, "car.png")); plot(1:20); dev.off()
image_paths <- list.files(image_dir, full.names = TRUE, pattern = "\\.png$")

# Calculate similarity based on ResNet-18 avgpool and layer4 features
sim_results <- im_feature_sim_tv(
  impaths = image_paths,
  model_name = "resnet18",
  source = "torchvision",
  module_names = c("avgpool", "layer4"), # Request features from two layers
  metric = "cosine",
  flatten_acts = TRUE, # Flatten layer4 activations
  device = "cpu"
)

# Explore results
print(names(sim_results))
print(dim(sim_results$avgpool))
print(sim_results$avgpool)

# Example showing argument validation (will error)
im_feature_sim_tv(
  impaths = image_paths,
  model_name = "",
  source = "torchvision",
  module_names = "avgpool"
)
#> Error: 'model_name' must be a non-empty character string.

# Clean up
unlink(image_dir, recursive = TRUE)
} # }
```
