
<!-- README.md is generated from README.Rmd. Please edit that file -->

# imfeatures

<!-- badges: start -->

[![R-CMD-check](https://github.com/bbuchsbaum/imfeatures/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bbuchsbaum/imfeatures/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

`imfeatures` extracts visual features from 2D images using deep learning
models and traditional computer vision techniques. It combines fast
R-native image descriptors with optional Python-backed deep-learning
features.

## What it does

- **Edge entropy**: `compute_edge_entropy()` measures first-order
  oriented-edge entropy and pairwise edge entropy across distance
  ranges. It works on grayscale matrices or image paths and does not
  require Python.
- **Multiscale color entropy**: `image_mse()` computes mean multiscale
  entropy for the `H`, `S`, and `V` channels of an `imager::cimg` image.
  This is also R-native.
- **VGG-16 features**: `extract_vgg_features()` extracts pooled low-,
  mid-, high-, or semantic-tier VGG-16 activations. `im_features()` and
  `im_feature_sim()` expose lower-level layer selection and similarity
  matrices.
- **Model zoo features**: `im_features_tv()` uses Python `thingsvision`
  for torchvision, timm, SSL, CLIP, and related models.
- **Vision-language features**: `caption_features()` can generate
  captions and optional caption embeddings through `ellmer` providers.

## Install

``` r
# install.packages("devtools")
devtools::install_github("bbuchsbaum/imfeatures")
```

Python is only needed for the deep-learning and vision-language
backends. The entropy functions can be used after installing the R
package.

## Quick Examples

Compute edge entropy for a grayscale matrix:

``` r
library(imfeatures)

set.seed(1)
img <- matrix(runif(96 * 96), nrow = 96)

edge <- compute_edge_entropy(img)
edge[, c("entropy", "pentropy_20_80", "pentropy_80_160", "pentropy_160_240")]
```

Compute multiscale HSV entropy for an `imager` image:

``` r
library(imfeatures)
library(imager)

img <- load.example("parrots")
image_mse(img)
```

Extract VGG-16 features from your own image directory:

``` r
library(imfeatures)

imgs <- list.files(
  "path/to/images",
  pattern = "\\.(jpe?g|png)$",
  full.names = TRUE,
  ignore.case = TRUE
)
stopifnot(length(imgs) >= 2)

vgg <- extract_vgg_features(imgs, tier = "mid", pooling = "avg")
dim(vgg$features)
```

Compute pairwise feature similarities from selected VGG-16 layers:

``` r
library(imfeatures)

sim <- im_feature_sim(
  imgs,
  layers = c("block3_conv3", "block4_conv3"),
  spatial_pooling = "avg",
  lowmem = FALSE
)

sim$layer_block3_conv3
```

Generate captions or caption embeddings when an `ellmer` provider is
configured:

``` r
library(imfeatures)

captions <- caption_features(
  imgs,
  caption_provider = "openai",
  template = "dense",
  compute_embedding = TRUE,
  embedding_backend = "openai"
)
```

## Python Setup

For workstation use, `imfeatures_config()` creates or activates the
Python environment used by Keras, `thingsvision`, PyTorch, and
`open_clip`:

``` r
library(imfeatures)
imfeatures_config()
```

If you already manage Python yourself, point `reticulate` at that
interpreter:

``` r
library(imfeatures)
use_existing_python("/full/path/to/python")
```

## HPC Setup

On HPC systems, avoid automatic Conda setup during package load. The
usual pattern is:

``` r
Sys.setenv(IMFEATURES_SKIP_PYTHON = "TRUE")
library(imfeatures)
use_existing_python("/full/path/to/venvs/imfeatures/bin/python")
```

For a persistent setup, put both variables in `~/.Renviron`:

``` text
IMFEATURES_SKIP_PYTHON=TRUE
RETICULATE_PYTHON=/full/path/to/venvs/imfeatures/bin/python
```

Install the Python packages into a cluster-managed module or virtual
environment, not an automatically created R miniconda. At minimum the
Python side needs `Pillow` and `numpy`; model backends add packages such
as `tensorflow`/`keras`, `torch`, `torchvision`, `thingsvision`,
`resmem`, and `open-clip-torch`.

The full HPC walkthrough is in:

``` r
vignette("hpc-setup", package = "imfeatures")
```

## Documentation

- Website: <https://bbuchsbaum.github.io/imfeatures/>
- Setup vignette:
  `vignette("five-minute-setup", package = "imfeatures")`
- HPC vignette: `vignette("hpc-setup", package = "imfeatures")`
