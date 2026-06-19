# Extract VGG-16 features by tier

Convenience wrapper around
[`im_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features.md)
to extract VGG-16 features grouped by spatial tiers:

- `"low"`: conv1_1, conv1_2, conv2_1, conv2_2

- `"mid"`: conv3_1 through conv4_3

- `"high"`: conv5_1 through conv5_3

- `"semantic"`: fc1 (fc6) and fc2 (fc7)

Layers are retrieved by name (e.g., `"block1_conv1"`) instead of numeric
indices.

## Usage

``` r
extract_vgg_features(
  impaths,
  tier = c("low", "mid", "high", "semantic"),
  model = NULL,
  target_size = c(224, 224),
  pooling = "avg"
)
```

## Arguments

- impaths:

  Character vector of image file paths.

- tier:

  Character; one of "low", "mid", "high", or "semantic".

- model:

  Preloaded Keras VGG-16 model object. If NULL, defaults to
  `keras3::application_vgg16(weights = 'imagenet')`.

- target_size:

  Numeric vector of length 2 specifying image resize dimensions (width,
  height).

- pooling:

  Character string specifying spatial pooling; passed to the
  `spatial_pooling` argument of `im_features`. Defaults to "avg" (global
  average pooling). Other options: "none", "max", "resize_3x3",
  "resize_5x5", "resize_7x7".

## Value

An S3 object of class `vgg_feature_set`, a list with components:

- features:

  Numeric matrix (N_images × total_channels) of pooled features.

- image_paths:

  Character vector of input image paths.

- tier:

  The tier name.

- pooling:

  Pooling type used.

- layer_indices:

  Numeric indices of the selected layers (derived from `layer_names`).

- layer_names:

  Character names of VGG-16 layers used.

- model_name:

  Character, set to "vgg16".

- target_size:

  Numeric vector of image resize dimensions.
