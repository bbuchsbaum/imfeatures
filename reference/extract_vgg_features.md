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
  pooling = "avg",
  batch_size = 8L
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

  Numeric vector of length 2 specifying image resize dimensions (height,
  width).

- pooling:

  Character string specifying how each layer's feature map is reduced
  over height and width before the layers are concatenated: `"avg"`
  (default, global average pooling), `"max"`, `"none"`, or
  `"resize_HxW"` such as `"resize_3x3"`. The options are those of the
  `spatial_pooling` argument of
  [`extract_features`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features.md);
  see its Spatial pooling section. The dense layers of the `"semantic"`
  tier have no spatial axes and are unaffected.

- batch_size:

  Integer; number of images per forward pass. Images are pushed through
  a single multi-output model in batches rather than one at a time,
  which is substantially faster for large image sets. Defaults to 8 to
  bound peak memory use for early VGG layers.

## Value

An S3 object of class `vgg_feature_set`, a list with components:

- features:

  Numeric matrix with one row per image and one column per pooled
  feature (see Feature dimensions).

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

- batch_size:

  Integer number of images processed per forward pass.

## Feature dimensions

The same pooling is applied to every layer in the tier, and the pooled
layers are concatenated in network order. With 224 x 224 input the
number of columns in `features` is:

|              |                        |                    |              |
|--------------|------------------------|--------------------|--------------|
| **Tier**     | **`"avg"` or `"max"`** | **`"resize_3x3"`** | **`"none"`** |
| `"low"`      | 384                    | 3,456              | 9,633,792    |
| `"mid"`      | 2,304                  | 20,736             | 3,612,672    |
| `"high"`     | 1,536                  | 13,824             | 301,056      |
| `"semantic"` | 8,192                  | 8,192              | 8,192        |

`"none"` keeps every position of every layer and is rarely practical
beyond a few images. To pool each layer differently, or to use grid, DCT
or radial pooling, use
[`extract_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md).

## See also

[`extract_features`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features.md)
for single layers,
[`extract_feature_hierarchy`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md)
for per-stage poolers, and
[`vignette("spatial-pooling", package = "imfeatures")`](https://bbuchsbaum.github.io/imfeatures/articles/spatial-pooling.md).

## Examples

``` r
if (FALSE) { # \dontrun{
imgs <- list.files("stimuli", pattern = "\\.jpg$", full.names = TRUE)

# One value per channel: 2,304 columns for the mid tier
mid_avg <- extract_vgg_features(imgs, tier = "mid", pooling = "avg")

# Keep a coarse 3 x 3 layout: 13,824 columns for the high tier
high_3x3 <- extract_vgg_features(imgs, tier = "high", pooling = "resize_3x3")
dim(high_3x3$features)
} # }
```
