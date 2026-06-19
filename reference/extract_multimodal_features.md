# Extract multimodal features (visual + caption)

Extract multimodal features (visual + caption)

## Usage

``` r
extract_multimodal_features(
  impath,
  visual_layers = c(15, 17, 19),
  visual_model = NULL,
  caption_provider = "openai",
  caption_template = "dense",
  ...
)
```

## Arguments

- impath:

  Image file path(s).

- visual_layers:

  Layers for visual feature extraction.

- visual_model:

  Keras model for visual features.

- caption_provider:

  Provider for caption generation.

- caption_template:

  Template for captions.

- ...:

  Additional arguments for caption_features.

## Value

A tibble with both visual and caption features.
