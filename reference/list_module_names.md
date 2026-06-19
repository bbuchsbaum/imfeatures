# List available module names for a model

Convenience function to discover available layer/module names for
feature extraction from a given model. This is useful for finding which
layers you can extract features from.

## Usage

``` r
list_module_names(
  model_name,
  source = "torchvision",
  device = "cpu",
  pretrained = TRUE,
  model_parameters = NULL
)
```

## Arguments

- model_name:

  Character string. The name of the model architecture.

- source:

  Character string. The source library of the model. Defaults to
  "torchvision".

- device:

  Character string. The compute device ("cpu", "cuda", "cuda:0").
  Defaults to "cpu" for quick inspection.

- pretrained:

  Logical. Use pretrained model weights? Defaults to TRUE.

- model_parameters:

  Named list (optional). Additional parameters for specific models
  (e.g., list(variant = "ViT-B/32") for CLIP).

## Value

Invisibly returns the extractor object. The module architecture is
printed to the console, showing all available module names.

## Details

This function creates a model extractor and displays its architecture,
allowing you to see all available module/layer names that can be used
with functions like \`extract_features_tv()\` and
\`compute_feature_similarity_tv()\`.

The output shows the model's hierarchical structure with module names
that can be used for feature extraction. Look for layers like: - Conv2d
layers for early visual features - BatchNorm layers (often paired with
conv layers) - ReLU/GELU activation layers - Pooling layers (MaxPool,
AvgPool) - Linear/Dense layers for high-level features - Special layers
like "avgpool", "features", "classifier"

## See also

[`get_default_module_name`](https://bbuchsbaum.github.io/imfeatures/reference/get_default_module_name.md),
[`get_recommended_layers`](https://bbuchsbaum.github.io/imfeatures/reference/get_recommended_layers.md),
[`tv_get_extractor`](https://bbuchsbaum.github.io/imfeatures/reference/tv_get_extractor.md),
[`show_model`](https://bbuchsbaum.github.io/imfeatures/reference/show_model.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Show available layers for ResNet50
list_module_names("resnet50")

# Show layers for a CLIP model
list_module_names("clip", model_parameters = list(variant = "ViT-B/32"))

# Show layers for a Vision Transformer from timm
list_module_names("vit_base_patch16_224", source = "timm")
} # }
```
