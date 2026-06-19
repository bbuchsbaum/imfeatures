# Get recommended layers for feature extraction

Returns a character vector of recommended layer names for a given model,
useful when extracting features from multiple layers.

## Usage

``` r
get_recommended_layers(model_name, source = "torchvision", level = "high")
```

## Arguments

- model_name:

  Character string. The name of the model architecture.

- source:

  Character string. The source library of the model.

- level:

  Character string. Level of features: "high" (default), "multi", or
  "all". - "high": Returns the single best high-level layer - "multi":
  Returns 2-3 layers at different depths - "all": Returns many layers
  for comprehensive analysis

## Value

Character vector of module names.

## Examples

``` r
if (FALSE) { # \dontrun{
get_recommended_layers("resnet50", level = "high")   # "avgpool"
get_recommended_layers("resnet50", level = "multi")  # c("layer3", "layer4", "avgpool")
} # }
```
