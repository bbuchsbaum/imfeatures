# Get default module name for a model

Determines the appropriate default layer/module name for feature
extraction based on the model architecture. Generally returns the
penultimate layer (before the final classifier) which provides good
semantic representations.

## Usage

``` r
get_default_module_name(model_name, source = "torchvision")
```

## Arguments

- model_name:

  Character string. The name of the model architecture.

- source:

  Character string. The source library of the model.

## Value

Character string with the default module name.

## Details

Default module names by architecture family: - ResNet variants
(resnet18, resnet50, etc.): "avgpool" - VGG variants: "classifier.6"
(penultimate FC layer) - AlexNet: "classifier.6" - DenseNet variants:
"features" - MobileNet variants: "classifier.0" - EfficientNet variants:
"avgpool" - Vision Transformers (ViT): "pre_logits" - CLIP models:
"visual" - DINOv2: "norm" - ConvNeXt: "avgpool" - Inception: "avgpool" -
SqueezeNet: "classifier.1"

For unknown models, returns "avgpool" as a reasonable default.

## Examples

``` r
if (FALSE) { # \dontrun{
get_default_module_name("resnet50")  # Returns "avgpool"
get_default_module_name("vgg16")     # Returns "classifier.6"
get_default_module_name("clip")      # Returns "visual"
} # }
```
