# Get a thingsvision extractor object

This function wraps the \`get_extractor\` function from the Python
\`thingsvision\` library, allowing you to instantiate a feature
extractor for a wide variety of computer vision models.

## Usage

``` r
tv_get_extractor(
  model_name,
  source,
  device = "cuda",
  pretrained = TRUE,
  model_parameters = NULL
)
```

## Arguments

- model_name:

  Character string. The name of the model you want to use. See Details
  for examples based on the source.

- source:

  Character string. The library or source from which the model
  originates. Must be one of "torchvision", "timm", "keras", "ssl", or
  "custom".

- device:

  Character string. The compute device to use, e.g., "cpu", "cuda", or
  "cuda:0" for the first GPU. Defaults to "cuda" if available, otherwise
  reticulate might fall back to "cpu".

- pretrained:

  Logical. Whether to load pretrained weights for the model. Defaults to
  TRUE. Pretrained weights are typically from ImageNet or the dataset
  specified in the model's original publication (e.g., LAION for
  OpenCLIP).

- model_parameters:

  Named list (optional). Additional parameters required by certain
  models, especially those from the "custom" or "ssl" source. See
  Details.

## Value

A reticulate Python object reference to the configured thingsvision
extractor.

## Details

The combination of \`model_name\` and \`source\` determines which model
is loaded. Here's a guide to common options:

**Sources and Example Models:**

- **\`source = "torchvision"\`**: Accesses models from PyTorch's
  \`torchvision.models\`.

  - Common \`model_name\` examples: \`"alexnet"\`, \`"vgg16"\`,
    \`"resnet18"\`, \`"resnet50"\`, \`"vit_b_16"\`

  - Pretrained weights are typically ImageNet-1k.

  - \`model_parameters\`: Can sometimes be used to specify specific
    weights, e.g., \`list(weights = 'IMAGENET1K_V2')\` for ResNet50,
    though \`"DEFAULT"\` is often sufficient. See torchvision docs for
    available weights per model.

- **\`source = "timm"\`**: Accesses models from the
  \`pytorch-image-models\` library (a very extensive collection).

  - Common \`model_name\` examples: \`"efficientnet_b0"\`,
    \`"convnext_tiny"\`, \`"vit_base_patch16_224"\`, \`"resnet50"\`

  - Find available models via \`timm\` documentation or
    \`timm.list_models()\` in Python.

  - \`model_parameters\`: Usually not needed for basic extraction.

- **\`source = "keras"\`**: Accesses models from
  \`tensorflow.keras.applications\`.

  - Common \`model_name\` examples: \`"VGG16"\`, \`"ResNet50"\`,
    \`"InceptionV3"\`, \`"EfficientNetB0"\` (Note: often capitalized).

  - Pretrained weights are typically ImageNet-1k.

  - \`model_parameters\`: Usually not needed.

- **\`source = "ssl"\`**: Accesses Self-Supervised Learning models.

  - ResNet50 variants: \`"simclr-rn50"\`, \`"mocov2-rn50"\`,
    \`"barlowtwins-rn50"\`, \`"vicreg-rn50"\`, \`"swav-rn50"\`, etc.

  - DINO Vision Transformers: \`"dino-vit-small-p8"\`,
    \`"dino-vit-base-p16"\`, etc.

  - DINOv2 Vision Transformers: \`"dinov2-vit-small-p14"\`,
    \`"dinov2-vit-base-p14"\`, etc.

  - MAE Vision Transformers: \`"mae-vit-base-p16"\`,
    \`"mae-vit-large-p16"\`, etc.

  - \`model_parameters\`: \*\*Important for ViT models (DINO, MAE)!\*\*
    Use \`list(token_extraction = ...)\` to specify how to handle output
    tokens. Options are:

    - \`"cls_token"\`: Use only the \[CLS\] token output.

    - \`"avg_pool"\`: Average pool the patch tokens (excluding \[CLS\]).

    - \`"cls_token+avg_pool"\`: Concatenate the \[CLS\] token and the
      averaged patch tokens.

- **\`source = "custom"\`**: Accesses models specifically packaged or
  handled by \`thingsvision\`.

  - Official CLIP: \`model_name = "clip"\`. Requires \`model_parameters
    = list(variant = "ViT-B/32")\` or \`"RN50"\`, etc. Needs \`pip
    install git+https://github.com/openai/CLIP.git\` in the Python env.

  - OpenCLIP: \`model_name = "OpenCLIP"\`. Requires \`model_parameters =
    list(variant = "ViT-B-32", dataset = "laion2b_s34b_b79k")\`, etc.
    Check OpenCLIP repo for available variant/dataset pairs.

  - CORnet: \`model_name = "cornet_s"\`, \`"cornet_r"\`,
    \`"cornet_rt"\`, \`"cornet_z"\`. Recurrent vision models.

  - Ecoset Trained Models: \`model_name = "Alexnet_ecoset"\`,
    \`"VGG16_ecoset"\`, \`"Resnet50_ecoset"\`, \`"Inception_ecoset"\`.
    Trained on Ecoset dataset.

  - Harmonization Models: \`model_name = "Harmonization"\`. Requires
    \`model_parameters = list(variant = "ViT_B16")\` or \`"ResNet50"\`,
    etc. Needs extra installation steps (see thingsvision README).

  - DreamSim Models: \`model_name = "DreamSim"\`. Requires
    \`model_parameters = list(variant = "open_clip_vitb32")\` or
    \`"clip_vitb32"\`, etc. Needs \`pip install dreamsim==0.1.2\` in the
    Python env.

  - Segment Anything (SAM): \`model_name = "SegmentAnything"\`. Requires
    \`model_parameters = list(variant = "vit_h")\` or \`"vit_l"\`,
    \`"vit_b"\`.

  - Kakaobrain ALIGN: \`model_name = "Kakaobrain_Align"\`.

**\`model_parameters\` Argument:** This R \`list\` is converted to a
Python dictionary and passed to the underlying \`thingsvision\` or model
loading function. It's essential for models where just the
\`model_name\` isn't enough, like specifying variants (\`"ViT-B/32"\`
for CLIP), training datasets (\`"laion2b_s34b_b79k"\` for OpenCLIP), or
special extraction methods (\`token_extraction\` for DINO/MAE ViTs).

**Return Value:** The function returns a \`reticulate\` Python object.
This object is a wrapper around the Python \`thingsvision\` extractor
instance. You will pass this object to other functions like
\`tv_extract()\` or \`show_model()\`.

**Finding Models:** For the most up-to-date and comprehensive list of
models available through \`torchvision\`, \`timm\`, \`keras\`, and
\`ssl\`, please refer to their respective documentations. For \`custom\`
models, refer to the \`thingsvision\` documentation:
<https://vicco-group.github.io/thingsvision/AvailableModels.html>

## See also

[`imfeatures_config`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md),
[`tv_extract`](https://bbuchsbaum.github.io/imfeatures/reference/tv_extract.md),
[`show_model`](https://bbuchsbaum.github.io/imfeatures/reference/show_model.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Ensure Python env is configured first, e.g. after imfeatures_config()
# reticulate::use_condaenv("r-thingsvision", required = TRUE)

# Example 1: ResNet-18 from Torchvision
extractor_rn18 <- tv_get_extractor(model_name = "resnet18", source = "torchvision")
# tv_show_model(extractor_rn18)

# Example 2: CLIP ViT-B/32 from Custom
extractor_clip <- tv_get_extractor(
   model_name = "clip",
   source = "custom",
   model_parameters = list(variant = "ViT-B/32")
)
# tv_show_model(extractor_clip)

# Example 3: DINO ViT Base/16 from SSL (using cls_token)
extractor_dino <- tv_get_extractor(
   model_name = "dino-vit-base-p16",
   source = "ssl",
   model_parameters = list(token_extraction = "cls_token")
)
# tv_show_model(extractor_dino)

# Example 4: Timm EfficientNet B0
extractor_effnet <- tv_get_extractor(model_name = "efficientnet_b0", source = "timm")
# tv_show_model(extractor_effnet)
} # }
```
