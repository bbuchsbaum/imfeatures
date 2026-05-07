#' Get default module name for a model
#'
#' @description
#' Determines the appropriate default layer/module name for feature extraction
#' based on the model architecture. Generally returns the penultimate layer
#' (before the final classifier) which provides good semantic representations.
#'
#' @param model_name Character string. The name of the model architecture.
#' @param source Character string. The source library of the model.
#'
#' @return Character string with the default module name.
#'
#' @details
#' Default module names by architecture family:
#' - ResNet variants (resnet18, resnet50, etc.): "avgpool"
#' - VGG variants: "classifier.6" (penultimate FC layer)
#' - AlexNet: "classifier.6"
#' - DenseNet variants: "features"
#' - MobileNet variants: "classifier.0"
#' - EfficientNet variants: "avgpool"
#' - Vision Transformers (ViT): "pre_logits"
#' - CLIP models: "visual"
#' - DINOv2: "norm"
#' - ConvNeXt: "avgpool"
#' - Inception: "avgpool"
#' - SqueezeNet: "classifier.1"
#'
#' For unknown models, returns "avgpool" as a reasonable default.
#'
#' @examples
#' \dontrun{
#' get_default_module_name("resnet50") # Returns "avgpool"
#' get_default_module_name("vgg16") # Returns "classifier.6"
#' get_default_module_name("clip") # Returns "visual"
#' }
#'
#' @export
get_default_module_name <- function(model_name, source = "torchvision") {
  # Convert to lowercase for matching
  model_lower <- tolower(model_name)

  # Handle CLIP models
  if (grepl("clip", model_lower)) {
    return("visual")
  }

  # Handle ResNet family (includes Wide ResNet, ResNeXt)
  if (grepl("resnet|resnext|wide_resnet", model_lower)) {
    return("avgpool")
  }

  # Handle VGG family
  if (grepl("vgg", model_lower)) {
    return("classifier.6") # Penultimate FC layer
  }

  # Handle AlexNet
  if (grepl("alexnet", model_lower)) {
    return("classifier.6")
  }

  # Handle DenseNet family
  if (grepl("densenet", model_lower)) {
    return("features")
  }

  # Handle MobileNet family
  if (grepl("mobilenet", model_lower)) {
    return("classifier.0")
  }

  # Handle EfficientNet family
  if (grepl("efficientnet", model_lower)) {
    return("avgpool")
  }

  # Handle Vision Transformers
  if (grepl("vit_|vision_transformer|deit", model_lower)) {
    # For ViT models, the CLS token or pre_logits is typically best
    if (source == "timm") {
      return("pre_logits")
    } else {
      return("norm") # For torchvision ViT
    }
  }

  # Handle DINOv2
  if (grepl("dino", model_lower)) {
    return("norm")
  }

  # Handle ConvNeXt
  if (grepl("convnext", model_lower)) {
    return("avgpool")
  }

  # Handle Inception family
  if (grepl("inception", model_lower)) {
    return("avgpool")
  }

  # Handle SqueezeNet
  if (grepl("squeezenet", model_lower)) {
    return("classifier.1")
  }

  # Handle Swin Transformer
  if (grepl("swin", model_lower)) {
    return("avgpool")
  }

  # Handle RegNet
  if (grepl("regnet", model_lower)) {
    return("avgpool")
  }

  # Handle ShuffleNet
  if (grepl("shufflenet", model_lower)) {
    return("fc")
  }

  # Handle BERT-like vision models
  if (grepl("beit|deit", model_lower)) {
    return("pooler")
  }

  # Default fallback - avgpool is common in many architectures
  # Issue a message so users know a default was used
  message(
    "Using default module 'avgpool' for model '", model_name,
    "'. Use tv_show_model() to see available layers if this doesn't work."
  )
  return("avgpool")
}

#' Get recommended layers for feature extraction
#'
#' @description
#' Returns a character vector of recommended layer names for a given model,
#' useful when extracting features from multiple layers.
#'
#' @param model_name Character string. The name of the model architecture.
#' @param source Character string. The source library of the model.
#' @param level Character string. Level of features: "high" (default), "multi", or "all".
#'   - "high": Returns the single best high-level layer
#'   - "multi": Returns 2-3 layers at different depths
#'   - "all": Returns many layers for comprehensive analysis
#'
#' @return Character vector of module names.
#'
#' @examples
#' \dontrun{
#' get_recommended_layers("resnet50", level = "high") # "avgpool"
#' get_recommended_layers("resnet50", level = "multi") # c("layer3", "layer4", "avgpool")
#' }
#'
#' @export
get_recommended_layers <- function(model_name, source = "torchvision", level = "high") {
  model_lower <- tolower(model_name)

  if (level == "high") {
    # Just return the single best layer
    return(get_default_module_name(model_name, source))
  }

  # Multi-level extraction
  if (grepl("resnet|resnext", model_lower)) {
    if (level == "multi") {
      return(c("layer3", "layer4", "avgpool"))
    } else { # "all"
      return(c("layer1", "layer2", "layer3", "layer4", "avgpool"))
    }
  }

  if (grepl("vgg", model_lower)) {
    if (level == "multi") {
      return(c("features.20", "features.30", "classifier.6"))
    } else { # "all"
      # Return key conv layers and FC layers
      return(c(
        "features.5", "features.10", "features.17",
        "features.24", "features.30", "classifier.3", "classifier.6"
      ))
    }
  }

  if (grepl("densenet", model_lower)) {
    if (level == "multi") {
      return(c("features.denseblock3", "features.denseblock4", "features"))
    } else {
      return(c(
        "features.denseblock1", "features.denseblock2",
        "features.denseblock3", "features.denseblock4", "features"
      ))
    }
  }

  if (grepl("efficientnet", model_lower)) {
    if (level == "multi") {
      return(c("features.6", "features.7", "avgpool"))
    } else {
      return(c("features.3", "features.5", "features.6", "features.7", "avgpool"))
    }
  }

  # For unknown models or when in doubt, return the default
  return(get_default_module_name(model_name, source))
}

#' List available module names for a model
#'
#' @description
#' Convenience function to discover available layer/module names for feature
#' extraction from a given model. This is useful for finding which layers
#' you can extract features from.
#'
#' @param model_name Character string. The name of the model architecture.
#' @param source Character string. The source library of the model.
#'   Defaults to "torchvision".
#' @param device Character string. The compute device ("cpu", "cuda", "cuda:0").
#'   Defaults to "cpu" for quick inspection.
#' @param pretrained Logical. Use pretrained model weights? Defaults to TRUE.
#' @param model_parameters Named list (optional). Additional parameters for
#'   specific models (e.g., list(variant = "ViT-B/32") for CLIP).
#'
#' @return Invisibly returns the extractor object. The module architecture
#'   is printed to the console, showing all available module names.
#'
#' @details
#' This function creates a model extractor and displays its architecture,
#' allowing you to see all available module/layer names that can be used
#' with functions like `extract_features_tv()` and `compute_feature_similarity_tv()`.
#'
#' The output shows the model's hierarchical structure with module names
#' that can be used for feature extraction. Look for layers like:
#' - Conv2d layers for early visual features
#' - BatchNorm layers (often paired with conv layers)
#' - ReLU/GELU activation layers
#' - Pooling layers (MaxPool, AvgPool)
#' - Linear/Dense layers for high-level features
#' - Special layers like "avgpool", "features", "classifier"
#'
#' @examples
#' \dontrun{
#' # Show available layers for ResNet50
#' list_module_names("resnet50")
#'
#' # Show layers for a CLIP model
#' list_module_names("clip", model_parameters = list(variant = "ViT-B/32"))
#'
#' # Show layers for a Vision Transformer from timm
#' list_module_names("vit_base_patch16_224", source = "timm")
#' }
#'
#' @seealso
#' \code{\link{get_default_module_name}}, \code{\link{get_recommended_layers}},
#' \code{\link{tv_get_extractor}}, \code{\link{show_model}}
#'
#' @export
list_module_names <- function(model_name,
                              source = "torchvision",
                              device = "cpu",
                              pretrained = TRUE,
                              model_parameters = NULL) {
  # Get the extractor
  extractor <- tv_get_extractor(
    model_name = model_name,
    source = source,
    device = device,
    pretrained = pretrained,
    model_parameters = model_parameters
  )

  cat("Available module names for", model_name, "from", source, ":\n")
  cat("=", rep("=", 60), "\n", sep = "")

  # Show the model architecture
  show_model(extractor)

  cat("\n", "=", rep("=", 60), "\n", sep = "")
  cat("Tips:\n")
  cat("- Default module for this model: '",
    get_default_module_name(model_name, source), "'\n",
    sep = ""
  )
  cat("- Use any of the module names shown above with extract_features_tv()\n")
  cat("- Deeper layers (e.g., 'avgpool', 'classifier') give more semantic features\n")
  cat("- Earlier layers (e.g., 'conv1', 'features.0') give more visual/texture features\n")

  invisible(extractor)
}
