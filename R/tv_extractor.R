#' Get a thingsvision extractor object
#'
#' This function wraps the `get_extractor` function from the Python `thingsvision`
#' library, allowing you to instantiate a feature extractor for a wide variety
#' of computer vision models.
#'
#' @param model_name Character string. The name of the model you want to use.
#'        See Details for examples based on the source.
#' @param source Character string. The library or source from which the model originates.
#'        Must be one of "torchvision", "timm", "keras", "ssl", or "custom".
#' @param device Character string. The compute device to use, e.g., "cpu", "cuda",
#'        or "cuda:0" for the first GPU. Defaults to "cuda" if available, otherwise
#'        reticulate might fall back to "cpu".
#' @param pretrained Logical. Whether to load pretrained weights for the model.
#'        Defaults to TRUE. Pretrained weights are typically from ImageNet or the
#'        dataset specified in the model's original publication (e.g., LAION for OpenCLIP).
#' @param model_parameters Named list (optional). Additional parameters required by
#'        certain models, especially those from the "custom" or "ssl" source. See Details.
#'
#' @details
#' The combination of `model_name` and `source` determines which model is loaded.
#' Here's a guide to common options:
#'
#' \strong{Sources and Example Models:}
#'
#' \itemize{
#'   \item \strong{`source = "torchvision"`}: Accesses models from PyTorch's `torchvision.models`.
#'     \itemize{
#'       \item Common `model_name` examples: `"alexnet"`, `"vgg16"`, `"resnet18"`, `"resnet50"`, `"vit_b_16"`
#'       \item Pretrained weights are typically ImageNet-1k.
#'       \item `model_parameters`: Can sometimes be used to specify specific weights, e.g., `list(weights = 'IMAGENET1K_V2')` for ResNet50, though `"DEFAULT"` is often sufficient. See torchvision docs for available weights per model.
#'     }
#'   \item \strong{`source = "timm"`}: Accesses models from the `pytorch-image-models` library (a very extensive collection).
#'     \itemize{
#'       \item Common `model_name` examples: `"efficientnet_b0"`, `"convnext_tiny"`, `"vit_base_patch16_224"`, `"resnet50"`
#'       \item Find available models via `timm` documentation or `timm.list_models()` in Python.
#'       \item `model_parameters`: Usually not needed for basic extraction.
#'     }
#'   \item \strong{`source = "keras"`}: Accesses models from `tensorflow.keras.applications`.
#'     \itemize{
#'       \item Common `model_name` examples: `"VGG16"`, `"ResNet50"`, `"InceptionV3"`, `"EfficientNetB0"` (Note: often capitalized).
#'       \item Pretrained weights are typically ImageNet-1k.
#'       \item `model_parameters`: Usually not needed.
#'     }
#'   \item \strong{`source = "ssl"`}: Accesses Self-Supervised Learning models.
#'     \itemize{
#'       \item ResNet50 variants: `"simclr-rn50"`, `"mocov2-rn50"`, `"barlowtwins-rn50"`, `"vicreg-rn50"`, `"swav-rn50"`, etc.
#'       \item DINO Vision Transformers: `"dino-vit-small-p8"`, `"dino-vit-base-p16"`, etc.
#'       \item DINOv2 Vision Transformers: `"dinov2-vit-small-p14"`, `"dinov2-vit-base-p14"`, etc.
#'       \item MAE Vision Transformers: `"mae-vit-base-p16"`, `"mae-vit-large-p16"`, etc.
#'       \item `model_parameters`: **Important for ViT models (DINO, MAE)!** Use `list(token_extraction = ...)` to specify how to handle output tokens. Options are:
#'         \itemize{
#'           \item `"cls_token"`: Use only the [CLS] token output.
#'           \item `"avg_pool"`: Average pool the patch tokens (excluding [CLS]).
#'           \item `"cls_token+avg_pool"`: Concatenate the [CLS] token and the averaged patch tokens.
#'         }
#'     }
#'   \item \strong{`source = "custom"`}: Accesses models specifically packaged or handled by `thingsvision`.
#'     \itemize{
#'       \item Official CLIP: `model_name = "clip"`. Requires `model_parameters = list(variant = "ViT-B/32")` or `"RN50"`, etc. Needs `pip install git+https://github.com/openai/CLIP.git` in the Python env.
#'       \item OpenCLIP: `model_name = "OpenCLIP"`. Requires `model_parameters = list(variant = "ViT-B-32", dataset = "laion2b_s34b_b79k")`, etc. Check OpenCLIP repo for available variant/dataset pairs.
#'       \item CORnet: `model_name = "cornet_s"`, `"cornet_r"`, `"cornet_rt"`, `"cornet_z"`. Recurrent vision models.
#'       \item Ecoset Trained Models: `model_name = "Alexnet_ecoset"`, `"VGG16_ecoset"`, `"Resnet50_ecoset"`, `"Inception_ecoset"`. Trained on Ecoset dataset.
#'       \item Harmonization Models: `model_name = "Harmonization"`. Requires `model_parameters = list(variant = "ViT_B16")` or `"ResNet50"`, etc. Needs extra installation steps (see `install_thingsvision()` docs or thingsvision README).
#'       \item DreamSim Models: `model_name = "DreamSim"`. Requires `model_parameters = list(variant = "open_clip_vitb32")` or `"clip_vitb32"`, etc. Needs `pip install dreamsim==0.1.2` in the Python env.
#'       \item Segment Anything (SAM): `model_name = "SegmentAnything"`. Requires `model_parameters = list(variant = "vit_h")` or `"vit_l"`, `"vit_b"`.
#'       \item Kakaobrain ALIGN: `model_name = "Kakaobrain_Align"`.
#'     }
#' }
#'
#' \strong{`model_parameters` Argument:}
#' This R `list` is converted to a Python dictionary and passed to the underlying
#' `thingsvision` or model loading function. It's essential for models where just
#' the `model_name` isn't enough, like specifying variants (`"ViT-B/32"` for CLIP),
#' training datasets (`"laion2b_s34b_b79k"` for OpenCLIP), or special extraction
#' methods (`token_extraction` for DINO/MAE ViTs).
#'
#' \strong{Return Value:}
#' The function returns a `reticulate` Python object. This object is a wrapper
#' around the Python `thingsvision` extractor instance. You will pass this object
#' to other functions like `tv_extract_features()` or `tv_show_model()`.
#'
#' \strong{Finding Models:}
#' For the most up-to-date and comprehensive list of models available through
#' `torchvision`, `timm`, `keras`, and `ssl`, please refer to their respective
#' documentations. For `custom` models, refer to the `thingsvision` documentation:
#' \url{https://vicco-group.github.io/thingsvision/AvailableModels.html}
#'
#' @return A reticulate Python object reference to the configured thingsvision extractor.
#' @export
#' @seealso \code{\link{install_thingsvision}}, \code{\link{tv_extract_features}}, \code{\link{tv_show_model}}
#' @examples
#' \dontrun{
#' # Ensure Python env is configured first, e.g. after install_thingsvision()
#' # reticulate::use_condaenv("r-thingsvision", required = TRUE)
#'
#' # Example 1: ResNet-18 from Torchvision
#' extractor_rn18 <- tv_get_extractor(model_name = "resnet18", source = "torchvision")
#' # tv_show_model(extractor_rn18)
#'
#' # Example 2: CLIP ViT-B/32 from Custom
#' extractor_clip <- tv_get_extractor(
#'    model_name = "clip",
#'    source = "custom",
#'    model_parameters = list(variant = "ViT-B/32")
#' )
#' # tv_show_model(extractor_clip)
#'
#' # Example 3: DINO ViT Base/16 from SSL (using cls_token)
#' extractor_dino <- tv_get_extractor(
#'    model_name = "dino-vit-base-p16",
#'    source = "ssl",
#'    model_parameters = list(token_extraction = "cls_token")
#' )
#' # tv_show_model(extractor_dino)
#'
#' # Example 4: Timm EfficientNet B0
#' extractor_effnet <- tv_get_extractor(model_name = "efficientnet_b0", source = "timm")
#' # tv_show_model(extractor_effnet)
#' }
#' Get a thingsvision extractor R object
#'
#' This function instantiates a feature extractor from the Python `thingsvision`
#' library and wraps it in an R object of class `thingsvision_extractor` for
#' easier use within R.
#'
#' @param model_name Character string. The name of the model.
#' @param source Character string. The source library ("torchvision", "timm", etc.).
#' @param device Character string. Compute device ("cpu", "cuda", etc.).
#' @param pretrained Logical. Use pretrained weights?
#' @param model_parameters Named list (optional). Model-specific parameters.
#'
#' @details (Keep the detailed documentation about models/sources/params as before)
#' ...
#'
#' \strong{Return Value:}
#' Returns an R object of class `thingsvision_extractor`. This object encapsulates
#' the underlying Python extractor and provides R methods (like `print`, `extract`,
#' `align`) for interaction. Use this object with functions designed for it.
#'
#' @return An R object of class `thingsvision_extractor`.
#' @export
#' @seealso \code{\link{install_thingsvision}}, \code{\link{extract.thingsvision_extractor}}, \code{\link{print.thingsvision_extractor}}, \code{\link{align.thingsvision_extractor}}
#' @examples
#' \dontrun{
#' # reticulate::use_condaenv("r-thingsvision", required = TRUE)
#' extractor_rn18 <- tv_get_extractor(model_name = "resnet18", source = "torchvision", device="cpu")
#' print(extractor_rn18)
#' }
tv_get_extractor <- function(model_name, source, device = "cuda", pretrained = TRUE, model_parameters = NULL) {
  # Ensure tv (main thingsvision module) is loaded via reticulate
  if (is.null(tv) || reticulate::py_is_null_xptr(tv)) {
     stop("Python 'thingsvision' module not imported. Did you run configure_thingsvision_python() or install_thingsvision() and configure reticulate (e.g., use_condaenv)?")
  }

  source <- match.arg(source, c("torchvision", "timm", "keras", "ssl", "custom"))

  py_model_params <- if (!is.null(model_parameters)) {
                         if (!is.list(model_parameters) || is.null(names(model_parameters))) {
                            warning("'model_parameters' should be a named list. Attempting conversion.")
                         }
                         reticulate::r_to_py(model_parameters)
                       } else {
                         NULL
                       }

  py_extractor <- tryCatch({
    tv$get_extractor(
      model_name = model_name,
      source = source,
      device = device,
      pretrained = pretrained,
      model_parameters = py_model_params
    )
  }, error = function(e) {
    stop("Failed to get Python thingsvision extractor for model '", model_name, "' from source '", source, "'.\nPython error: ", e$message)
  })

  if (reticulate::py_is_null_xptr(py_extractor)) {
      stop("tv$get_extractor returned a NULL object. Check model_name, source, and parameters.")
  }

  # Store key info and the python object in the R object's list structure
  extractor_r <- structure(
    list(
      py_obj = py_extractor,
      model_name = model_name,
      source = source,
      device = py_extractor$device # Get actual device used by Python obj
      # Add other relevant info if needed
    ),
    class = "thingsvision_extractor"
  )

  return(extractor_r)
}

# Define a generic for extract if one doesn't exist that fits
# if (!exists("extract")) {
#   extract <- function(object, ...) {
#     UseMethod("extract")
#   }
# }
# Or use a more specific name like tv_extract to avoid conflicts
tv_extract <- function(object, ...) {
  UseMethod("tv_extract")
}
# Define generic for align
# if (!exists("align")) {
#   align <- function(object, ...) {
#     UseMethod("align")
#   }
# }
# Or use tv_align
tv_align <- function(object, ...) {
  UseMethod("tv_align")
}


#' Print method for thingsvision_extractor objects
#'
#' Displays basic information about the configured thingsvision extractor,
#' including the model name, source, and device. For full architecture,
#' use `show_model()`.
#'
#' @param x An object of class `thingsvision_extractor`.
#' @param ... Additional arguments (ignored).
#' @export
#' @method print thingsvision_extractor
print.thingsvision_extractor <- function(x, ...) {
  cat("--- thingsvision Extractor (R Object) ---\n")
  cat("  Model Name: ", x$model_name, "\n")
  cat("  Source:     ", x$source, "\n")
  cat("  Device:     ", x$device, "\n")
  cat("---------------------------------------\n")
  cat("Use `show_model(extractor)` to view architecture details.\n")
  invisible(x)
}

#' Show Model Architecture for thingsvision_extractor
#'
#' Prints the architecture of the underlying Python model, showing available
#' layers/modules for feature extraction.
#'
#' @param object An object of class `thingsvision_extractor`.
#' @param ... Additional arguments (ignored).
#' @return Invisibly returns the input object.
#' @export
show_model <- function(object, ...) {
   UseMethod("show_model")
}

#' @export
#' @method show_model thingsvision_extractor
show_model.thingsvision_extractor <- function(object, ...) {
  if (reticulate::py_is_null_xptr(object$py_obj)) {
     stop("The underlying Python extractor object is NULL.")
  }
  # Capture Python output
  output <- reticulate::py_capture_output({
     print(object$py_obj$show_model())
  })
  cat("--- Model Architecture (from Python) ---\n")
  cat(output)
  cat("---------------------------------------\n")
  invisible(object)
}


#' Get Preprocessing Transformations
#'
#' Retrieves the image preprocessing function (as a Python function object)
#' associated with the extractor. This is typically used internally by
#' dataset creation functions.
#'
#' @param object An object of class `thingsvision_extractor`.
#' @param ... Arguments passed to the underlying `get_transformations` Python method
#'        (e.g., `resize_dim`, `crop_dim`). Usually not needed as defaults are inferred.
#' @return A `reticulate` reference to the Python preprocessing callable.
#' @export
get_transformations <- function(object, ...) {
  UseMethod("get_transformations")
}

#' @export
#' @method get_transformations thingsvision_extractor
get_transformations.thingsvision_extractor <- function(object, ...) {
   if (reticulate::py_is_null_xptr(object$py_obj)) {
     stop("The underlying Python extractor object is NULL.")
   }
  return(object$py_obj$get_transformations(...))
}

#' Get Backend Name
#'
#' Returns the backend ('pt' for PyTorch or 'tf' for TensorFlow) of the
#' underlying Python model.
#'
#' @param object An object of class `thingsvision_extractor`.
#' @param ... Additional arguments (ignored).
#' @return Character string ("pt" or "tf").
#' @export
get_backend <- function(object, ...) {
   UseMethod("get_backend")
}

#' @export
get_backend.thingsvision_extractor <- function(object, ...) {
   if (reticulate::py_is_null_xptr(object$py_obj)) {
     stop("The underlying Python extractor object is NULL.")
   }
   return(object$py_obj$get_backend())
}


#' Extract features using a thingsvision_extractor object
#'
#' This method uses the configured extractor to extract features from the
#' provided data loader.
#'
#' @param object An object of class `thingsvision_extractor`.
#' @param dataloader A `reticulate` reference to a Python `thingsvision.DataLoader` object,
#'        typically created using \code{\link{tv_create_dataloader}}.
#' @param module_name Character string. The layer/module name to extract from.
#' @param flatten_acts Logical. Flatten activations?
#' @param output_type Character string ("ndarray" or "tensor"). The desired Python output type
#'        before conversion to R. Defaults to "ndarray".
#' @param output_dir Character string (optional). Directory to save features iteratively.
#' @param step_size Integer (optional). Step size for saving if `output_dir` is used. Must be a finite numeric scalar.
#' @param ... Additional arguments (currently ignored).
#'
#' @return An R matrix or array containing the features, or `NULL` invisibly if
#'         `output_dir` is specified.
#' @export
#' @method tv_extract thingsvision_extractor
tv_extract.thingsvision_extractor <- function(object, dataloader, module_name, flatten_acts = FALSE, output_type = "ndarray", output_dir = NULL, step_size = NULL, ...) {

   if (reticulate::py_is_null_xptr(object$py_obj)) {
     stop("The underlying Python extractor object is NULL.")
   }
   # Check if dataloader seems like a reticulate object
   if (!inherits(dataloader, "python.builtin.object")) {
      warning("'dataloader' does not appear to be a reticulate Python object reference.")
   }

   if (!is.null(step_size)) {
      if (!is.numeric(step_size) || length(step_size) != 1) {
         stop("'step_size' must be a numeric scalar if provided.")
      }
      if (!is.finite(step_size)) {
         stop("'step_size' must be a finite number.")
      }
      py_step_size <- as.integer(step_size)
   } else {
      py_step_size <- NULL
   }

   features_py <- tryCatch({
      object$py_obj$extract_features(
        batches = dataloader,
        module_name = module_name,
        flatten_acts = flatten_acts,
        output_type = output_type,
        output_dir = output_dir,
        step_size = py_step_size
      )
   }, error = function(e) {
      stop("Python feature extraction failed for module '", module_name, "':\n", e$message)
   })


   # Handle return based on output_dir
   if (!is.null(output_dir)) {
     message("Features saved iteratively to: ", output_dir)
     return(invisible(NULL))
   }

   # Convert Python result back to R
   if (reticulate::py_is_null_xptr(features_py)) {
        warning("Python extraction returned NULL features for module '", module_name, "'.")
        return(NULL)
   }

   features_r <- reticulate::py_to_r(features_py)

   if (!is.matrix(features_r) && !is.array(features_r)) {
        warning("Conversion from Python resulted in a non-matrix/array R object. Check output.")
   }
   return(features_r)
}

#' Align features using a thingsvision_extractor object
#'
#' Applies alignment transformations (e.g., gLocal) to features using the
#' extractor's `align` method.
#'
#' @param object An object of class `thingsvision_extractor`.
#' @param features An R matrix or array of features (will be converted to Python).
#' @param module_name The module name corresponding to the features being aligned.
#' @param alignment_type Character string. The alignment method (e.g., "gLocal"). Must be a character scalar. A warning is issued if the type is not recognized.
#' @param ... Additional arguments (currently ignored).
#'
#' @return Aligned features as an R matrix or array.
#' @export
#' @method tv_align thingsvision_extractor
tv_align.thingsvision_extractor <- function(object, features, module_name, alignment_type = "gLocal", ...) {
   if (reticulate::py_is_null_xptr(object$py_obj)) {
     stop("The underlying Python extractor object is NULL.")
   }
   if (!inherits(features, c("matrix", "array"))) {
      stop("'features' must be an R matrix or array.")
   }
   if (!is.character(alignment_type) || length(alignment_type) != 1) {
       stop("'alignment_type' must be a character scalar.")
   }
   known_types <- c("gLocal")
   if (!(alignment_type %in% known_types)) {
       warning("Unknown alignment_type '", alignment_type, "'.")
   }

   features_py <- reticulate::r_to_py(features)

   aligned_features_py <- tryCatch({
      object$py_obj$align(
         features = features_py,
         module_name = module_name,
         alignment_type = alignment_type
      )
   }, error = function(e) {
      stop("Python feature alignment failed for module '", module_name, "':\n", e$message)
   })


   if (reticulate::py_is_null_xptr(aligned_features_py)) {
        warning("Python alignment returned NULL features for module '", module_name, "'.")
        return(NULL)
   }

   aligned_features_r <- reticulate::py_to_r(aligned_features_py)
   return(aligned_features_r)
}

#' Create a thingsvision ImageDataset
#' @param root Path to the common root directory for images. File names passed to
#'   the Python dataset should be relative to this directory.
#' @param out_path Path for storing file order list
#' @param extractor An R object of class `thingsvision_extractor`. # MODIFIED
#' @param transforms Optional Python transforms object (usually get from extractor)
#' @param ... Additional arguments for ImageDataset (e.g., class_names, file_names)
#' @return A reticulate Python object reference to the ImageDataset
#' @export
tv_create_dataset <- function(root, out_path, extractor, transforms = NULL, ...) {
  # Input check for R object type
  if (!inherits(extractor, "thingsvision_extractor")) {
      stop("'extractor' must be an object of class 'thingsvision_extractor'.")
  }
  if (is.null(tv_data)) { stop("thingsvision.utils.data not imported.") }

  # Access underlying python object for its methods
  py_extractor <- extractor$py_obj
  if (reticulate::py_is_null_xptr(py_extractor)) {
     stop("The underlying Python extractor object is NULL in the provided R object.")
  }

  if (is.null(transforms)) {
    transforms <- py_extractor$get_transformations() # Call method on Python object
  }

  dataset <- tv_data$ImageDataset(
    root = root,
    out_path = out_path,
    backend = py_extractor$get_backend(), # Call method on Python object
    transforms = transforms,
    ...
  )
  return(dataset)
}


#' Create a thingsvision ImageDataLoader
#' @param dataset A thingsvision ImageDataset object
#' @param batch_size Integer batch size
#' @param extractor An R object of class `thingsvision_extractor`
#' @param ... Additional arguments for ImageDataLoader (e.g., shuffle, num_workers)
#' @return A reticulate Python object reference to the ImageDataLoader
#' @export
tv_create_dataloader <- function(dataset, batch_size, extractor, ...) {
  if (!inherits(extractor, "thingsvision_extractor")) {
    stop("'extractor' must be an object of class 'thingsvision_extractor'.")
  }
  if (is.null(tv_data)) {
    stop("thingsvision.utils.data not imported.")
  }

  py_extractor <- extractor$py_obj
  if (reticulate::py_is_null_xptr(py_extractor)) {
    stop("The underlying Python extractor object is NULL in the provided R object.")
  }


  dl <- tv_data$ImageDataLoader(
    dataset = dataset,
    batch_size = as.integer(batch_size),
    backend = py_extractor$get_backend(),
    ...
  )

  return(dl)
}
