#' Extract CLIP embeddings (final or intermediate layers)
#'
#' @param impath  Path to image file (jpg/png).
#' @param layers  Character or integer vector. "final" (default) returns the
#'                CLIP image embedding. For intermediate transformer layers,
#'                provide full layer names (e.g., "visual.transformer.resblocks.6")
#'                or integer indices (0-based) representing transformer blocks.
#'                Other layer names (e.g., "visual.class_embedding", "visual.ln_post")
#'                can also be specified if known.
#' @param model_name  CLIP model string (e.g. "ViT-B-32", "RN50", "ViT-L-14").
#' @param num_transformer_blocks Integer, number of transformer blocks in the
#'        vision model. Defaults to 12 (e.g., for ViT-B). For ViT-L, this would be 24.
#'        Only relevant if integer indices are used for `layers`.
#' @param device  "cpu" or "cuda".
#' @return A named list of numeric arrays (one per requested layer).
#' @export
clip_features <- function(impath,
                          layers = "final",
                          model_name = "ViT-B-32",
                          num_transformer_blocks = 12, # Default for ViT-B
                          device = c("cpu", "cuda")) {

  device <- match.arg(device)

  if (!file.exists(impath)) {
    stop("Image file not found: ", impath)
  }

  # Python helper function string
  py_helpers <- "
import torch

def get_submodule_by_path(model, module_path_str):
    module = model
    try:
        for part in module_path_str.split('.'):
            if part.isdigit() and hasattr(module, '__getitem__') and not isinstance(module, torch.Tensor):
                module = module[int(part)]
            else:
                module = getattr(module, part)
    except (AttributeError, IndexError) as e:
        # Raise a more informative error that can be caught in R
        # Using .format() instead of f-string for R compatibility
        error_message = \"Failed to access submodule: '{}'. Part '{}' not found or index out of bounds. Error: {}\".format(module_path_str, part, str(e))
        raise type(e)(error_message)
    return module

# r_hooks_env is an R environment (passed as a Python dict by reticulate)
# key_for_r_env is the string key for storing
def register_hook_on_submodule_helper(model, module_path_str, r_hooks_env, key_for_r_env):
    module_to_hook = get_submodule_by_path(model, module_path_str)

    def hook_fn(module, input, output):
        activation_tensor = output
        if isinstance(output, tuple): # Handle cases where output is a tuple (e.g. from some transformer blocks)
            activation_tensor = output[0]
        r_hooks_env[key_for_r_env] = activation_tensor.detach().cpu().numpy()

    return module_to_hook.register_forward_hook(hook_fn)
"
  # Make helper available in Python main module
  reticulate::py_run_string(py_helpers)
  py_register_hook <- reticulate::py$register_hook_on_submodule_helper
  py_get_submodule <- reticulate::py$get_submodule_by_path


  # ------------------------------------------------------------------
  # 1.  Bridge to Python open_clip
  # ------------------------------------------------------------------
  oc    <- reticulate::import("open_clip", delay_load = TRUE)
  pil   <- reticulate::import("PIL.Image", delay_load = TRUE)
  torch <- reticulate::import("torch", delay_load = TRUE)

  # Corrected multiple assignment
  model_and_transforms <- oc$create_model_and_transforms(
    model_name,
    pretrained = "openai", # or other pretrained weights like 'laion2b_s34b_b79k'
    device = device
  )
  model      <- model_and_transforms[[1]]
  preprocess <- model_and_transforms[[2]]
  model$eval()

  # ------------------------------------------------------------------
  # 2.  Load & preprocess image
  # ------------------------------------------------------------------
  img   <- pil$open(impath)$convert("RGB")
  img_t <- preprocess(img)$unsqueeze(0L)$to(device = device) # Use 0L for integer

  # Results list
  out <- list()
  # R environment to store activations from hooks
  captured_activations_env <- new.env(parent = emptyenv())


  # ------------------------------------------------------------------
  # 3. Layer Processing & Hook Registration
  # ------------------------------------------------------------------
  needs_final_embedding <- "final" %in% layers
  intermediate_layers_requested <- setdiff(layers, "final")
  numeric_idx <- grepl("^[0-9]+$", intermediate_layers_requested)
  intermediate_layers_requested[numeric_idx] <-
    as.integer(intermediate_layers_requested[numeric_idx])

  # Generate standard block names if integer indices are used
  # Base path for vision transformer residual blocks
  vis_transformer_resblocks_base = "visual.transformer.resblocks"

  # Resolve all requested intermediate layers to their full module path strings
  resolved_intermediate_module_paths <- character(0)
  if (length(intermediate_layers_requested) > 0) {
    resolved_intermediate_module_paths <- sapply(intermediate_layers_requested, function(lyr) {
      if (is.numeric(lyr)) {
        if (lyr >= 0 && lyr < num_transformer_blocks) {
          return(sprintf("%s.%d", vis_transformer_resblocks_base, as.integer(lyr)))
        } else {
          warning(sprintf("Integer layer index %d is out of bounds (0-%d). Skipping.",
                          as.integer(lyr), num_transformer_blocks - 1))
          return(NA_character_)
        }
      } else if (is.character(lyr)) {
        # Assume it's a full module path string if character
        # Basic check to see if the module path is accessible (optional, can be slow)
        # tryCatch({ py_get_submodule(model, lyr) }, error = function(e) {
        #   warning(sprintf("Layer '%s' not found or accessible in model. Skipping. Error: %s", lyr, e$message))
        #   return(NA_character_)
        # })
        return(lyr)
      } else {
        warning(sprintf("Invalid layer specification: %s. Skipping.", lyr))
        return(NA_character_)
      }
    }, USE.NAMES = FALSE)
    resolved_intermediate_module_paths <- resolved_intermediate_module_paths[!is.na(resolved_intermediate_module_paths)]
    resolved_intermediate_module_paths <- unique(resolved_intermediate_module_paths) # Avoid duplicate hooks
  }

  hook_handles <- list()
  if (length(resolved_intermediate_module_paths) > 0) {
    message("Registering hooks for: ", paste(resolved_intermediate_module_paths, collapse=", "))
    for (module_path_str in resolved_intermediate_module_paths) {
      tryCatch({
        # The key for storing in r_hooks_env will be the module_path_str itself
        handle <- py_register_hook(model, module_path_str, captured_activations_env, module_path_str)
        hook_handles[[module_path_str]] <- handle
      }, error = function(e) {
        warning(sprintf("Failed to register hook for layer '%s'. Error: %s", module_path_str, e$message))
      })
    }
  }

  # ------------------------------------------------------------------
  # 4. Forward Pass
  # ------------------------------------------------------------------
  # Perform a single forward pass if any features are needed
  if (needs_final_embedding || length(hook_handles) > 0) {
    with(torch$no_grad(), {
      # This single pass calculates final embedding AND triggers hooks
      final_embedding_tensor <- model$encode_image(img_t)
      if (needs_final_embedding) {
        out[["final"]] <- as.numeric(final_embedding_tensor$cpu()$numpy())
      }
    })
  }

  # ------------------------------------------------------------------
  # 5. Collect Intermediate Features & Clean Up Hooks
  # ------------------------------------------------------------------
  if (length(hook_handles) > 0) {

    # Create a map for original integer requests to their resolved block names for output naming
    for (orig_lyr_req_idx in seq_along(intermediate_layers_requested)) {
        orig_lyr <- intermediate_layers_requested[[orig_lyr_req_idx]]
        if (is.numeric(orig_lyr)) {
            resolved_path <- sprintf("%s.%d", vis_transformer_resblocks_base, as.integer(orig_lyr))
            # Use the original numeric request as the key for the output list element,
            # but retrieve from captured_activations_env using the resolved_path
            # For the output list, we want the "user-friendly" name if they gave an int.
            user_friendly_key <- sprintf("%s.%d", vis_transformer_resblocks_base, as.integer(orig_lyr))
            if (resolved_path %in% names(captured_activations_env)) {
                 out[[user_friendly_key]] <- as.array(captured_activations_env[[resolved_path]])
            }
        } else if (is.character(orig_lyr) && orig_lyr %in% names(captured_activations_env)) {
            # If user gave a string, and it was successfully hooked and captured
            out[[orig_lyr]] <- as.array(captured_activations_env[[orig_lyr]])
        }
    }

    # Clean up hooks
    lapply(hook_handles, function(h) {
      if (!is.null(h) && reticulate::py_has_attr(h, "remove")) {
        h$remove()
      }
    })
  }

  # Ensure layers in `out` match the order of unique valid `layers` requested, if possible.
  # This is a bit complex if mapping from original request to potentially modified keys.
  # For now, the names will be "final" and the resolved module paths or original strings.

  return(out)
}