#'
#' @param texts Character vector of text strings to encode.
#' @param layers Character or integer vector. "final" (default) returns the
#'        final CLIP text embeddings. For intermediate transformer layers,
#'        provide full layer names (e.g., "transformer.resblocks.6")
#'        or integer indices (0-based) representing transformer blocks.
#'        Other layer names (e.g., "token_embedding", "ln_final")
#'        can also be specified if known.
#' @param model_name CLIP model string (e.g. "ViT-B-32", "RN50", "ViT-L-14").
#' @param num_text_transformer_blocks Integer, number of transformer blocks in the
#'        text model. Defaults to 12 (e.g., for ViT-B/32 text transformer).
#'        Only relevant if integer indices are used for `layers`.
#' @param text_module_prefix Character string, the base path to the text model's
#'        transformer blocks. Defaults to "transformer".
#' @param device "cpu" or "cuda".
#' @return A named list of numeric arrays (one per requested layer).
#'         `out$final` will be a matrix (N_texts x EmbeddingDim).
#'         Intermediate layers will be 3D arrays (N_texts x SeqLen x HiddenDim).
#' @export
clip_text_features <- function(texts,
                               layers = "final",
                               model_name = "ViT-B-32",
                               num_text_transformer_blocks = 12, # Default for ViT-B/32
                               text_module_prefix = "transformer", # e.g. model.transformer.resblocks
                               device = c("cpu", "cuda")) {

  device <- match.arg(device)

  # Python helper function string (can be shared if in the same package)
  py_helpers <- "
import torch

def get_submodule_by_path(model, module_path_str):
    module = model
    try:
        for part in module_path_str.split('.'):
            if part.isdigit() and hasattr(module, '__getitem__') and not isinstance(module, torch.Tensor):
                # Attempt to access list-like elements (e.g., resblocks in a ModuleList)
                module = module[int(part)]
            else:
                module = getattr(module, part)
    except (AttributeError, IndexError) as e:
        error_message = \"Failed to access submodule: '{}'. Part '{}' not found or index out of bounds. Error: {}\".format(module_path_str, part, str(e))
        raise type(e)(error_message)
    return module

# r_hooks_env is an R environment (passed as a Python dict by reticulate)
# key_for_r_env is the string key for storing
def register_hook_on_submodule_helper(model, module_path_str, r_hooks_env, key_for_r_env):
    module_to_hook = get_submodule_by_path(model, module_path_str)

    def hook_fn(module, input, output):
        activation_tensor = output
        # Transformer blocks often output a tuple (hidden_states, attention_weights)
        # We usually want the hidden_states (first element)
        if isinstance(output, tuple):
            activation_tensor = output[0]
        r_hooks_env[key_for_r_env] = activation_tensor.detach().cpu().numpy()

    return module_to_hook.register_forward_hook(hook_fn)
"
  # Make helper available in Python main module
  reticulate::py_run_string(py_helpers)
  py_register_hook <- reticulate::py$register_hook_on_submodule_helper
  # py_get_submodule <- reticulate::py$get_submodule_by_path # Not strictly needed by this R function directly

  # ------------------------------------------------------------------
  # 1.  Bridge to Python open_clip
  # ------------------------------------------------------------------
  oc    <- reticulate::import("open_clip", delay_load = TRUE)
  torch <- reticulate::import("torch", delay_load = TRUE)

  # Create model and tokenizer (preprocess is for images)
  model_and_transforms <- oc$create_model_and_transforms(
    model_name,
    pretrained = "openai",
    device = device
  )

  model     <- model_and_transforms[[1]]
  tokenizer <- oc$get_tokenizer(model_name) # Get tokenizer separately
  model$eval()

  # ------------------------------------------------------------------
  # 2.  Tokenize texts
  # ------------------------------------------------------------------
  # open_clip.tokenize takes a list/vector of strings
  text_tokens <- tokenizer(texts)$to(device = device)

  # Results list
  out <- list()
  # R environment to store activations from hooks
  captured_activations_env <- new.env(parent = emptyenv())

  # ------------------------------------------------------------------
  # 3. Layer Processing & Hook Registration
  # ------------------------------------------------------------------
  needs_final_embedding <- "final" %in% layers
  intermediate_layers_requested <- setdiff(layers, "final")

  # Base path for text transformer residual blocks
  text_transformer_resblocks_base <- sprintf("%s.resblocks", text_module_prefix)

  # Resolve all requested intermediate layers to their full module path strings
  resolved_intermediate_module_paths <- character(0)
  if (length(intermediate_layers_requested) > 0) {
    resolved_intermediate_module_paths <- sapply(intermediate_layers_requested, function(lyr) {
      if (is.numeric(lyr)) {
        if (lyr >= 0 && lyr < num_text_transformer_blocks) {
          return(sprintf("%s.%d", text_transformer_resblocks_base, as.integer(lyr)))
        } else {
          warning(sprintf("Integer layer index %d is out of bounds for text transformer (0-%d). Skipping.",
                          as.integer(lyr), num_text_transformer_blocks - 1))
          return(NA_character_)
        }
      } else if (is.character(lyr)) {
        return(lyr) # Assume full module path string
      } else {
        warning(sprintf("Invalid layer specification: %s. Skipping.", lyr))
        return(NA_character_)
      }
    }, USE.NAMES = FALSE)
    resolved_intermediate_module_paths <- resolved_intermediate_module_paths[!is.na(resolved_intermediate_module_paths)]
    resolved_intermediate_module_paths <- unique(resolved_intermediate_module_paths)
  }

  hook_handles <- list()
  if (length(resolved_intermediate_module_paths) > 0) {
    message("Registering text model hooks for: ", paste(resolved_intermediate_module_paths, collapse=", "))
    for (module_path_str in resolved_intermediate_module_paths) {
      tryCatch({
        handle <- py_register_hook(model, module_path_str, captured_activations_env, module_path_str)
        hook_handles[[module_path_str]] <- handle
      }, error = function(e) {
        warning(sprintf("Failed to register hook for text layer '%s'. Error: %s", module_path_str, e$message))
      })
    }
  }

  # ------------------------------------------------------------------
  # 4. Forward Pass (encode_text)
  # ------------------------------------------------------------------
  if (needs_final_embedding || length(hook_handles) > 0) {
    with(torch$no_grad(), {
      final_text_embeddings_tensor <- model$encode_text(text_tokens)
      if (needs_final_embedding) {
        # Result is N_texts x EmbeddingDim
        out[["final"]] <- final_text_embeddings_tensor$cpu()$numpy()
      }
    })
  }

  # ------------------------------------------------------------------
  # 5. Collect Intermediate Features & Clean Up Hooks
  # ------------------------------------------------------------------
  if (length(hook_handles) > 0) {
    for (module_path_key in resolved_intermediate_module_paths) {
      # Determine the key for the output list based on original user request
      output_key <- module_path_key # Default to the full path
      # Check if this module_path_key was derived from an integer request
      is_numeric_derived <- FALSE
      for(orig_lyr in intermediate_layers_requested) {
          if(is.numeric(orig_lyr)){
              if (sprintf("%s.%d", text_transformer_resblocks_base, as.integer(orig_lyr)) == module_path_key) {
                  # For output, use the user-friendly name if derived from an integer
                  output_key <- sprintf("%s.%d", text_transformer_resblocks_base, as.integer(orig_lyr))
                  is_numeric_derived <- TRUE
                  break
              }
          }
      }
      # If not from numeric, and user provided a string that matches the resolved path, use that original string.
      # This ensures if a user provides "transformer.ln_final" it stays that way, not potentially
      # a transformed version if text_module_prefix was involved differently.
      if (!is_numeric_derived && module_path_key %in% intermediate_layers_requested) {
          output_key <- module_path_key
      }

      if (module_path_key %in% names(captured_activations_env)) {
        # Activations are typically Batch x SeqLen x HiddenDim for text transformers
        out[[output_key]] <- as.array(captured_activations_env[[module_path_key]])
      }
    }

    # Clean up hooks
    lapply(hook_handles, function(h) {
      if (!is.null(h) && reticulate::py_has_attr(h, "remove")) {
        h$remove()
      }
    })
  }

  return(out)
}
