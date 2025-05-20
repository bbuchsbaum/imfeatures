#' Extract Deep Learning Features using thingsvision
#'
#' This is the primary user-facing function to extract features from images
#' using a wide variety of models available through the Python `thingsvision` library.
#' It handles model loading, data preparation, feature extraction, and returns
#' features in a standard R format.
#'
#' @param impaths Character vector. A vector of full file paths to the images
#'        for which features should be extracted. The order of features in the
#'        output will correspond to the order of paths in this vector.
#'        The images can be located in different directories. A common root
#'        directory is automatically derived and the paths are passed to the
#'        Python dataset relative to that root.
#' @param model_name Character string. The name of the model architecture (e.g.,
#'        `"resnet50"`, `"clip"`, `"dino-vit-base-p16"`). See the Details section
#'        of \code{\link{tv_get_extractor}} for available models.
#' @param source Character string. The source library of the model (e.g.,
#'        `"torchvision"`, `"timm"`, `"ssl"`, `"custom"`). See the Details section
#'        of \code{\link{tv_get_extractor}} for sources.
#' @param module_name Character string. The specific layer or module within the
#'        model from which to extract activations. Use
#'        `tv_show_model(tv_get_extractor(model_name, source))` to list available
#'        module names for a given model. Common examples include final layers
#'        like `"avgpool"` or `"fc"` (in ResNets), `"classifier.6"` (in VGG/AlexNet),
#'        `"visual"` (in CLIP), or intermediate layers like `"features.10"` (in VGG).
#' @param device Character string. The compute device ("cpu", "cuda", "cuda:0").
#'        Defaults to "cuda".
#' @param pretrained Logical. Use pretrained model weights? Defaults to TRUE.
#' @param model_parameters Named list (optional). Additional parameters needed for
#'        specific models (e.g., `list(variant = "ViT-B/32")` for CLIP). See
#'        \code{\link{tv_get_extractor}} documentation for details. Defaults to NULL.
#' @param flatten_acts Logical. If TRUE, flattens activations from intermediate
#'        layers (e.g., convolutional or transformer layers with spatial dimensions)
#'        into a 2D matrix (n_images x n_features). If FALSE, retains the original
#'        dimensions (e.g., n_images x channels x height x width for CNNs, or
#'        n_images x tokens x dimensions for ViTs, after token extraction).
#'        Defaults to FALSE. Flattening is often necessary for standard machine
#'        learning or statistical analysis comparing images.
#' @param batch_size Integer. Number of images to process in each batch. Adjust based
#'        on available GPU memory and model size. Defaults to 32.
#' @param temp_out_dir Character string. Path to a directory where a temporary file
#'        listing the order of processed images will be written. This is required
#'        internally by the underlying Python `ImageDataset` to ensure correct
#'        feature ordering. The contents are usually not needed by the end-user.
#'        Defaults to `tempdir()`.
#' @param output_dir Character string (optional). If provided, features will be saved
#'        iteratively to this directory in batches (as `.npy` files, since
#'        `output_type="ndarray"` is used internally) instead of being returned directly.
#'        This is useful for very large datasets or models that produce large feature
#'        maps, preventing potential out-of-memory errors in R. If used, the function
#'        returns `NULL` invisibly. Defaults to NULL (features returned in memory).
#'
#' @details
#' \strong{Workflow:}
#' This function performs the following steps internally:
#' \enumerate{
#'  \item Calls \code{\link{tv_get_extractor}} to load the specified model via `reticulate`.
#'  \item Creates a `thingsvision` `ImageDataset` and `DataLoader` to handle image loading,
#'        preprocessing (using transforms from the extractor), and batching. It writes a
#'        `file_names.txt` in `temp_out_dir` to preserve order.
#'  \item Calls \code{\link{tv_extract_features}} to perform the batched feature extraction
#'        from the specified `module_name`.
#'  \item Converts the extracted features (typically NumPy arrays from Python) into
#'        an R matrix or array.
#'  \item Optionally attempts to add image basenames (without extension) as rownames.
#' }
#'
#' \strong{Finding Module Names:}
#' The `module_name` is critical. To find valid names for your chosen `model_name` and `source`:
#' \preformatted{
#'   extractor <- tv_get_extractor("resnet50", "torchvision")
#'   tv_show_model(extractor) # Prints the model structure
#' }
#' Look for meaningful layer names in the output (e.g., `avgpool`, `layer4`, `fc` for ResNet).
#'
#' \strong{Output Dimensions:}
#' The dimensions of the returned object depend on the layer (`module_name`) and the
#' `flatten_acts` parameter:
#' \itemize{
#'  \item If `flatten_acts = TRUE`: Returns a 2D matrix (n_images x n_features).
#'  \item If `flatten_acts = FALSE`:
#'    \itemize{
#'      \item For typical final layers (like avgpool, fc, classifier): Often already 2D (n_images x n_features).
#'      \item For convolutional layers: Returns a 4D array (n_images x channels x height x width).
#'      \item For transformer layers (after token handling): May return a 3D array (n_images x n_tokens x embedding_dim) or 2D if only CLS token is kept. Check output carefully.
#'    }
#' }
#'
#' \strong{Low-Memory Extraction (`output_dir`):}
#' When `output_dir` is specified, features for each batch (or group of batches,
#' controlled by `step_size` in `tv_extract_features`, which this function doesn't expose
#' directly but uses a default) are saved as separate `.npy` files (e.g., `features_0-32.npy`,
#' `features_32-64.npy`, ...) in the specified directory. The main R function then returns `NULL`.
#' You would need to load and combine these files manually after the function completes, e.g., using:
#' \preformatted{
#'   feature_files <- list.files(output_dir, pattern = "^features_.*\\.npy$", full.names = TRUE)
#'   # Ensure correct order if needed, potentially by parsing filenames
#'   all_features_list <- lapply(sort(feature_files), RcppCNPy::npyLoad)
#'   all_features <- do.call(rbind, all_features_list) # If they are 2D matrices
#' }
#'
#' \strong{Prerequisites:}
#' Requires a correctly configured Python environment with `thingsvision` and its
#' dependencies installed. Use \code{\link{install_thingsvision}} to set this up
#' and configure `reticulate` (e.g., `reticulate::use_condaenv("r-thingsvision")`)
#' before calling this function.
#'
#' @return An R matrix (if `flatten_acts=TRUE` or the layer is naturally 2D) or
#'         an R array (if `flatten_acts=FALSE` and the layer has >2 dimensions).
#'         Rownames corresponding to the base image names (without extension) are
#'         attempted to be set. Returns `NULL` invisibly if `output_dir` is specified.
#'
#' @importFrom tools file_path_sans_ext
#' @export
#' @seealso \code{\link{install_thingsvision}}, \code{\link{tv_get_extractor}}, \code{\link{tv_show_model}}, \code{\link{tv_extract_features}}, \code{\link{tv_create_dataset}}, \code{\link{tv_create_dataloader}}
#' @examples
#' \dontrun{
#' # --- Prerequisites ---
#' # 1. Install thingsvision Python environment (only needs to be done once)
#' # install_thingsvision()
#'
#' # 2. Load library and configure reticulate for the current session
#' library(imfeatures)
#' library(reticulate)
#' tryCatch({
#'   use_condaenv("r-thingsvision", required = TRUE)
#'   tv <- import("thingsvision") # Ensure it's loaded
#' }, error = function(e) {
#'   message("Python environment 'r-thingsvision' not found or reticulate setup failed.")
#'   message("Make sure you ran install_thingsvision() and reticulate is configured.")
#' })
#'
#' # --- Example Usage ---
#' # Create some dummy image files for demonstration
#' image_dir <- file.path(tempdir(), "test_images")
#' dir.create(image_dir, showWarnings = FALSE)
#' png(file.path(image_dir, "img1.png")); plot(1:10); dev.off()
#' png(file.path(image_dir, "img2.png")); plot(rnorm(100)); dev.off()
#' image_paths <- list.files(image_dir, full.names = TRUE, pattern = "\\.png$")
#'
#' # Example 1: Extract ResNet-18 'avgpool' features (flattened by default layer shape)
#' features_rn18 <- im_features_tv(
#'   impaths = image_paths,
#'   model_name = "resnet18",
#'   source = "torchvision",
#'   module_name = "avgpool",
#'   flatten_acts = TRUE, # Explicitly flatten (though avgpool often is already flat)
#'   device = "cpu" # Use CPU for this example if no GPU
#' )
#' print(dim(features_rn18)) # Should be n_images x 512
#' print(rownames(features_rn18))
#'
#' # Example 2: Extract CLIP 'visual' features (ViT-B/32 variant)
#' # Note: Requires pip install git+https://github.com/openai/CLIP.git in the env
#' features_clip <- im_features_tv(
#'    impaths = image_paths,
#'    model_name = "clip",
#'    source = "custom",
#'    module_name = "visual", # The image encoder output
#'    model_parameters = list(variant = "ViT-B/32"),
#'    flatten_acts = TRUE, # Usually needed for downstream tasks
#'    device = "cpu"
#' )
#' print(dim(features_clip)) # Should be n_images x 512 for ViT-B/32
#'
#' # Example 3: Extract intermediate VGG layer (without flattening)
#' features_vgg_conv <- im_features_tv(
#'   impaths = image_paths,
#'   model_name = "vgg16",
#'   source = "torchvision",
#'   module_name = "features.10", # An intermediate conv layer
#'   flatten_acts = FALSE,
#'   device = "cpu"
#' )
#' print(dim(features_vgg_conv)) # Should be 4D: n_images x channels x H x W
#'
#' # Example 4: Low-memory extraction
#' low_mem_dir <- file.path(tempdir(), "low_mem_features")
#' dir.create(low_mem_dir)
#' result <- im_features_tv(
#'   impaths = image_paths,
#'   model_name = "alexnet",
#'   source = "torchvision",
#'   module_name = "features.8",
#'   output_dir = low_mem_dir, # Specify output directory
#'   flatten_acts = TRUE,
#'   device = "cpu"
#' )
#' print(result) # Should print NULL
#' print(list.files(low_mem_dir)) # Shows the saved .npy files
#'
#' # Clean up dummy files
#' unlink(image_dir, recursive = TRUE)
#' unlink(low_mem_dir, recursive = TRUE)
#' }
im_features_tv <- function(impaths, model_name, source, module_name,
                          device = "cuda", pretrained = TRUE, model_parameters = NULL,
                          flatten_acts = FALSE, batch_size = 32, temp_out_dir = tempdir(),
                          output_dir = NULL) {

  # --- Input Validation and Setup ---
  if (!is.character(impaths) || length(impaths) == 0) {
    stop("'impaths' must be a character vector of image file paths.")
  }
  if (!all(file.exists(impaths))) {
     missing_files <- impaths[!file.exists(impaths)]
     stop("Some image paths do not exist: ", paste(missing_files, collapse=", "))
  }
  # Determine a common root directory and relative paths
  image_root <- .common_root(impaths)
  image_fnames <- .relative_to_root(impaths, image_root)
  if (!dir.exists(image_root))
    stop("Computed common root directory not found: ", image_root)

  # Ensure temp_out_dir exists for the file list
  if (!dir.exists(temp_out_dir)) {
     message("Creating temporary directory for file list: ", temp_out_dir)
     dir.create(temp_out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  # Ensure output_dir exists if specified
  if (!is.null(output_dir) && !dir.exists(output_dir)) {
     message("Creating output directory for low-memory features: ", output_dir)
     dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  # --- Core Logic ---
  tryCatch({
    # 1. Get Extractor (R object wrapper)
    message("Loading extractor: ", model_name, " from ", source, "...")
    # This now returns the R object of class 'thingsvision_extractor'
    extractor <- tv_get_extractor(model_name, source, device, pretrained, model_parameters)
    message("Extractor loaded (", class(extractor)[1] ," R object).")

    # 2. Create Dataset and Dataloader
    # Assuming tv_create_dataset and tv_create_dataloader now accept the R extractor object
    message("Creating dataset and dataloader...")

    dataset <- tv_create_dataset(
       root = image_root,
       out_path = temp_out_dir, 
       extractor = extractor, # Pass the R object
       file_names = image_fnames 
    )
    # Assuming tv_create_dataloader also takes the R extractor object
    dataloader <- tv_create_dataloader(dataset, batch_size, extractor)
    message("Dataloader ready.")

    # 3. Extract Features using the S3 generic
    message("Starting feature extraction from module: ", module_name, "...")
    # Use the tv_extract generic which dispatches to tv_extract.thingsvision_extractor
    features <- tv_extract(
      object = extractor, # Pass the R object
      dataloader = dataloader,
      module_name = module_name,
      flatten_acts = flatten_acts,
      output_type = "ndarray",
      output_dir = output_dir 
    )
    message("Feature extraction finished.")

    # 4. Post-processing and Return
    if (!is.null(output_dir)) {
       message("Features saved iteratively to: ", output_dir)
       return(invisible(NULL)) # Return NULL as instructed when output_dir is used
    } else if (!is.null(features)) {
       # Attempt to add rownames
       img_basenames <- try(tools::file_path_sans_ext(basename(impaths)), silent = TRUE)
       if (!inherits(img_basenames, "try-error") && length(img_basenames) == NROW(features)) {
          try(rownames(features) <- img_basenames, silent = TRUE)
       } else {
          warning("Could not set rownames for features. Number of images might not match feature rows, or basename extraction failed.")
       }
       return(features)
    } else {
       # This case might happen if tv_extract returns NULL unexpectedly
       warning("Feature extraction did not return features and output_dir was not set.")
       return(NULL)
    }

  }, error = function(e) {
     # Catch errors from any step (extractor, dataset, extraction)
     stop("Error during thingsvision feature extraction: \n", e$message)
  })

}

# Add necessary imports if not already present in DESCRIPTION or NAMESPACE
# Imports: proxy, tools

#' Compute Similarity Matrix using thingsvision Features
#'
#' Calculates the pairwise similarity between a set of images based on features
#' extracted from specified model layers using the `thingsvision` backend.
#'
#' @param impaths Character vector. A vector of full file paths to the images.
#'        The order determines the rows/columns of the output similarity matrices.
#'        Images can reside in different directories and will be processed
#'        relative to their computed common root.
#' @param model_name Character string. The name of the `thingsvision` model architecture
#'        (e.g., `"resnet50"`, `"clip"`).
#' @param source Character string. The source library of the model
#'        (e.g., `"torchvision"`, `"custom"`).
#' @param module_names Character vector. The specific layer/module names within the
#'        model from which to extract features for similarity calculation. Use
#'        `tv_show_model(tv_get_extractor(model_name, source))` to find valid names.
#' @param metric Character string. The similarity metric to use. Defaults to "cosine".
#'        Common options include "cosine", "correlation". See `proxy::pr_simil_funs`
#'        for available metrics supported by the `proxy` package.
#' @param flatten_acts Logical. Should activations from the specified `module_names`
#'        be flattened into vectors before calculating similarity? This is almost
#'        always required for standard similarity metrics like cosine or correlation.
#'        Defaults to TRUE. Setting to FALSE will likely cause errors unless the
#'        metric can handle multi-dimensional arrays and the chosen layer output
#'        is suitable.
#' @param device Character string. The compute device ("cpu", "cuda", "cuda:0").
#'        Defaults to "cuda".
#' @param pretrained Logical. Use pretrained model weights? Defaults to TRUE.
#' @param model_parameters Named list (optional). Additional parameters for specific
#'        models (e.g., `list(variant = "ViT-B/32")` for CLIP). Defaults to NULL.
#' @param batch_size Integer. Batch size for feature extraction. Defaults to 32.
#' @param temp_out_dir Character string. Temporary directory for internal file list
#'        used during feature extraction. Defaults to `tempdir()`.
#'
#' @details
#' This function streamlines the process of calculating representational similarity
#' matrices (RSMs) using features from the `thingsvision` ecosystem.
#'
#' \strong{Workflow:}
#' \enumerate{
#'  \item It iterates through each `module_name` provided.
#'  \item For each module, it calls \code{\link{im_features_tv}} to extract features
#'        for all images specified in `impaths`. The `flatten_acts` parameter is
#'        crucial here to ensure features are in a suitable format (usually 2D matrix)
#'        for standard similarity calculation.
#'  \item It then calculates the full pairwise similarity matrix for the extracted
#'        features using the specified `metric` via the `proxy` package (or `coop`
#'        for optimized cosine).
#'  \item Rownames and colnames of the similarity matrices are set based on the
#'        image basenames.
#' }
#'
#' \strong{Memory Considerations:}
#' This function extracts features for *all* images for a given module *before*
#' calculating the similarity matrix for that module. This is generally efficient
#' if the features for all images fit into memory. It does *not* currently implement
#' the pair-by-pair extraction (`lowmem=TRUE`) strategy found in the original
#' `im_feature_sim` function, as the primary bottleneck is often feature extraction
#' itself when using large models. If memory issues arise during the similarity
#' calculation step (after feature extraction), consider using metrics optimized
#' for memory or processing subsets of images. The `output_dir` option in
#' `im_features_tv` can handle cases where the features *themselves* don't fit
#' in memory during extraction.
#'
#' \strong{Prerequisites:}
#' Requires a correctly configured Python environment with `thingsvision` installed.
#' Use \code{\link{install_thingsvision}} and configure `reticulate` before use.
#'
#' @return A named list where each element corresponds to a `module_name` provided
#'         in the input. Each element contains a square similarity matrix
#'         (n_images x n_images).
#'
#' @importFrom proxy simil
#' @importFrom tools file_path_sans_ext
# @importFrom coop tcosine # Optional: uncomment if using coop specifically
#' @export
#' @seealso \code{\link{im_features_tv}}, \code{\link{install_thingsvision}}, \code{\link{tv_get_extractor}}
#' @examples
#' \dontrun{
#' # --- Prerequisites ---
#' # install_thingsvision()
#' library(imfeatures)
#' library(reticulate)
#' tryCatch({
#'   use_condaenv("r-thingsvision", required = TRUE)
#'   tv <- import("thingsvision")
#' }, error = function(e) message("Python env 'r-thingsvision' not found."))
#'
#' # --- Example Usage ---
#' # Create dummy image files
#' image_dir <- file.path(tempdir(), "sim_test_images")
#' dir.create(image_dir, showWarnings = FALSE)
#' png(file.path(image_dir, "cat.png")); plot(1:5); dev.off()
#' png(file.path(image_dir, "dog.png")); plot(rnorm(50)); dev.off()
#' png(file.path(image_dir, "car.png")); plot(1:20); dev.off()
#' image_paths <- list.files(image_dir, full.names = TRUE, pattern = "\\.png$")
#'
#' # Calculate similarity based on ResNet-18 avgpool and layer4 features
#' sim_results <- im_feature_sim_tv(
#'   impaths = image_paths,
#'   model_name = "resnet18",
#'   source = "torchvision",
#'   module_names = c("avgpool", "layer4"), # Request features from two layers
#'   metric = "cosine",
#'   flatten_acts = TRUE, # Flatten layer4 activations
#'   device = "cpu"
#' )
#'
#' # Explore results
#' print(names(sim_results))
#' print(dim(sim_results$avgpool))
#' print(sim_results$avgpool)
#'
#' # Clean up
#' unlink(image_dir, recursive = TRUE)
#' }
im_feature_sim_tv <- function(impaths, model_name, source, module_names,
                              metric = "cosine",
                              flatten_acts = TRUE,
                              device = "cuda", pretrained = TRUE, model_parameters = NULL,
                              batch_size = 32, temp_out_dir = tempdir()) {

  # --- Input Checks ---
  if (!is.character(impaths) || length(impaths) < 2) {
    stop("'impaths' must be a character vector with at least two image paths.")
  }
  if (!is.character(module_names) || length(module_names) == 0) {
    stop("'module_names' must be a character vector with at least one module name.")
  }
  # Add check for metric validity using proxy?
  if (!metric %in% proxy::pr_simil_funs()) {
     warning("Metric '", metric, "' not found in proxy::pr_simil_funs. Calculation might fail.")
  }

  # --- Feature Extraction and Similarity Calculation ---
  sim_matrices <- list()
  image_basenames <- try(tools::file_path_sans_ext(basename(impaths)), silent = TRUE)

  for (mod_name in module_names) {
    message("Processing module: ", mod_name, "...")

    # Extract features for the current module for ALL images
    # This now calls the updated im_features_tv which handles the R extractor object
    features_r <- tryCatch({
        im_features_tv(
            impaths = impaths,
            model_name = model_name,
            source = source,
            module_name = mod_name,
            device = device,
            pretrained = pretrained,
            model_parameters = model_parameters,
            flatten_acts = flatten_acts, # Use the function argument
            batch_size = batch_size,
            temp_out_dir = temp_out_dir,
            output_dir = NULL # Ensure features are returned in memory
        )
     }, error = function(e) {
        warning("Failed to extract features for module '", mod_name, "': ", e$message)
        return(NULL) # Return NULL to skip similarity calculation for this module
     })

    if (is.null(features_r)) {
      message("Skipping similarity calculation for module '", mod_name, "' due to feature extraction error.")
      next # Skip to the next module
    }

    # Ensure features are a 2D matrix for standard similarity metrics
    if (length(dim(features_r)) != 2) {
       warning("Features for module '", mod_name, "' are not a 2D matrix (Dimensions: ", paste(dim(features_r), collapse="x"), "). ",
               "Similarity calculation with metric '", metric, "' might fail or produce unexpected results. ",
               "Ensure 'flatten_acts=TRUE' is appropriate for this layer or use a suitable metric.")
        # Optionally attempt flattening here if flatten_acts was FALSE but user still proceeded?
        # Or just let proxy::simil handle it / error out.
    }

    message("Calculating similarity matrix (", metric, ") for module: ", mod_name, "...")
    # Calculate similarity matrix
    if (metric == "cosine" && requireNamespace("coop", quietly = TRUE)) {
        # Use optimized cosine if available
        sim_mat <- tryCatch({
             coop::tcosine(t(features_r)) # coop::tcosine expects features as columns
        }, error = function(e){
             warning("coop::tcosine failed: ", e$message, ". Falling back to proxy::simil.")
             proxy::simil(features_r, method = metric)
        })
    } else {
        sim_mat <- proxy::simil(features_r, method = metric)
    }

    # Convert to standard matrix and add names
    sim_mat <- as.matrix(sim_mat)
    if (!inherits(image_basenames, "try-error") && length(image_basenames) == nrow(sim_mat)) {
       rownames(sim_mat) <- image_basenames
       colnames(sim_mat) <- image_basenames
    } else {
       warning("Could not set row/column names for similarity matrix of module '", mod_name, "'.")
    }

    # Store in the list, named by module
    # Use make.names to ensure valid R list names if module names contain weird characters
    list_name <- make.names(mod_name)
    sim_matrices[[list_name]] <- sim_mat
    message("Similarity matrix for ", mod_name, " calculated.")

  } # End loop over module_names

  if (length(sim_matrices) == 0) {
     warning("No similarity matrices were calculated, possibly due to errors in feature extraction for all requested modules.")
     return(list())
  }

  return(sim_matrices)
}
