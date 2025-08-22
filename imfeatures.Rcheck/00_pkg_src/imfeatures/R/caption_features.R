#' Extract Caption Features from Images
#'
#' @description
#' Generate natural language captions for images and optionally compute
#' text embeddings for the captions. Uses state-of-the-art vision-language
#' models via the ellmer package.
#'
#' @param impath Path to image file(s).
#' @param caption_provider LLM provider for captioning. Options: "openai",
#'   "anthropic", "gemini", "azure_openai", "ollama", "huggingface",
#'   "openrouter", "vllm".
#' @param caption_model Model name for captioning. If NULL, uses provider default.
#' @param template Caption template. Options: "factual", "dense", "alt_text",
#'   "product", "art".
#' @param focus Character vector of focus areas. Options: "objects", "text",
#'   "layout", "colors", "materials", "actions", "people".
#' @param min_words Minimum words in caption.
#' @param max_words Maximum words in caption.
#' @param temperature Generation temperature (0-1).
#' @param max_tokens Maximum tokens to generate.
#' @param seed Random seed for reproducibility.
#' @param extra_instructions Additional prompt instructions.
#' @param compute_embedding Whether to compute text embeddings for captions.
#' @param embedding_backend Backend for embeddings: "openai", "gemini", "hf".
#' @param embedding_model Model for embeddings. If NULL, uses backend default.
#' @param embedding_dim Optional embedding dimensions.
#' @param gemini_task Task type for Gemini embeddings.
#' @param echo Ellmer echo mode.
#'
#' @return A tibble with class `imfeatures_feature_tbl` containing:
#'   - image: Image file path
#'   - caption: Generated caption text
#'   - embedding: Text embedding vector (if compute_embedding = TRUE)
#'   - embedding_dim: Embedding dimensions
#'   - metadata: List column with generation parameters
#'
#' @examples
#' \dontrun{
#' # Basic caption generation
#' captions <- caption_features(
#'   "image.jpg",
#'   caption_provider = "openai",
#'   template = "dense"
#' )
#'
#' # Caption with embeddings
#' features <- caption_features(
#'   c("img1.jpg", "img2.jpg"),
#'   caption_provider = "gemini",
#'   compute_embedding = TRUE,
#'   embedding_backend = "gemini"
#' )
#' }
#'
#' @export
caption_features <- function(
  impath,
  caption_provider = "openai",
  caption_model = NULL,
  template = "dense",
  focus = c("objects", "text", "layout", "colors"),
  min_words = 60,
  max_words = 120,
  temperature = 0.2,
  max_tokens = 512,
  seed = NULL,
  extra_instructions = NULL,
  compute_embedding = TRUE,
  embedding_backend = "openai",
  embedding_model = NULL,
  embedding_dim = NULL,
  gemini_task = NULL,
  echo = "none"
) {
  # Validate image paths
  assert_image(impath)
  
  # Check ellmer availability
  if (!requireNamespace("ellmer", quietly = TRUE)) {
    stop("Package 'ellmer' is required for caption generation. ",
         "Please install it with: install.packages('ellmer')")
  }
  
  # Use default model if not specified
  if (is.null(caption_model)) {
    caption_model <- get_default_caption_model(caption_provider)
  }
  
  # Check authentication
  if (!check_caption_provider_auth(caption_provider)) {
    stop("API key not configured for provider '", caption_provider, "'. ",
         "Please set the appropriate environment variable.")
  }
  
  # System prompt for caption generation
  system_prompt <- "You are a careful vision assistant. Be precise, concrete, and avoid speculation."
  
  # Build caption prompt
  prompt <- build_caption_prompt(
    template = template,
    focus = focus,
    min_words = min_words,
    max_words = max_words,
    extra_instructions = extra_instructions
  )
  
  # Configure chat provider
  chat <- .select_chat(
    provider = caption_provider,
    model = caption_model,
    system_prompt = system_prompt,
    temperature = temperature,
    max_tokens = max_tokens,
    seed = seed,
    echo = echo
  )
  
  # Process each image
  results <- lapply(impath, function(img) {
    tryCatch({
      # Generate caption
      caption <- chat$chat(
        ellmer::content_image_file(img),
        prompt
      )
      
      # Compute embedding if requested
      if (compute_embedding) {
        embedding <- embed_text(
          caption,
          backend = embedding_backend,
          model = embedding_model,
          dimensions = embedding_dim,
          task_type = gemini_task
        )
        embedding_dim_actual <- length(embedding)
      } else {
        embedding <- numeric(0)
        embedding_dim_actual <- NA_integer_
      }
      
      # Create metadata
      metadata <- list(
        caption_provider = caption_provider,
        caption_model = caption_model,
        template = template,
        focus = focus,
        min_words = min_words,
        max_words = max_words,
        temperature = temperature,
        seed = seed
      )
      
      if (compute_embedding) {
        metadata$embedding_backend <- embedding_backend
        metadata$embedding_model <- embedding_model
        metadata$embedding_dim <- embedding_dim_actual
      }
      
      list(
        image = img,
        caption = caption,
        embedding = embedding,  # Don't wrap in another list
        embedding_dim = embedding_dim_actual,
        metadata = metadata  # Don't wrap in another list
      )
    }, error = function(e) {
      list(
        image = img,
        caption = paste0("ERROR: ", conditionMessage(e)),
        embedding = numeric(0),
        embedding_dim = NA_integer_,
        metadata = list(error = conditionMessage(e))
      )
    })
  })
  
  # Convert to tibble
  tbl <- tibble::tibble(
    image = sapply(results, `[[`, "image"),
    caption = sapply(results, `[[`, "caption"),
    embedding = lapply(results, `[[`, "embedding"),  # Each embedding is already a vector
    embedding_dim = sapply(results, `[[`, "embedding_dim"),
    metadata = lapply(results, `[[`, "metadata")  # Each metadata is already a list
  )
  
  # Add class for consistency with package
  new_feature_tbl(tbl)
}

#' Batch caption feature extraction with progress
#'
#' @param impaths Vector of image file paths.
#' @param ... Arguments passed to caption_features.
#' @param .progress Show progress bar.
#'
#' @return A tibble with class `imfeatures_feature_tbl`.
#' @export
caption_features_many <- function(impaths, ..., .progress = interactive()) {
  impaths <- as.character(impaths)
  
  if (.progress) {
    pb <- progress::progress_bar$new(
      total = length(impaths),
      format = "Captioning [:bar] :current/:total (:percent) ETA: :eta"
    )
  }
  
  results <- lapply(seq_along(impaths), function(i) {
    if (.progress) pb$tick()
    caption_features(impaths[i], ...)
  })
  
  dplyr::bind_rows(results)
}

#' Compute similarity between caption embeddings
#'
#' @param caption_features A tibble from caption_features with embeddings.
#' @param metric Similarity metric (default: "cosine").
#'
#' @return Similarity matrix.
#' @export
compute_caption_similarity <- function(caption_features, metric = "cosine") {
  if (!"embedding" %in% names(caption_features)) {
    stop("No embeddings found. Run caption_features with compute_embedding = TRUE")
  }
  
  # Extract embeddings
  embeddings <- do.call(rbind, caption_features$embedding)
  
  # Remove any rows with NA embeddings
  valid_rows <- !is.na(caption_features$embedding_dim)
  if (!all(valid_rows)) {
    warning("Removing ", sum(!valid_rows), " images with failed embeddings")
    embeddings <- embeddings[valid_rows, , drop = FALSE]
  }
  
  # Compute similarity
  if (metric == "cosine") {
    sim_matrix <- coop::tcosine(embeddings)
  } else {
    sim_matrix <- as.matrix(proxy::simil(embeddings, method = metric))
  }
  
  # Set row/column names
  rownames(sim_matrix) <- basename(caption_features$image[valid_rows])
  colnames(sim_matrix) <- basename(caption_features$image[valid_rows])
  
  sim_matrix
}

#' Extract multimodal features (visual + caption)
#'
#' @param impath Image file path(s).
#' @param visual_layers Layers for visual feature extraction.
#' @param visual_model Keras model for visual features.
#' @param caption_provider Provider for caption generation.
#' @param caption_template Template for captions.
#' @param ... Additional arguments for caption_features.
#'
#' @return A tibble with both visual and caption features.
#' @export
extract_multimodal_features <- function(
  impath,
  visual_layers = c(15, 17, 19),
  visual_model = NULL,
  caption_provider = "openai",
  caption_template = "dense",
  ...
) {
  # Extract visual features
  visual_features <- extract_features(
    impath,
    layers = visual_layers,
    model = visual_model
  )
  
  # Extract caption features
  caption_features <- caption_features(
    impath,
    caption_provider = caption_provider,
    template = caption_template,
    ...
  )
  
  # Combine results
  combined <- dplyr::left_join(
    visual_features,
    caption_features,
    by = "image",
    suffix = c("_visual", "_caption")
  )
  
  new_feature_tbl(combined)
}