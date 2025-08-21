# Caption Feature Integration Example
# This demonstrates the new caption features integrated into imfeatures

library(imfeatures)

# Setup (uncomment and set your API keys)
# Sys.setenv(OPENAI_API_KEY = "your-key-here")
# Sys.setenv(GEMINI_API_KEY = "your-key-here")
# Sys.setenv(HUGGINGFACE_API_KEY = "your-key-here")

# Example 1: Generate captions with different templates
# ------------------------------------------------------

# Build custom prompts
factual_prompt <- build_caption_prompt(
  template = "factual",
  focus = c("objects", "layout"),
  min_words = 40,
  max_words = 80
)

dense_prompt <- build_caption_prompt(
  template = "dense",
  focus = c("objects", "text", "colors", "materials"),
  min_words = 80,
  max_words = 150
)

cat("Factual prompt:\n", factual_prompt, "\n\n")
cat("Dense prompt:\n", dense_prompt, "\n\n")

# Example 2: Check available providers
# -------------------------------------

providers <- list_caption_providers()
print(providers)

# Get available templates and focus areas
templates <- get_caption_templates()
focus_areas <- get_caption_focus_areas()

cat("\nAvailable templates:", paste(templates, collapse = ", "), "\n")
cat("Available focus areas:", paste(focus_areas, collapse = ", "), "\n\n")

# Example 3: Caption a single image (requires API key)
# -----------------------------------------------------

if (check_caption_provider_auth("openai")) {
  # Create a test image
  test_img <- tempfile(fileext = ".jpg")
  jpeg::writeJPEG(array(runif(100*100*3), dim = c(100, 100, 3)), test_img)
  
  # Generate caption with embeddings
  result <- caption_features(
    test_img,
    caption_provider = "openai",
    caption_model = "gpt-4o-mini",
    template = "dense",
    focus = c("objects", "colors"),
    compute_embedding = TRUE,
    embedding_backend = "openai"
  )
  
  print(result)
  
  # Clean up
  unlink(test_img)
}

# Example 4: Batch processing with progress
# ------------------------------------------

if (check_caption_provider_auth("openai")) {
  # Create multiple test images
  test_imgs <- sapply(1:3, function(i) {
    img_path <- tempfile(fileext = ".jpg")
    jpeg::writeJPEG(
      array(runif(100*100*3), dim = c(100, 100, 3)), 
      img_path
    )
    img_path
  })
  
  # Process with progress bar
  results <- caption_features_many(
    test_imgs,
    caption_provider = "openai",
    template = "alt_text",
    compute_embedding = TRUE,
    .progress = TRUE
  )
  
  print(results)
  
  # Compute similarity between captions
  if (nrow(results) > 1 && all(!is.na(results$embedding_dim))) {
    sim_matrix <- compute_caption_similarity(results)
    print(sim_matrix)
  }
  
  # Clean up
  unlink(test_imgs)
}

# Example 5: Multimodal features (visual + caption)
# --------------------------------------------------

if (check_caption_provider_auth("openai") && 
    requireNamespace("keras", quietly = TRUE)) {
  
  # Create test image
  test_img <- tempfile(fileext = ".jpg")
  jpeg::writeJPEG(array(runif(224*224*3), dim = c(224, 224, 3)), test_img)
  
  # Extract both visual and caption features
  multimodal <- extract_multimodal_features(
    test_img,
    visual_layers = c(15, 17),  # VGG layers
    caption_provider = "openai",
    caption_template = "dense",
    compute_embedding = TRUE
  )
  
  print(multimodal)
  
  # Clean up
  unlink(test_img)
}

# Example 6: Text embedding without captioning
# ---------------------------------------------

# Embed any text
if (nzchar(Sys.getenv("OPENAI_API_KEY"))) {
  texts <- c(
    "A beautiful sunset over the ocean",
    "Mountains covered in snow",
    "A bustling city street at night"
  )
  
  embeddings <- embed_text(texts, backend = "openai")
  print(dim(embeddings))  # Should be 3 x embedding_dim
  
  # Compute similarity
  sim <- cosine_similarity(embeddings[1,], embeddings[2,])
  cat("Similarity between first two texts:", sim, "\n")
}

cat("\nCaption feature integration complete!\n")