# Test Caption Features with Space Invaders Image
# This script demonstrates caption generation using the classic Space Invaders arcade game image

library(imfeatures)

# Path to Space Invaders image
img_path <- "Space_Invaders.jpg"
if (!file.exists(img_path)) {
  img_path <- "../Space_Invaders.jpg"
  if (!file.exists(img_path)) {
    stop("Space_Invaders.jpg not found. Please ensure it's in the package root directory.")
  }
}

cat("Testing caption features with Space Invaders image\n")
cat("Image path:", img_path, "\n\n")

# Helper function to test a provider
test_provider <- function(provider, model = NULL, embedding_backend = NULL) {
  cat(paste0("\n", strrep("=", 50), "\n"))
  cat("Testing provider:", provider, "\n")
  
  if (!check_caption_provider_auth(provider)) {
    cat("  ⚠️  API key not configured for", provider, "\n")
    cat("  Set environment variable for this provider to test.\n")
    return(NULL)
  }
  
  if (is.null(model)) {
    model <- get_default_caption_model(provider)
  }
  cat("  Model:", model, "\n")
  
  # Determine embedding backend
  if (is.null(embedding_backend)) {
    embedding_backend <- if (provider == "gemini") "gemini" else "openai"
  }
  
  # Should we compute embeddings?
  compute_emb <- provider %in% c("openai", "gemini", "anthropic") ||
                 embedding_backend == "hf"
  
  tryCatch({
    # Test different templates
    templates <- c("factual", "dense", "alt_text")
    results <- list()
    
    for (template in templates) {
      cat("\n  Template:", template, "\n")
      
      result <- caption_features(
        img_path,
        caption_provider = provider,
        caption_model = model,
        template = template,
        focus = c("objects", "colors", "layout"),
        min_words = if (template == "alt_text") 30 else 60,
        max_words = if (template == "alt_text") 60 else 120,
        compute_embedding = compute_emb && template == "dense",  # Only embed dense
        embedding_backend = embedding_backend
      )
      
      # Display caption
      cat("  Caption:", strwrap(result$caption, width = 70, prefix = "    "), sep = "\n")
      
      if (compute_emb && template == "dense" && !is.na(result$embedding_dim)) {
        cat("  Embedding dims:", result$embedding_dim, "\n")
      }
      
      results[[template]] <- result
    }
    
    return(results)
    
  }, error = function(e) {
    cat("  ❌ Error:", e$message, "\n")
    return(NULL)
  })
}

# Test all available providers
cat("\n🎮 SPACE INVADERS CAPTION TESTING 🎮\n")
cat("=====================================\n")

# Store results for comparison
all_results <- list()

# 1. Test Ollama (local, no API key needed)
cat("\n1. Testing Ollama (local provider)...")
if (system2("which", "ollama", stdout = FALSE, stderr = FALSE) == 0) {
  cat(" ✓ Ollama installed\n")
  # Check if llava model is available
  models <- system2("ollama", "list", stdout = TRUE, stderr = FALSE)
  if (any(grepl("llava", models))) {
    all_results$ollama <- test_provider("ollama", "llava:7b")
  } else {
    cat("  ℹ️  llava model not found. Install with: ollama pull llava:7b\n")
  }
} else {
  cat(" ⚠️  Ollama not installed\n")
  cat("  Install from: https://ollama.ai\n")
}

# 2. Test OpenAI
all_results$openai <- test_provider("openai", "gpt-4o-mini", "openai")

# 3. Test Gemini
all_results$gemini <- test_provider("gemini", "gemini-2.0-flash-exp", "gemini")

# 4. Test Anthropic
all_results$anthropic <- test_provider("anthropic", "claude-3-5-sonnet-20241022", "openai")

# 5. Test HuggingFace
if (nzchar(Sys.getenv("HUGGINGFACE_API_KEY"))) {
  all_results$huggingface <- test_provider("huggingface", 
                                          "meta-llama/Llama-3.2-11B-Vision-Instruct",
                                          "hf")
}

# Compare results
cat(paste0("\n", strrep("=", 50), "\n"))
cat("COMPARISON OF RESULTS\n")
cat(strrep("=", 50), "\n")

# Extract dense captions for comparison
dense_captions <- list()
for (provider in names(all_results)) {
  if (!is.null(all_results[[provider]])) {
    if ("dense" %in% names(all_results[[provider]])) {
      dense_captions[[provider]] <- all_results[[provider]]$dense$caption
    }
  }
}

if (length(dense_captions) > 0) {
  cat("\nDense template captions by provider:\n\n")
  for (provider in names(dense_captions)) {
    cat(toupper(provider), ":\n")
    cat(strwrap(dense_captions[[provider]], width = 70, prefix = "  "), sep = "\n")
    cat("\n")
  }
  
  # Analyze common themes
  cat("\n📊 CONTENT ANALYSIS\n")
  cat(strrep("-", 20), "\n")
  
  all_text <- paste(unlist(dense_captions), collapse = " ")
  all_lower <- tolower(all_text)
  
  # Check for key terms
  terms <- list(
    "Game references" = c("game", "arcade", "video game", "gaming", "retro"),
    "Space Invaders" = c("space invaders", "invaders", "aliens", "invader"),
    "Visual style" = c("pixel", "8-bit", "pixelated", "retro", "classic"),
    "Colors" = c("green", "white", "black", "bright green", "neon"),
    "Objects" = c("spacecraft", "ship", "barrier", "shield", "defense"),
    "Layout" = c("row", "formation", "grid", "arranged", "bottom")
  )
  
  for (category in names(terms)) {
    found <- terms[[category]][sapply(terms[[category]], function(t) grepl(t, all_lower))]
    if (length(found) > 0) {
      cat("✓", category, ":", paste(found, collapse = ", "), "\n")
    }
  }
  
  # Compute embedding similarities if available
  providers_with_embeddings <- names(all_results)[
    sapply(all_results, function(r) {
      !is.null(r) && "dense" %in% names(r) && 
      !is.na(r$dense$embedding_dim) && length(r$dense$embedding[[1]]) > 0
    })
  ]
  
  if (length(providers_with_embeddings) >= 2) {
    cat("\n📐 EMBEDDING SIMILARITIES\n")
    cat(strrep("-", 20), "\n")
    
    # Create a combined tibble for similarity computation
    emb_results <- lapply(providers_with_embeddings, function(p) {
      res <- all_results[[p]]$dense
      res$image <- paste0("Space_Invaders_", p)
      res
    })
    
    combined <- do.call(rbind, emb_results)
    
    # Ensure all embeddings have same dimension by using first provider's dimension
    target_dim <- combined$embedding_dim[1]
    if (all(combined$embedding_dim == target_dim)) {
      sim_matrix <- compute_caption_similarity(combined)
      
      # Display similarity matrix
      for (i in 1:nrow(sim_matrix)) {
        for (j in i:ncol(sim_matrix)) {
          if (i != j) {
            cat(sprintf("%s vs %s: %.3f\n", 
                       providers_with_embeddings[i],
                       providers_with_embeddings[j],
                       sim_matrix[i, j]))
          }
        }
      }
    } else {
      cat("Cannot compare embeddings with different dimensions:\n")
      for (p in providers_with_embeddings) {
        cat("  ", p, ":", all_results[[p]]$dense$embedding_dim, "dims\n")
      }
    }
  }
}

cat("\n✅ Testing complete!\n")

# Save results for further analysis
if (length(dense_captions) > 0) {
  results_df <- data.frame(
    provider = names(dense_captions),
    caption = unlist(dense_captions),
    stringsAsFactors = FALSE
  )
  
  cat("\nResults saved to: space_invaders_captions.csv\n")
  write.csv(results_df, "space_invaders_captions.csv", row.names = FALSE)
}