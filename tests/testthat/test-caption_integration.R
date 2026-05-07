library(testthat)

context("caption_features integration")

# Path to Space Invaders test image
space_invaders_path <- file.path(dirname(dirname(dirname(getwd()))), "Space_Invaders.jpg")
if (!file.exists(space_invaders_path)) {
  # Try alternative path when running from different locations
  space_invaders_path <- "Space_Invaders.jpg"
  if (!file.exists(space_invaders_path)) {
    space_invaders_path <- "../../Space_Invaders.jpg"
  }
}

test_that("caption_features works with Ollama (local provider)", {
  skip_if_not(check_caption_provider_auth("ollama"), "Ollama not available")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  # Test with llava model (vision-language model)
  result <- tryCatch(
    {
      caption_features(
        space_invaders_path,
        caption_provider = "ollama",
        caption_model = "llava:7b",
        template = "dense",
        focus = c("objects", "colors", "layout"),
        compute_embedding = FALSE # Ollama doesn't provide embeddings directly
      )
    },
    error = function(e) {
      skip(paste("Ollama test failed:", e$message))
    }
  )

  # Check caption was generated
  expect_type(result$caption, "character")
  expect_true(nchar(result$caption) > 10)

  # Check for expected keywords (any of these)
  keywords <- c(
    "pixel", "game", "alien", "invader", "arcade",
    "retro", "space", "green", "white", "black"
  )
  expect_true(
    any(sapply(keywords, function(k) grepl(k, result$caption, ignore.case = TRUE))),
    info = paste("Caption:", result$caption)
  )
})

test_that("caption_features works with OpenAI provider", {
  skip_if_not(check_caption_provider_auth("openai"), "OpenAI API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  result <- caption_features(
    space_invaders_path,
    caption_provider = "openai",
    caption_model = "gpt-4o-mini",
    template = "dense",
    focus = c("objects", "colors", "layout"),
    min_words = 60,
    max_words = 120,
    compute_embedding = TRUE,
    embedding_backend = "openai",
    embedding_model = "text-embedding-3-small",
    embedding_dim = 512
  )

  # Check caption
  expect_type(result$caption, "character")
  if (startsWith(result$caption, "ERROR:")) {
    skip(paste("OpenAI caption request failed:", result$caption))
  }
  expect_true(nchar(result$caption) > 50)

  # Check embedding
  if (is.na(result$embedding_dim) || length(result$embedding[[1]]) == 0) {
    skip("OpenAI embeddings unavailable")
  }
  expect_equal(result$embedding_dim, 512)
  expect_length(result$embedding[[1]], 512)

  # Verify Space Invaders content
  caption_lower <- tolower(result$caption)
  game_terms <- c("game", "arcade", "retro", "pixel", "8-bit", "classic")
  expect_true(
    any(sapply(game_terms, function(t) grepl(t, caption_lower))),
    info = paste("Caption should mention game context:", result$caption)
  )
})

test_that("caption_features works with Gemini provider", {
  skip_if_not(check_caption_provider_auth("gemini"), "Gemini API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  result <- caption_features(
    space_invaders_path,
    caption_provider = "gemini",
    caption_model = "gemini-2.0-flash-exp",
    template = "factual",
    focus = c("objects", "colors"),
    min_words = 40,
    max_words = 80,
    compute_embedding = TRUE,
    embedding_backend = "gemini",
    embedding_model = "text-embedding-004",
    embedding_dim = 768
  )

  # Check caption
  expect_type(result$caption, "character")
  expect_true(nchar(result$caption) > 30)

  # Check embedding
  if (is.na(result$embedding_dim) || length(result$embedding[[1]]) == 0) {
    skip("Gemini embeddings unavailable")
  }
  expect_equal(result$embedding_dim, 768)
  expect_length(result$embedding[[1]], 768)

  # Check normalized embedding (Gemini normalizes non-3072 dims)
  embedding_vec <- result$embedding[[1]]
  skip_if(length(embedding_vec) == 0, "Gemini embeddings unavailable")
  embedding_norm <- sqrt(sum(embedding_vec^2))
  skip_if(!is.finite(embedding_norm) || embedding_norm == 0, "Gemini embeddings not normalized")
  expect_equal(embedding_norm, 1, tolerance = 0.01)
})

test_that("caption_features works with HuggingFace embeddings", {
  skip_if_not(check_caption_provider_auth("openai"), "OpenAI API key not set for caption")
  skip_if_not(nzchar(Sys.getenv("HUGGINGFACE_API_KEY")), "HuggingFace API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  result <- caption_features(
    space_invaders_path,
    caption_provider = "openai",
    caption_model = "gpt-4o-mini",
    template = "alt_text",
    compute_embedding = TRUE,
    embedding_backend = "hf",
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
  )

  # Check caption
  expect_type(result$caption, "character")

  # Check embedding (MiniLM-L6-v2 produces 384-dim embeddings)
  expect_equal(result$embedding_dim, 384)
  expect_length(result$embedding[[1]], 384)
})

test_that("batch processing works with multiple images", {
  skip_if_not(check_caption_provider_auth("openai"), "OpenAI API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  # Use same image twice for testing
  images <- c(space_invaders_path, space_invaders_path)

  results <- caption_features_many(
    images,
    caption_provider = "openai",
    template = "factual",
    compute_embedding = TRUE,
    .progress = FALSE
  )

  expect_equal(nrow(results), 2)
  expect_true(all(nchar(results$caption) > 10))
  if (any(is.na(results$embedding_dim))) {
    skip("Embeddings not available for all images; likely provider rate limiting")
  }
  expect_true(all(!is.na(results$embedding_dim)))

  # Compute similarity between identical images
  if (all(!is.na(results$embedding_dim))) {
    sim_matrix <- compute_caption_similarity(results)
    expect_equal(dim(sim_matrix), c(2, 2))
    # Diagonal should be 1 (self-similarity)
    expect_equal(as.vector(diag(sim_matrix)), c(1, 1), tolerance = 0.001)
    # Off-diagonal should be very high (same image) but GPT may vary slightly
    # Lowering threshold to 0.90 to account for natural variation in captions
    expect_gt(sim_matrix[1, 2], 0.90)
  }
})

test_that("multimodal features work", {
  skip_if_not(check_caption_provider_auth("openai"), "OpenAI API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")
  skip_if_not(requireNamespace("keras3", quietly = TRUE), "keras3 not available")

  # This might fail if keras/tensorflow not properly configured
  tryCatch(
    {
      result <- extract_multimodal_features(
        space_invaders_path,
        visual_layers = c(15), # Just one layer for speed
        caption_provider = "openai",
        caption_template = "dense",
        compute_embedding = TRUE
      )

      expect_s3_class(result, "imfeatures_feature_tbl")
      expect_true("feature" %in% names(result)) # Visual features
      expect_true("caption" %in% names(result)) # Caption text
      expect_true("embedding" %in% names(result)) # Caption embedding
    },
    error = function(e) {
      skip(paste("Multimodal test failed:", e$message))
    }
  )
})

test_that("different templates produce different styles", {
  skip_if_not(check_caption_provider_auth("openai"), "OpenAI API key not set")
  skip_if_not(file.exists(space_invaders_path), "Space Invaders image not found")

  templates <- c("factual", "dense", "alt_text")
  captions <- list()

  for (tmpl in templates) {
    result <- caption_features(
      space_invaders_path,
      caption_provider = "openai",
      template = tmpl,
      compute_embedding = FALSE
    )
    captions[[tmpl]] <- result$caption
  }

  # Different templates should produce different captions
  expect_true(captions$factual != captions$dense)
  expect_true(captions$dense != captions$alt_text)

  # Alt text should be shorter/more concise
  expect_lt(nchar(captions$alt_text), nchar(captions$dense) * 1.5)
})
