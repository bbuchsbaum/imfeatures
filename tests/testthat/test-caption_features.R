library(testthat)

context("caption_features")

# Test caption prompt generation
test_that("build_caption_prompt works correctly", {
  # Test basic prompt generation
  prompt <- build_caption_prompt(
    template = "dense",
    focus = c("objects", "colors"),
    min_words = 50,
    max_words = 100
  )

  expect_type(prompt, "character")
  expect_true(grepl("50-100 words", prompt))
  expect_true(grepl("objects", prompt, ignore.case = TRUE))
  expect_true(grepl("colors", prompt, ignore.case = TRUE))

  # Test all templates
  templates <- c("factual", "dense", "alt_text", "product", "art")
  for (tmpl in templates) {
    prompt <- build_caption_prompt(template = tmpl)
    expect_type(prompt, "character")
    expect_true(nchar(prompt) > 0)
  }

  # Test with extra instructions
  prompt <- build_caption_prompt(
    template = "factual",
    extra_instructions = "Focus on the background"
  )
  expect_true(grepl("Focus on the background", prompt))

  # Test with unknown focus areas (should warn)
  expect_warning(
    build_caption_prompt(focus = c("objects", "invalid_focus")),
    "unknown focus keyword"
  )
})

test_that("get_caption_templates returns correct values", {
  templates <- get_caption_templates()
  expect_equal(
    sort(templates),
    sort(c("factual", "dense", "alt_text", "product", "art"))
  )
})

test_that("get_caption_focus_areas returns correct values", {
  areas <- get_caption_focus_areas()
  expect_true("objects" %in% areas)
  expect_true("text" %in% areas)
  expect_true("layout" %in% areas)
  expect_true("colors" %in% areas)
})

# Test embedding functions
test_that("embed_text handles missing API keys gracefully", {
  # Temporarily unset API keys
  old_openai <- Sys.getenv("OPENAI_API_KEY")
  old_gemini <- Sys.getenv("GEMINI_API_KEY")
  old_hf <- Sys.getenv("HUGGINGFACE_API_KEY")

  Sys.setenv(OPENAI_API_KEY = "")
  Sys.setenv(GEMINI_API_KEY = "")
  Sys.setenv(HUGGINGFACE_API_KEY = "")

  expect_error(
    embed_text("test", backend = "openai"),
    "OPENAI_API_KEY not set"
  )

  expect_error(
    embed_text("test", backend = "gemini"),
    "GEMINI_API_KEY not set"
  )

  expect_error(
    embed_text("test", backend = "hf"),
    "HUGGINGFACE_API_KEY not set"
  )

  # Restore API keys
  Sys.setenv(OPENAI_API_KEY = old_openai)
  Sys.setenv(GEMINI_API_KEY = old_gemini)
  Sys.setenv(HUGGINGFACE_API_KEY = old_hf)
})

test_that("cosine_similarity computes correctly", {
  # Test vector-vector similarity
  v1 <- c(1, 0, 0)
  v2 <- c(0, 1, 0)
  expect_equal(cosine_similarity(v1, v2), 0)

  v3 <- c(1, 1, 0) / sqrt(2)
  expect_equal(cosine_similarity(v1, v3), 1 / sqrt(2), tolerance = 1e-6)

  # Test identical vectors
  expect_equal(cosine_similarity(v1, v1), 1)
})

# Test provider functions
test_that("get_default_caption_model returns sensible defaults", {
  expect_equal(get_default_caption_model("openai"), "gpt-4o-mini")
  expect_equal(get_default_caption_model("anthropic"), "claude-3-5-sonnet-20241022")
  expect_equal(get_default_caption_model("gemini"), "gemini-2.0-flash-exp")

  # Test all providers
  providers <- c(
    "openai", "anthropic", "gemini", "azure_openai",
    "ollama", "huggingface", "openrouter", "vllm"
  )
  for (p in providers) {
    model <- get_default_caption_model(p)
    expect_type(model, "character")
    expect_true(nchar(model) > 0)
  }
})

test_that("check_caption_provider_auth works", {
  # Ollama shouldn't need auth
  expect_true(check_caption_provider_auth("ollama"))

  # vLLM shouldn't need auth
  expect_true(check_caption_provider_auth("vllm"))

  # Others depend on env variables
  if (nzchar(Sys.getenv("OPENAI_API_KEY"))) {
    expect_true(check_caption_provider_auth("openai"))
  }
})

test_that("list_caption_providers returns data frame", {
  providers_df <- list_caption_providers()

  expect_s3_class(providers_df, "data.frame")
  expect_true("provider" %in% names(providers_df))
  expect_true("default_model" %in% names(providers_df))
  expect_true("auth_configured" %in% names(providers_df))
  expect_gt(nrow(providers_df), 0)
})

# Test main caption features function with mock
test_that("caption_features validates inputs", {
  # Test with missing file
  expect_error(
    caption_features("nonexistent.jpg"),
    "impath file\\(s\\) not found"
  )

  # Test with invalid provider (use temp file that exists)
  temp_img <- tempfile(fileext = ".jpg")
  writeLines("fake", temp_img)
  on.exit(unlink(temp_img))

  expect_error(
    caption_features(temp_img, caption_provider = "invalid_provider"),
    "'arg' should be one of"
  )
})

# Create a simple test image helper
create_test_image <- function(filename = "test_image.png", dir = tempdir()) {
  path <- file.path(dir, filename)
  if (!requireNamespace("png", quietly = TRUE)) {
    skip("png package not available")
  }
  img <- matrix(runif(64 * 64 * 3), 64, 64 * 3)
  png::writePNG(img, path)
  return(path)
}

test_that("caption_features errors on missing ellmer package", {
  skip_if(requireNamespace("ellmer", quietly = TRUE))

  test_img <- create_test_image()
  on.exit(unlink(test_img))

  expect_error(
    caption_features(test_img),
    "Package 'ellmer' is required"
  )
})
