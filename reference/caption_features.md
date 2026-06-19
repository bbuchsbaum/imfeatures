# Extract Caption Features from Images

Generate natural language captions for images and optionally compute
text embeddings for the captions. Uses state-of-the-art vision-language
models via the ellmer package.

## Usage

``` r
caption_features(
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
)
```

## Arguments

- impath:

  Path to image file(s).

- caption_provider:

  LLM provider for captioning. Options: "openai", "anthropic", "gemini",
  "azure_openai", "ollama", "huggingface", "openrouter", "vllm".

- caption_model:

  Model name for captioning. If NULL, uses provider default.

- template:

  Caption template. Options: "factual", "dense", "alt_text", "product",
  "art".

- focus:

  Character vector of focus areas. Options: "objects", "text", "layout",
  "colors", "materials", "actions", "people".

- min_words:

  Minimum words in caption.

- max_words:

  Maximum words in caption.

- temperature:

  Generation temperature (0-1).

- max_tokens:

  Maximum tokens to generate.

- seed:

  Random seed for reproducibility.

- extra_instructions:

  Additional prompt instructions.

- compute_embedding:

  Whether to compute text embeddings for captions.

- embedding_backend:

  Backend for embeddings: "openai", "gemini", "hf".

- embedding_model:

  Model for embeddings. If NULL, uses backend default.

- embedding_dim:

  Optional embedding dimensions.

- gemini_task:

  Task type for Gemini embeddings.

- echo:

  Ellmer echo mode.

## Value

A tibble with class \`imfeatures_feature_tbl\` containing: - image:
Image file path - caption: Generated caption text - embedding: Text
embedding vector (if compute_embedding = TRUE) - embedding_dim:
Embedding dimensions - metadata: List column with generation parameters

## Examples

``` r
if (FALSE) { # \dontrun{
# Basic caption generation
captions <- caption_features(
  "image.jpg",
  caption_provider = "openai",
  template = "dense"
)

# Caption with embeddings
features <- caption_features(
  c("img1.jpg", "img2.jpg"),
  caption_provider = "gemini",
  compute_embedding = TRUE,
  embedding_backend = "gemini"
)
} # }
```
