# Gemini text embeddings

Gemini text embeddings

## Usage

``` r
.embed_gemini(
  text,
  model = "text-embedding-004",
  output_dimensionality = NULL,
  task_type = NULL,
  base_url = "https://generativelanguage.googleapis.com/v1beta",
  api_key = Sys.getenv("GEMINI_API_KEY")
)
```

## Arguments

- text:

  Character string to embed.

- model:

  Model name (e.g., "text-embedding-004", "gemini-embedding-001").

- output_dimensionality:

  Optional output dimensions.

- task_type:

  Optional task type (e.g., "RETRIEVAL_DOCUMENT",
  "SEMANTIC_SIMILARITY").

- base_url:

  Gemini API base URL.

- api_key:

  Gemini API key.

## Value

Numeric vector of embeddings.
