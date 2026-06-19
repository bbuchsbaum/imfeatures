# Compute text embeddings

Compute text embeddings

## Usage

``` r
embed_text(
  text,
  backend = c("openai", "gemini", "hf"),
  model = NULL,
  dimensions = NULL,
  task_type = NULL,
  normalize = FALSE,
  api_key = NULL
)
```

## Arguments

- text:

  Character string or vector of strings to embed.

- backend:

  Embedding backend to use: "openai", "gemini", or "hf".

- model:

  Model name specific to the backend.

- dimensions:

  Optional dimension specification (backend-specific).

- task_type:

  Optional task type (Gemini only).

- normalize:

  Logical; whether to normalize embeddings to unit length.

- api_key:

  Optional API key override.

## Value

For a single text: numeric vector. For multiple texts: matrix with one
row per text.

## Examples

``` r
if (FALSE) { # \dontrun{
# Single text embedding
emb <- embed_text("A beautiful sunset", backend = "openai")

# Multiple texts
texts <- c("First caption", "Second caption")
emb_matrix <- embed_text(texts, backend = "gemini", model = "text-embedding-004")
} # }
```
