# HuggingFace text embeddings via Inference API

HuggingFace text embeddings via Inference API

## Usage

``` r
.embed_hf(
  text,
  model = "sentence-transformers/all-MiniLM-L6-v2",
  api_key = Sys.getenv("HUGGINGFACE_API_KEY"),
  base_url = "https://api-inference.huggingface.co/pipeline/feature-extraction"
)
```

## Arguments

- text:

  Character string to embed.

- model:

  Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2").

- api_key:

  HuggingFace API key.

- base_url:

  HuggingFace Inference API base URL.

## Value

Numeric vector of embeddings.
