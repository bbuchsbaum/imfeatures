# Text Embedding Functions

Functions for computing text embeddings using various backends including
OpenAI, Google Gemini, and HuggingFace sentence transformers.

## Usage

``` r
.embed_openai(
  text,
  model = "text-embedding-3-large",
  dimensions = NULL,
  base_url = "https://api.openai.com/v1",
  api_key = Sys.getenv("OPENAI_API_KEY"),
  organization = Sys.getenv("OPENAI_ORG")
)
```

## Arguments

- text:

  Character string to embed.

- model:

  Model name (e.g., "text-embedding-3-large").

- dimensions:

  Optional embedding dimensions.

- base_url:

  OpenAI API base URL.

- api_key:

  OpenAI API key.

- organization:

  Optional OpenAI organization ID.

## Value

Numeric vector of embeddings.
