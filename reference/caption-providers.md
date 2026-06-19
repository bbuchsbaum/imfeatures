# Caption Provider Management

Functions for managing LLM providers for image captioning using the
ellmer package.

## Usage

``` r
.select_chat(
  provider = c("openai", "anthropic", "gemini", "azure_openai", "ollama", "huggingface",
    "openrouter", "vllm"),
  model,
  system_prompt = NULL,
  temperature = 0.2,
  max_tokens = 512,
  seed = NULL,
  echo = "none",
  api_args = list(),
  api_headers = character()
)
```

## Arguments

- provider:

  Provider name: "openai", "anthropic", "gemini", "azure_openai",
  "ollama", "huggingface", "openrouter", or "vllm".

- model:

  Model name specific to the provider.

- system_prompt:

  System prompt for the model.

- temperature:

  Temperature parameter for generation (0-1).

- max_tokens:

  Maximum tokens to generate.

- seed:

  Random seed for reproducibility.

- echo:

  Echo mode for ellmer.

- api_args:

  Additional API arguments.

- api_headers:

  Additional API headers.

## Value

An ellmer chat object configured for the specified provider.
