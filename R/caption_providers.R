#' Caption Provider Management
#'
#' @description
#' Functions for managing LLM providers for image captioning using the ellmer package.
#'
#' @name caption-providers
#' @keywords internal

#' Select and configure chat provider
#'
#' @param provider Provider name: "openai", "anthropic", "gemini", "azure_openai",
#'   "ollama", "huggingface", "openrouter", or "vllm".
#' @param model Model name specific to the provider.
#' @param system_prompt System prompt for the model.
#' @param temperature Temperature parameter for generation (0-1).
#' @param max_tokens Maximum tokens to generate.
#' @param seed Random seed for reproducibility.
#' @param echo Echo mode for ellmer.
#' @param api_args Additional API arguments.
#' @param api_headers Additional API headers.
#'
#' @return An ellmer chat object configured for the specified provider.
#' @keywords internal
.select_chat <- function(
  provider = c(
    "openai", "anthropic", "gemini", "azure_openai",
    "ollama", "huggingface", "openrouter", "vllm"
  ),
  model,
  system_prompt = NULL,
  temperature = 0.2,
  max_tokens = 512,
  seed = NULL,
  echo = "none",
  api_args = list(),
  api_headers = character()
) {
  provider <- match.arg(provider)

  # Check if ellmer is available
  if (!requireNamespace("ellmer", quietly = TRUE)) {
    stop(
      "Package 'ellmer' is required for caption generation. ",
      "Please install it with: install.packages('ellmer')"
    )
  }

  # Prepare API args with temperature and max_tokens
  if (length(api_args) == 0) {
    api_args <- list()
  }
  api_args$temperature <- temperature
  api_args$max_tokens <- max_tokens

  # Select appropriate chat function based on provider
  chat <- switch(provider,
    openai = ellmer::chat_openai(
      system_prompt = system_prompt,
      model = model,
      seed = seed,
      api_args = api_args,
      echo = echo
    ),
    anthropic = ellmer::chat_claude(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    gemini = ellmer::chat_google_gemini(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    azure_openai = ellmer::chat_azure_openai(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    ollama = ellmer::chat_ollama(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    huggingface = {
      # HuggingFace might not be directly supported
      stop("HuggingFace provider not directly supported by ellmer. Use OpenRouter instead.")
    },
    openrouter = ellmer::chat_openrouter(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    vllm = ellmer::chat_vllm(
      system_prompt = system_prompt,
      model = model,
      api_args = api_args,
      echo = echo
    ),
    stop("Unsupported provider: ", provider)
  )

  chat
}

#' Get default model for provider
#'
#' @param provider Provider name.
#' @return Default model name for the provider.
#' @export
get_default_caption_model <- function(provider) {
  provider <- match.arg(provider, c(
    "openai", "anthropic", "gemini",
    "azure_openai", "ollama", "huggingface",
    "openrouter", "vllm"
  ))

  switch(provider,
    openai = "gpt-4o-mini",
    anthropic = "claude-3-5-sonnet-20241022",
    gemini = "gemini-2.0-flash-exp",
    azure_openai = "gpt-4o-mini",
    ollama = "llava:latest",
    huggingface = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    openrouter = "openai/gpt-4o-mini",
    vllm = "llava-hf/llava-1.5-7b-hf",
    "gpt-4o-mini" # fallback
  )
}

#' Check if provider API key is configured
#'
#' @param provider Provider name.
#' @return Logical indicating if API key is set.
#' @export
check_caption_provider_auth <- function(provider) {
  provider <- match.arg(provider, c(
    "openai", "anthropic", "gemini",
    "azure_openai", "ollama", "huggingface",
    "openrouter", "vllm"
  ))

  key_var <- switch(provider,
    openai = "OPENAI_API_KEY",
    anthropic = "ANTHROPIC_API_KEY",
    gemini = "GEMINI_API_KEY",
    azure_openai = "AZURE_OPENAI_API_KEY",
    huggingface = "HUGGINGFACE_API_KEY",
    openrouter = "OPENROUTER_API_KEY",
    ollama = "", # Ollama doesn't need API key
    vllm = "", # vLLM is self-hosted
    ""
  )

  if (key_var == "") {
    return(TRUE) # No API key needed
  }

  nzchar(Sys.getenv(key_var))
}

#' List available caption providers
#'
#' @return A data frame with provider information.
#' @export
list_caption_providers <- function() {
  providers <- c(
    "openai", "anthropic", "gemini", "azure_openai",
    "ollama", "huggingface", "openrouter", "vllm"
  )

  df <- data.frame(
    provider = providers,
    default_model = sapply(providers, get_default_caption_model),
    auth_configured = sapply(providers, check_caption_provider_auth),
    stringsAsFactors = FALSE
  )

  row.names(df) <- NULL
  df
}
