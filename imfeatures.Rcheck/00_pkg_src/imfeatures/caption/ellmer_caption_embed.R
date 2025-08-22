
# ellmer_caption_embed.R
# A tiny R helper to caption an image (with ellmer) and return a text embedding.
# - Captioning: via ellmer (supports OpenAI, Anthropic, Gemini/Vertex, etc.)
# - Embedding: via OpenAI (HTTR2 call) or Gemini (HTTR2 call)
# - No reticulate required

# Dependencies: ellmer, httr2, jsonlite, tibble, purrr, cli
# install.packages(c("ellmer","httr2","jsonlite","tibble","purrr","cli"))

suppressPackageStartupMessages({
  library(ellmer)
  library(httr2)
  library(jsonlite)
  library(tibble)
  library(purrr)
  library(cli)
})

# --------------------------
# Prompt templates & builder
# --------------------------

.caption_focus_map <- list(
  objects   = "Name concrete objects and their attributes (color, shape, material, count).",
  text      = "Transcribe any legible text exactly as it appears.",
  layout    = "Describe spatial arrangement (foreground/background, left/right, center, perspective).",
  colors    = "Mention prominent colors and relationships (dominant hues, accents).",
  materials = "Note textures and materials when visually evident.",
  actions   = "Describe actions or interactions, if any.",
  people    = "If people appear, describe non-identifying attributes (approximate age group, clothing); never guess identity."
)

.caption_negatives_default <- c(
  "Avoid speculation that is not visible in the image.",
  "Avoid brand/identity guesses unless text is plainly visible.",
  "Avoid subjective marketing language (premium, refined, elegant)."
)

.caption_templates <- list(
  factual = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Provide a precise, objective visual description of the image in ",
      min_words, "-", max_words, " words.\n",
      if (length(focus_lines)) paste0("Focus on:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- ", paste(c(negatives, "Write in complete sentences."), collapse = "\n- ")
    )
  },
  dense = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Provide a thorough, information-dense description suited for retrieval in ",
      min_words, "-", max_words, " words.\n",
      "Enumerate salient entities, attributes, colors, text, and composition.\n",
      if (length(focus_lines)) paste0("Emphasize:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Be concrete and specific; avoid opinions.\n- ",
      paste(c(negatives, "Prefer nouns/adjectives over metaphors."), collapse = "\n- ")
    )
  },
  alt_text = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Write screen-reader-friendly ALT TEXT in ",
      min_words, "-", max_words, " words, capturing the essential content.\n",
      if (length(focus_lines)) paste0("Include:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Prioritize what a blind user needs to know.\n- ",
      paste(c(negatives, "Keep tone neutral; no redundant 'Image of...' preface."), collapse = "\n- ")
    )
  },
  product = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Describe the product(s) in the image in ", min_words, "-", max_words, " words.\n",
      "Include form, materials, colorway, condition, key features, and any legible labeling.\n",
      if (length(focus_lines)) paste0("Emphasize:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- No pricing or marketing hype.\n- ",
      paste(c(negatives, "Do not invent specs you cannot see."), collapse = "\n- ")
    )
  },
  art = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Describe the artwork photographically and formally in ", min_words, "-", max_words, " words.\n",
      "Note medium, palette, composition, motifs, and visible inscriptions.\n",
      if (length(focus_lines)) paste0("Consider:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Avoid speculative provenance or biography.\n- ",
      paste(c(negatives, "If style is clear (e.g., cubist), state cautiously."), collapse = "\n- ")
    )
  }
)

build_caption_prompt <- function(
  template = c("factual","dense","alt_text","product","art"),
  focus = c("objects","text","layout","colors"),
  min_words = 40,
  max_words = 120,
  negatives = .caption_negatives_default,
  extra_instructions = NULL
) {
  template <- match.arg(template)
  unknown <- setdiff(focus, names(.caption_focus_map))
  if (length(unknown)) {
    cli_warn("Ignoring unknown focus keyword(s): {.val {unknown}}")
    focus <- intersect(focus, names(.caption_focus_map))
  }
  focus_lines <- unname(.caption_focus_map[focus])
  prompt <- .caption_templates[[template]](min_words, max_words, focus_lines, negatives)
  if (!is.null(extra_instructions) && nzchar(extra_instructions)) {
    prompt <- paste(prompt, "\nAdditional instructions:\n", extra_instructions)
  }
  prompt
}

# --------------------------
# Provider selection (ellmer)
# --------------------------

.select_chat <- function(provider = c("openai","anthropic","gemini","azure_openai","ollama","huggingface","openrouter","vllm"),
                         model,
                         system_prompt = NULL,
                         temperature = 0.2,
                         max_tokens = 512,
                         seed = NULL,
                         echo = "none",
                         api_args = list(),
                         api_headers = character()) {
  provider <- match.arg(provider)
  prm <- params(temperature = temperature, max_tokens = max_tokens, seed = seed)

  switch(provider,
    openai = chat_openai(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    anthropic = chat_anthropic(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    gemini = chat_google_gemini(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    azure_openai = chat_azure_openai(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    ollama = chat_ollama(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    huggingface = chat_huggingface(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    openrouter = chat_openrouter(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    vllm = chat_vllm(system_prompt = system_prompt, model = model, params = prm, echo = echo, api_args = api_args, api_headers = api_headers),
    stop("Unsupported provider: ", provider)
  )
}

# --------------------------
# Embedding backends (pure R)
# --------------------------

# OpenAI embeddings via REST (supports 'dimensions')
.embed_openai <- function(text,
                          model = "text-embedding-3-large",
                          dimensions = NULL,
                          base_url = "https://api.openai.com/v1",
                          api_key = Sys.getenv("OPENAI_API_KEY"),
                          organization = Sys.getenv("OPENAI_ORG")) {
  if (!nzchar(api_key)) stop("OPENAI_API_KEY not set")
  req <- request(base_url) |>
    req_url_path_append("embeddings") |>
    req_headers(Authorization = paste("Bearer", api_key)) |>
    req_body_json(list(
      model = model,
      input = text,
      !!!(if (!is.null(dimensions)) list(dimensions = as.integer(dimensions)) else list())
    ), auto_unbox = TRUE)
  if (nzchar(organization)) req <- req |> req_headers("OpenAI-Organization" = organization)
  resp <- req_perform(req)
  res <- resp_body_json(resp)
  unlist(res$data[[1]]$embedding, use.names = FALSE)
}

# Gemini embeddings via REST (supports output_dimensionality & task_type)
.embed_gemini <- function(text,
                          model = "gemini-embedding-001",
                          output_dimensionality = NULL,
                          task_type = NULL,
                          base_url = "https://generativelanguage.googleapis.com/v1beta",
                          api_key = Sys.getenv("GEMINI_API_KEY")) {
  if (!nzchar(api_key)) stop("GEMINI_API_KEY not set")
  url <- paste0(base_url, "/models/", model, ":embedContent")
  body <- list(
    contents = list(list(parts = list(list(text = text)))),
    embedding_config = list()
  )
  if (!is.null(output_dimensionality)) body$embedding_config$output_dimensionality <- as.integer(output_dimensionality)
  if (!is.null(task_type)) body$embedding_config$task_type <- task_type

  resp <- request(url) |>
    req_headers("x-goog-api-key" = api_key, "Content-Type" = "application/json") |>
    req_body_json(body, auto_unbox = TRUE) |>
    req_perform()

  res <- resp_body_json(resp)
  # Gemini returns a single embedding under $embedding or $embeddings[[1]]
  emb <- res$embedding$values %||% res$embeddings[[1]]$values
  v <- unlist(emb, use.names = FALSE)

  # If non-3072 dims, normalize vector (per docs) for cosine similarity
  if (!is.null(output_dimensionality) && output_dimensionality != 3072) {
    norm <- sqrt(sum(v * v))
    if (is.finite(norm) && norm > 0) v <- v / norm
  }
  v
}

# Hugging Face Inference API (optional) — feature-extraction pipeline
# Requires HUGGINGFACE_API_KEY; model e.g., "sentence-transformers/all-MiniLM-L6-v2"
.embed_hf <- function(text,
                      model = "sentence-transformers/all-MiniLM-L6-v2",
                      api_key = Sys.getenv("HUGGINGFACE_API_KEY"),
                      base_url = "https://api-inference.huggingface.co/pipeline/feature-extraction") {
  if (!nzchar(api_key)) stop("HUGGINGFACE_API_KEY not set")
  url <- paste0(base_url, "/", model)
  resp <- request(url) |>
    req_headers(Authorization = paste("Bearer", api_key), "Content-Type" = "application/json") |>
    req_body_json(list(inputs = text), auto_unbox = TRUE) |>
    req_perform()
  m <- resp_body_json(resp)
  # Some models return list(list(...)); flatten to numeric vector
  v <- unlist(m[[1]], use.names = FALSE)
  v
}

# --------------------------
# Main API
# --------------------------

#' Caption an image and return a caption + embedding
#'
#' @param image File path (string) to an image
#' @param caption_provider One of: "openai","anthropic","gemini","azure_openai","ollama","huggingface","openrouter","vllm"
#' @param caption_model Provider-specific model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet", "gemini-2.5-flash")
#' @param template Caption template: "factual","dense","alt_text","product","art"
#' @param focus Character vector of focus keywords (see names(.caption_focus_map))
#' @param min_words,max_words Length bounds for caption
#' @param temperature,max_tokens,seed Usual model parameters
#' @param extra_instructions Extra freeform instructions appended to the prompt
#' @param embedding_backend One of: "openai","gemini","hf"
#' @param embedding_model Backend-specific model (e.g., "text-embedding-3-large", "gemini-embedding-001")
#' @param embedding_dim Optional dimension (OpenAI: dimensions; Gemini: output_dimensionality)
#' @param gemini_task Optional Gemini task type (e.g., "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY")
#' @return A list with caption (string), embedding (numeric), and metadata (list)
caption_and_embed <- function(
  image,
  caption_provider = "openai",
  caption_model = "gpt-4o-mini",
  template = "dense",
  focus = c("objects","text","layout","colors"),
  min_words = 60,
  max_words = 120,
  temperature = 0.2,
  max_tokens = 512,
  seed = NULL,
  extra_instructions = NULL,
  embedding_backend = "openai",
  embedding_model = if (embedding_backend == "openai") "text-embedding-3-large" else "gemini-embedding-001",
  embedding_dim = NULL,
  gemini_task = NULL,
  echo = "none"
) {
  if (!file.exists(image)) stop("Image not found: ", image)
  sys <- "You are a careful vision assistant. Be precise, concrete, and avoid speculation."
  prompt <- build_caption_prompt(template = template, focus = focus,
                                 min_words = min_words, max_words = max_words,
                                 extra_instructions = extra_instructions)

  chat <- .select_chat(
    provider = caption_provider,
    model = caption_model,
    system_prompt = sys,
    temperature = temperature,
    max_tokens = max_tokens,
    seed = seed,
    echo = echo
  )

  # Ask with image
  cap <- chat$chat(content_image_file(image), prompt)

  # Compute embedding
  emb <- switch(embedding_backend,
    openai = .embed_openai(cap, model = embedding_model, dimensions = embedding_dim),
    gemini = .embed_gemini(cap, model = embedding_model, output_dimensionality = embedding_dim, task_type = gemini_task),
    hf     = .embed_hf(cap, model = embedding_model),
    stop("Unsupported embedding_backend: ", embedding_backend)
  )

  list(
    caption = cap,
    embedding = as.numeric(emb),
    metadata = list(
      caption_provider = caption_provider,
      caption_model = caption_model,
      embedding_backend = embedding_backend,
      embedding_model = embedding_model,
      embedding_dim = if (is.null(embedding_dim)) length(emb) else embedding_dim,
      template = template,
      focus = focus,
      min_words = min_words,
      max_words = max_words,
      temperature = temperature,
      seed = seed
    )
  )
}

#' Vectorized helper over many images
#' Returns a tibble with image, caption, and embedding (list-col)
caption_and_embed_many <- function(
  images,
  ...,
  .progress = interactive()
) {
  images <- as.character(images)
  res <- purrr::imap(images, function(img, i) {
    if (.progress) cli_inform(paste0("[", i, "/", length(images), "] ", basename(img)))
    tryCatch(
      {
        out <- caption_and_embed(img, ...)
        tibble::tibble(
          image = img,
          caption = out$caption,
          embedding = list(out$embedding),
          embedding_dim = length(out$embedding)
        )
      },
      error = function(e) {
        tibble::tibble(
          image = img,
          caption = paste0("ERROR: ", conditionMessage(e)),
          embedding = list(numeric(0)),
          embedding_dim = NA_integer_
        )
      }
    )
  })
  dplyr::bind_rows(res)
}

# Utilities
`%||%` <- function(x, y) if (!is.null(x)) x else y

# End of file
