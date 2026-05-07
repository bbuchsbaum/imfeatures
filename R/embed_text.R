#' Text Embedding Functions
#'
#' @description
#' Functions for computing text embeddings using various backends including
#' OpenAI, Google Gemini, and HuggingFace sentence transformers.
#'
#' @name embed-text
#' @keywords internal

#' OpenAI text embeddings
#'
#' @param text Character string to embed.
#' @param model Model name (e.g., "text-embedding-3-large").
#' @param dimensions Optional embedding dimensions.
#' @param base_url OpenAI API base URL.
#' @param api_key OpenAI API key.
#' @param organization Optional OpenAI organization ID.
#'
#' @return Numeric vector of embeddings.
#' @keywords internal
.embed_openai <- function(
  text,
  model = "text-embedding-3-large",
  dimensions = NULL,
  base_url = "https://api.openai.com/v1",
  api_key = Sys.getenv("OPENAI_API_KEY"),
  organization = Sys.getenv("OPENAI_ORG")
) {
  if (!nzchar(api_key)) {
    stop("OPENAI_API_KEY not set. Please set it using Sys.setenv(OPENAI_API_KEY = 'your-key')")
  }

  req <- httr2::request(base_url) |>
    httr2::req_url_path_append("embeddings") |>
    httr2::req_headers(Authorization = paste("Bearer", api_key))

  body <- list(model = model, input = text)
  if (!is.null(dimensions)) {
    body$dimensions <- as.integer(dimensions)
  }

  req <- req |> httr2::req_body_json(body, auto_unbox = TRUE)

  if (nzchar(organization)) {
    req <- req |> httr2::req_headers("OpenAI-Organization" = organization)
  }

  resp <- httr2::req_perform(req)
  res <- httr2::resp_body_json(resp)
  unlist(res$data[[1]]$embedding, use.names = FALSE)
}

#' Gemini text embeddings
#'
#' @param text Character string to embed.
#' @param model Model name (e.g., "text-embedding-004", "gemini-embedding-001").
#' @param output_dimensionality Optional output dimensions.
#' @param task_type Optional task type (e.g., "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY").
#' @param base_url Gemini API base URL.
#' @param api_key Gemini API key.
#'
#' @return Numeric vector of embeddings.
#' @keywords internal
.embed_gemini <- function(
  text,
  model = "text-embedding-004",
  output_dimensionality = NULL,
  task_type = NULL,
  base_url = "https://generativelanguage.googleapis.com/v1beta",
  api_key = Sys.getenv("GEMINI_API_KEY")
) {
  if (!nzchar(api_key)) {
    stop("GEMINI_API_KEY not set. Please set it using Sys.setenv(GEMINI_API_KEY = 'your-key')")
  }

  url <- paste0(base_url, "/models/", model, ":embedContent")

  body <- list(
    contents = list(list(parts = list(list(text = text)))),
    embedding_config = list()
  )

  if (!is.null(output_dimensionality)) {
    body$embedding_config$output_dimensionality <- as.integer(output_dimensionality)
  }

  if (!is.null(task_type)) {
    body$embedding_config$task_type <- task_type
  }

  resp <- httr2::request(url) |>
    httr2::req_headers("x-goog-api-key" = api_key, "Content-Type" = "application/json") |>
    httr2::req_body_json(body, auto_unbox = TRUE) |>
    httr2::req_perform()

  res <- httr2::resp_body_json(resp)

  # Handle different response structures
  emb <- res$embedding$values
  if (is.null(emb)) {
    emb <- res$embeddings[[1]]$values
  }

  v <- unlist(emb, use.names = FALSE)

  # Normalize if non-3072 dimensions (per Gemini docs)
  if (!is.null(output_dimensionality) && output_dimensionality != 3072) {
    norm <- sqrt(sum(v * v))
    if (is.finite(norm) && norm > 0) {
      v <- v / norm
    }
  }

  v
}

#' HuggingFace text embeddings via Inference API
#'
#' @param text Character string to embed.
#' @param model Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
#' @param api_key HuggingFace API key.
#' @param base_url HuggingFace Inference API base URL.
#'
#' @return Numeric vector of embeddings.
#' @keywords internal
.embed_hf <- function(
  text,
  model = "sentence-transformers/all-MiniLM-L6-v2",
  api_key = Sys.getenv("HUGGINGFACE_API_KEY"),
  base_url = "https://api-inference.huggingface.co/pipeline/feature-extraction"
) {
  if (!nzchar(api_key)) {
    stop("HUGGINGFACE_API_KEY not set. Please set it using Sys.setenv(HUGGINGFACE_API_KEY = 'your-key')")
  }

  url <- paste0(base_url, "/", model)

  resp <- httr2::request(url) |>
    httr2::req_headers(
      Authorization = paste("Bearer", api_key),
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(list(inputs = text), auto_unbox = TRUE) |>
    httr2::req_perform()

  m <- httr2::resp_body_json(resp)

  # HF models often return nested lists
  v <- unlist(m[[1]], use.names = FALSE)
  v
}

#' Compute text embeddings
#'
#' @param text Character string or vector of strings to embed.
#' @param backend Embedding backend to use: "openai", "gemini", or "hf".
#' @param model Model name specific to the backend.
#' @param dimensions Optional dimension specification (backend-specific).
#' @param task_type Optional task type (Gemini only).
#' @param normalize Logical; whether to normalize embeddings to unit length.
#' @param api_key Optional API key override.
#'
#' @return For a single text: numeric vector. For multiple texts: matrix with
#'   one row per text.
#'
#' @examples
#' \dontrun{
#' # Single text embedding
#' emb <- embed_text("A beautiful sunset", backend = "openai")
#'
#' # Multiple texts
#' texts <- c("First caption", "Second caption")
#' emb_matrix <- embed_text(texts, backend = "gemini", model = "text-embedding-004")
#' }
#'
#' @export
embed_text <- function(
  text,
  backend = c("openai", "gemini", "hf"),
  model = NULL,
  dimensions = NULL,
  task_type = NULL,
  normalize = FALSE,
  api_key = NULL
) {
  backend <- match.arg(backend)

  # Default models per backend
  if (is.null(model)) {
    model <- switch(backend,
      openai = "text-embedding-3-large",
      gemini = "text-embedding-004",
      hf = "sentence-transformers/all-MiniLM-L6-v2"
    )
  }

  # Handle multiple texts
  if (length(text) > 1) {
    embeddings <- lapply(text, function(t) {
      embed_text(
        t,
        backend = backend,
        model = model,
        dimensions = dimensions,
        task_type = task_type,
        normalize = normalize,
        api_key = api_key
      )
    })
    return(do.call(rbind, embeddings))
  }

  # Single text embedding
  emb <- switch(backend,
    openai = {
      args <- list(text = text, model = model, dimensions = dimensions)
      if (!is.null(api_key)) args$api_key <- api_key
      do.call(.embed_openai, args)
    },
    gemini = {
      args <- list(
        text = text,
        model = model,
        output_dimensionality = dimensions,
        task_type = task_type
      )
      if (!is.null(api_key)) args$api_key <- api_key
      do.call(.embed_gemini, args)
    },
    hf = {
      args <- list(text = text, model = model)
      if (!is.null(api_key)) args$api_key <- api_key
      do.call(.embed_hf, args)
    }
  )

  # Optional normalization
  if (normalize) {
    norm <- sqrt(sum(emb * emb))
    if (is.finite(norm) && norm > 0) {
      emb <- emb / norm
    }
  }

  emb
}

#' Compute cosine similarity between text embeddings
#'
#' @param emb1 First embedding vector or matrix.
#' @param emb2 Second embedding vector or matrix.
#'
#' @return Cosine similarity value(s).
#' @export
cosine_similarity <- function(emb1, emb2) {
  if (is.vector(emb1) && is.vector(emb2)) {
    # Vector-vector similarity
    sum(emb1 * emb2) / (sqrt(sum(emb1^2)) * sqrt(sum(emb2^2)))
  } else if (is.matrix(emb1) && is.matrix(emb2)) {
    # Matrix-matrix similarity
    coop::tcosine(emb1, emb2)
  } else if (is.matrix(emb1) && is.vector(emb2)) {
    # Matrix-vector similarity
    apply(emb1, 1, function(row) {
      sum(row * emb2) / (sqrt(sum(row^2)) * sqrt(sum(emb2^2)))
    })
  } else if (is.vector(emb1) && is.matrix(emb2)) {
    # Vector-matrix similarity
    apply(emb2, 1, function(row) {
      sum(emb1 * row) / (sqrt(sum(emb1^2)) * sqrt(sum(row^2)))
    })
  } else {
    stop("Invalid input types for cosine similarity")
  }
}
