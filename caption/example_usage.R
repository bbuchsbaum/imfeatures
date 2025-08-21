
# example_usage.R
# --- Setup ---
# Sys.setenv(OPENAI_API_KEY = "...")      # if using OpenAI embeddings or OpenAI for captioning
# Sys.setenv(GEMINI_API_KEY = "...")      # if using Gemini embeddings or Gemini for captioning

# install.packages(c("ellmer","httr2","jsonlite","tibble","purrr","cli"))

source("ellmer_caption_embed.R")

# Example 1: OpenAI for captioning + OpenAI embeddings (3072-dim)
res1 <- caption_and_embed(
  image = "path/to/your/image.jpg",
  caption_provider = "openai",
  caption_model = "gpt-4o-mini",
  template = "dense",
  focus = c("objects","text","layout","colors"),
  min_words = 60,
  max_words = 120,
  temperature = 0.2,
  embedding_backend = "openai",
  embedding_model = "text-embedding-3-large"
)
str(res1)

# Example 2: Gemini for captioning + Gemini embeddings (1536-dim, normalized)
res2 <- caption_and_embed(
  image = "path/to/your/image.jpg",
  caption_provider = "gemini",
  caption_model = "gemini-2.5-flash",
  template = "alt_text",
  min_words = 40,
  max_words = 80,
  embedding_backend = "gemini",
  embedding_model = "gemini-embedding-001",
  embedding_dim = 1536,                    # output_dimensionality
  gemini_task = "RETRIEVAL_DOCUMENT"       # task_type
)
str(res2)

# Example 3: OpenAI captioning + Hugging Face embeddings (MiniLM)
# Sys.setenv(HUGGINGFACE_API_KEY = "...")
res3 <- caption_and_embed(
  image = "path/to/your/image.jpg",
  caption_provider = "openai",
  caption_model = "gpt-4o-mini",
  template = "factual",
  embedding_backend = "hf",
  embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
)
str(res3)

# Batch mode
files <- c("img1.jpg", "img2.jpg")
df <- caption_and_embed_many(files,
  caption_provider = "openai",
  caption_model = "gpt-4o-mini",
  template = "dense",
  embedding_backend = "openai",
  embedding_model = "text-embedding-3-large"
)
print(df)
