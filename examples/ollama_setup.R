# Ollama Setup Guide for Local Caption Generation
# =================================================
# Ollama allows you to run LLMs locally without API keys
# Perfect for testing and development

cat("🦙 OLLAMA SETUP GUIDE FOR IMFEATURES\n")
cat("=====================================\n\n")

# Check if Ollama is installed
check_ollama <- function() {
  result <- system2("which", "ollama", stdout = FALSE, stderr = FALSE)
  return(result == 0)
}

if (check_ollama()) {
  cat("✅ Ollama is installed!\n\n")
  
  # Check version
  version <- system2("ollama", "--version", stdout = TRUE, stderr = FALSE)
  cat("Version:", version, "\n\n")
  
  # List installed models
  cat("📦 Installed models:\n")
  models <- system2("ollama", "list", stdout = TRUE, stderr = FALSE)
  cat(models, sep = "\n")
  
} else {
  cat("❌ Ollama is not installed\n\n")
  
  cat("📥 INSTALLATION INSTRUCTIONS:\n")
  cat("-" , strrep("-", 30), "\n\n")
  
  cat("macOS/Linux:\n")
  cat("  curl -fsSL https://ollama.ai/install.sh | sh\n\n")
  
  cat("Windows:\n")
  cat("  Download from: https://ollama.ai/download/windows\n\n")
  
  cat("Docker:\n")
  cat("  docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama\n\n")
}

cat("\n🎯 RECOMMENDED VISION MODELS FOR CAPTION GENERATION:\n")
cat(strrep("-", 50), "\n\n")

vision_models <- list(
  list(
    name = "llava:7b",
    size = "4.5GB",
    description = "Fast, good quality vision-language model",
    pull_cmd = "ollama pull llava:7b"
  ),
  list(
    name = "llava:13b", 
    size = "8.0GB",
    description = "Higher quality, slower than 7b",
    pull_cmd = "ollama pull llava:13b"
  ),
  list(
    name = "bakllava:7b",
    size = "4.5GB", 
    description = "Alternative vision model, sometimes better for games",
    pull_cmd = "ollama pull bakllava"
  ),
  list(
    name = "llava-llama3:8b",
    size = "5.5GB",
    description = "Based on Llama 3, newest architecture",
    pull_cmd = "ollama pull llava-llama3"
  )
)

for (model in vision_models) {
  cat(sprintf("📦 %s (%s)\n", model$name, model$size))
  cat(sprintf("   %s\n", model$description))
  cat(sprintf("   Install: %s\n\n", model$pull_cmd))
}

cat("\n🚀 QUICK START:\n")
cat(strrep("-", 15), "\n\n")

cat("1. Install Ollama (see above)\n")
cat("2. Pull a vision model:\n")
cat("   ollama pull llava:7b\n\n")
cat("3. Test in R:\n\n")

cat('library(imfeatures)\n')
cat('\n')
cat('# Test with local Ollama (no API key needed!)\n')
cat('result <- caption_features(\n')
cat('  "your_image.jpg",\n')
cat('  caption_provider = "ollama",\n')
cat('  caption_model = "llava:7b",\n')
cat('  template = "dense",\n')
cat('  compute_embedding = FALSE  # Ollama does not provide embeddings\n')
cat(')\n')
cat('\n')
cat('print(result$caption)\n\n')

# Test function
test_ollama_caption <- function(image_path = NULL) {
  if (!check_ollama()) {
    cat("❌ Please install Ollama first\n")
    return(NULL)
  }
  
  library(imfeatures)
  
  # Use Space Invaders if no image provided
  if (is.null(image_path)) {
    image_path <- "Space_Invaders.jpg"
    if (!file.exists(image_path)) {
      image_path <- "../Space_Invaders.jpg"
    }
  }
  
  if (!file.exists(image_path)) {
    cat("❌ Image not found:", image_path, "\n")
    return(NULL)
  }
  
  cat("\n🧪 TESTING OLLAMA CAPTION GENERATION\n")
  cat(strrep("-", 35), "\n")
  cat("Image:", image_path, "\n\n")
  
  # Check available models
  models <- system2("ollama", "list", stdout = TRUE, stderr = FALSE)
  vision_models_available <- c("llava", "bakllava", "llava-llama3")
  
  model_to_use <- NULL
  for (vm in vision_models_available) {
    if (any(grepl(vm, models))) {
      model_lines <- models[grepl(vm, models)]
      # Extract model name (first column)
      model_to_use <- strsplit(model_lines[1], "\\s+")[[1]][1]
      break
    }
  }
  
  if (is.null(model_to_use)) {
    cat("❌ No vision models found. Install one with:\n")
    cat("   ollama pull llava:7b\n")
    return(NULL)
  }
  
  cat("Using model:", model_to_use, "\n\n")
  
  # Generate caption
  result <- tryCatch({
    caption_features(
      image_path,
      caption_provider = "ollama",
      caption_model = model_to_use,
      template = "dense",
      focus = c("objects", "colors", "layout"),
      compute_embedding = FALSE
    )
  }, error = function(e) {
    cat("❌ Error:", e$message, "\n")
    NULL
  })
  
  if (!is.null(result)) {
    cat("✅ Caption generated successfully!\n\n")
    cat("Caption:\n")
    cat(strwrap(result$caption, width = 70, prefix = "  "), sep = "\n")
    cat("\n")
  }
  
  return(result)
}

# Performance tips
cat("\n⚡ PERFORMANCE TIPS:\n")
cat(strrep("-", 18), "\n\n")

cat("1. GPU Acceleration (NVIDIA):\n")
cat("   - Ollama automatically uses GPU if available\n")
cat("   - Check with: nvidia-smi\n\n")

cat("2. Memory Management:\n")
cat("   - 7B models need ~8GB RAM\n")
cat("   - 13B models need ~16GB RAM\n")
cat("   - Close other applications if needed\n\n")

cat("3. Model Loading:\n")
cat("   - First request is slow (model loading)\n")
cat("   - Subsequent requests are much faster\n")
cat("   - Models stay loaded for 5 minutes by default\n\n")

cat("4. Batch Processing:\n")
cat("   - Process images sequentially with Ollama\n")
cat("   - Model stays loaded between calls\n\n")

# Troubleshooting
cat("\n🔧 TROUBLESHOOTING:\n")
cat(strrep("-", 18), "\n\n")

cat("1. 'Ollama not running' error:\n")
cat("   Solution: Start Ollama service\n")
cat("   - macOS/Linux: ollama serve\n")
cat("   - Or just run: ollama list (auto-starts)\n\n")

cat("2. 'Model not found' error:\n")
cat("   Solution: Pull the model first\n")
cat("   - ollama pull llava:7b\n\n")

cat("3. Slow performance:\n")
cat("   - Try smaller model (7b instead of 13b)\n")
cat("   - Check available RAM\n")
cat("   - Ensure no other heavy processes running\n\n")

cat("4. Poor caption quality:\n")
cat("   - Try different model (bakllava vs llava)\n")
cat("   - Adjust prompt template\n")
cat("   - Use more specific focus areas\n\n")

# Run test if requested
cat("📝 To test Ollama with Space Invaders image, run:\n")
cat("   test_ollama_caption()\n\n")

cat("✅ Setup guide complete!\n")