# Extract CLIP text features from text strings

Encodes text strings using a CLIP model and returns embeddings from
specified layers of the text transformer.

## Usage

``` r
clip_text_features(
  texts,
  layers = "final",
  model_name = "ViT-B-32",
  num_text_transformer_blocks = 12,
  text_module_prefix = "transformer",
  device = c("cpu", "cuda")
)
```

## Arguments

- texts:

  Character vector of text strings to encode.

- layers:

  Character or integer vector. "final" (default) returns the final CLIP
  text embeddings. For intermediate transformer layers, provide full
  layer names (e.g., "transformer.resblocks.6") or integer indices
  (0-based) representing transformer blocks. Other layer names (e.g.,
  "token_embedding", "ln_final") can also be specified if known.

- model_name:

  CLIP model string (e.g. "ViT-B-32", "RN50", "ViT-L-14").

- num_text_transformer_blocks:

  Integer, number of transformer blocks in the text model. Defaults to
  12 (e.g., for ViT-B/32 text transformer). Only relevant if integer
  indices are used for \`layers\`.

- text_module_prefix:

  Character string, the base path to the text model's transformer
  blocks. Defaults to "transformer".

- device:

  "cpu" or "cuda".

## Value

A named list of numeric arrays (one per requested layer). \`out\$final\`
will be a matrix (N_texts x EmbeddingDim). Intermediate layers will be
3D arrays (N_texts x SeqLen x HiddenDim).
