% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/clip_features.R
\name{clip_features}
\alias{clip_features}
\title{Extract CLIP embeddings (final or intermediate layers)}
\usage{
clip_features(
  impath,
  layers = "final",
  model_name = "ViT-B-32",
  num_transformer_blocks = 12,
  device = c("cpu", "cuda")
)
}
\arguments{
\item{impath}{Path to image file (jpg/png).}

\item{layers}{Character or integer vector. "final" (default) returns the
CLIP image embedding. For intermediate transformer layers,
provide full layer names (e.g., "visual.transformer.resblocks.6")
or integer indices (0-based) representing transformer blocks.
Other layer names (e.g., "visual.class_embedding", "visual.ln_post")
can also be specified if known.}

\item{model_name}{CLIP model string (e.g. "ViT-B-32", "RN50", "ViT-L-14").}

\item{num_transformer_blocks}{Integer, number of transformer blocks in the
vision model. Defaults to 12 (e.g., for ViT-B). For ViT-L, this would be 24.
Only relevant if integer indices are used for `layers`.}

\item{device}{"cpu" or "cuda".}
}
\value{
A named list of numeric arrays (one per requested layer).
}
\description{
Extract CLIP embeddings (final or intermediate layers)
}
