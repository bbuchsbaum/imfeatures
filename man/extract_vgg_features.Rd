% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/vgg_feature_set.R
\name{extract_vgg_features}
\alias{extract_vgg_features}
\title{Extract VGG-16 features by tier}
\usage{
extract_vgg_features(
  impaths,
  tier = c("low", "mid", "high", "semantic"),
  model = NULL,
  target_size = c(224, 224),
  pooling = "avg"
)
}
\arguments{
\item{impaths}{Character vector of image file paths.}

\item{tier}{Character; one of "low", "mid", "high", or "semantic".}

\item{model}{Preloaded Keras VGG-16 model object. If NULL, defaults to \code{keras::application_vgg16(weights = 'imagenet')}.}

\item{target_size}{Numeric vector of length 2 specifying image resize dimensions (width, height).}

\item{pooling}{Character string specifying spatial pooling; passed to the \code{spatial_pooling} argument of \code{im_features}.
Defaults to "avg" (global average pooling). Other options: "none", "max", "resize_3x3", "resize_5x5", "resize_7x7".}
}
\value{
An S3 object of class \code{vgg_feature_set}, a list with components:
\describe{
  \item{features}{Numeric matrix (N_images × total_channels) of pooled features.}
  \item{image_paths}{Character vector of input image paths.}
  \item{tier}{The tier name.}
  \item{pooling}{Pooling type used.}
  \item{layer_indices}{Numeric indices of the selected layers (derived from \code{layer_names}).}
  \item{layer_names}{Character names of VGG-16 layers.}
  \item{model_name}{Character, set to "vgg16".}
  \item{target_size}{Numeric vector of image resize dimensions.}
}
}
\description{
Convenience wrapper around \code{im_features()} to extract VGG-16 features grouped by spatial tiers:
\\itemize{
  \\item{\\code{"low"}: conv1_1, conv1_2, conv2_1, conv2_2}
  \\item{\\code{"mid"}: conv3_1–conv4_3}
  \\item{\\code{"high"}: conv5_1–conv5_3}
  \\item{\\code{"semantic"}: fc1 (fc6) and fc2 (fc7)}
}
Layers are retrieved by name (e.g., \code{"block1_conv1"}) instead of numeric indices.
}
