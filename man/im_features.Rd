% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/im_features.R
\name{im_features}
\alias{im_features}
\title{extract features from intermediate layers}
\usage{
im_features(impath, layers, model = NULL, target_size = c(224, 224))
}
\arguments{
\item{impath}{path to image file}

\item{layers}{the layer indices}

\item{model}{the Keras model}

\item{target_size}{the target image dimensions for approproate for model}
}
\description{
extract features from intermediate layers
}
