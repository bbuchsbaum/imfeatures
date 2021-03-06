% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/im_features.R
\name{im_predict}
\alias{im_predict}
\title{predict the class of an image using Keras model}
\usage{
im_predict(impath, model = NULL, target_size = c(224, 224), topn = 12)
}
\arguments{
\item{impath}{path to image file}

\item{model}{the Keras model}

\item{target_size}{the target image dimensions for approproate for model}
}
\description{
predict the class of an image using Keras model
}
