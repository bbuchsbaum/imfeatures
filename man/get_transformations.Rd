% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tv_extractor.R
\name{get_transformations}
\alias{get_transformations}
\title{Get Preprocessing Transformations}
\usage{
get_transformations(object, ...)
}
\arguments{
\item{object}{An object of class `thingsvision_extractor`.}

\item{...}{Arguments passed to the underlying `get_transformations` Python method
(e.g., `resize_dim`, `crop_dim`). Usually not needed as defaults are inferred.}
}
\value{
A `reticulate` reference to the Python preprocessing callable.
}
\description{
Retrieves the image preprocessing function (as a Python function object)
associated with the extractor. This is typically used internally by
dataset creation functions.
}
