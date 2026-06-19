# Get Preprocessing Transformations

Retrieves the image preprocessing function (as a Python function object)
associated with the extractor. This is typically used internally by
dataset creation functions.

## Usage

``` r
get_transformations(object, ...)
```

## Arguments

- object:

  An object of class \`thingsvision_extractor\`.

- ...:

  Arguments passed to the underlying \`get_transformations\` Python
  method (e.g., \`resize_dim\`, \`crop_dim\`). Usually not needed as
  defaults are inferred.

## Value

A \`reticulate\` reference to the Python preprocessing callable.
