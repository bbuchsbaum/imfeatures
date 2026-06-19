# Print method for vgg_feature_set objects

Displays a summary of a VGG-16 feature set, including the tier, number
of images, feature dimensionality and pooling type.

## Usage

``` r
# S3 method for class 'vgg_feature_set'
print(x, ...)
```

## Arguments

- x:

  A `vgg_feature_set` object.

- ...:

  Additional arguments (ignored).

## Examples

``` r
if (FALSE) { # \dontrun{
img <- system.file("extdata", "cat.jpg", package = "imfeatures")
fs <- extract_vgg_features(img)
print(fs)
} # }
```
