# Generic feature extraction

Dispatches to class-specific implementations for extracting features
from configured extractors.

## Usage

``` r
tv_extract(object, ...)
```

## Arguments

- object:

  An object to extract features from.

- ...:

  Additional arguments passed to methods.

## Value

Method-dependent; see implementations such as
[`tv_extract.thingsvision_extractor`](https://bbuchsbaum.github.io/imfeatures/reference/tv_extract.thingsvision_extractor.md).
