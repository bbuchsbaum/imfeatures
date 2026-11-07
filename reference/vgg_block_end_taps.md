# Block-end tap names for a VGG architecture

Block-end tap names for a VGG architecture

## Usage

``` r
vgg_block_end_taps(architecture = c("vgg16", "vgg19"))
```

## Arguments

- architecture:

  `"vgg16"` or `"vgg19"`.

## Value

A named character vector of Keras layer names, including the virtual tap
`classifier_logits`.
