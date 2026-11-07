# Convert a 4-D activation tensor to NHWC layout

Convert a 4-D activation tensor to NHWC layout

## Usage

``` r
as_nhwc(a, layout = c("nhwc", "nchw"))
```

## Arguments

- a:

  Numeric array. 4-D tensors are permuted; other ranks are returned
  unchanged.

- layout:

  `"nhwc"` (Keras, default) or `"nchw"` (PyTorch).

## Value

The same array in `N × H × W × C` order when `a` is 4-D.
