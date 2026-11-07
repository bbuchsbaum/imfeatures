# Multiscale location entropy of a saliency map

Sums probability mass into clipped spatial blocks at several scales and
shifted origins. This describes one location distribution, not joint
entropy.

## Usage

``` r
saliency_multiscale_entropy(
  saliency,
  representation,
  signed_policy = "reject",
  scales = c(1/32, 1/16, 1/8, 1/4, 1/2),
  units = "relative",
  pixels_per_degree = NULL,
  origins = rbind(c(0, 0), c(0.5, 0), c(0, 0.5), c(0.5, 0.5))
)
```

## Arguments

- saliency:

  Nonempty numeric matrix of aligned saliency values.

- representation:

  Required: `"weights"` or `"log_density"` (natural logs).

- signed_policy:

  For raw weights: `"reject"`, `"rectify"`, or `"shift"`.

- scales:

  Positive block sizes, in the units selected by `units`.

- units:

  `"relative"` (fraction of the shorter image dimension), `"degrees"`,
  or explicit `"pixels"`.

- pixels_per_degree:

  Positive conversion factor required for degrees.

- origins:

  Two-column matrix of offsets as fractions of one block, in matrix-axis
  order. Values must be in `[0,1)`.

## Value

A list with origin-specific `curve`, origin-averaged `summary`, and
`metadata`. Boundary blocks use their actual areas for KL from uniform.

## Examples

``` r
saliency_multiscale_entropy(matrix(1, 8, 8), "weights", scales = c(.25, .5))
#> $curve
#>   scale_index scale block_pixels origin offset_1 offset_2 blocks H_location
#> 1           1  0.25            2      1        0        0     16       4.00
#> 2           1  0.25            2      2        1        0     20       4.25
#> 3           1  0.25            2      3        0        1     20       4.25
#> 4           1  0.25            2      4        1        1     25       4.50
#> 5           2  0.50            4      1        0        0      4       2.00
#> 6           2  0.50            4      2        2        0      6       2.50
#> 7           2  0.50            4      3        0        2      6       2.50
#> 8           2  0.50            4      4        2        2      9       3.00
#>   KL_uniform effective_fraction status
#> 1          0                  1     ok
#> 2          0                  1     ok
#> 3          0                  1     ok
#> 4          0                  1     ok
#> 5          0                  1     ok
#> 6          0                  1     ok
#> 7          0                  1     ok
#> 8          0                  1     ok
#> 
#> $summary
#>   scale_index scale block_pixels H_location KL_uniform effective_fraction
#> 1           1  0.25            2       4.25          0                  1
#> 2           2  0.50            4       2.50          0                  1
#>   status
#> 1     ok
#> 2     ok
#> 
#> $metadata
#> $metadata$units
#> [1] "relative"
#> 
#> $metadata$pixels_per_degree
#> NULL
#> 
#> $metadata$origins
#>      [,1] [,2]
#> [1,]  0.0  0.0
#> [2,]  0.5  0.0
#> [3,]  0.0  0.5
#> [4,]  0.5  0.5
#> 
#> $metadata$dimensions
#> [1] 8 8
#> 
#> $metadata$boundary
#> [1] "clipped"
#> 
#> 
```
