# Saliency location entropy and effective attended fraction

Normalizes a heat map over all pixels, separately from edge weighting. A
zero-mass map has unavailable features. Negative log densities are
valid.

## Usage

``` r
saliency_location_entropy(saliency, representation, signed_policy = "reject")
```

## Arguments

- saliency:

  Nonempty numeric matrix of aligned saliency values.

- representation:

  Required: `"weights"` or `"log_density"` (natural logs).

- signed_policy:

  For raw weights: `"reject"`, `"rectify"`, or `"shift"`.

## Value

A data frame containing `H_location` and `KL_uniform` in bits,
`effective_fraction = 2^-KL_uniform`, and `status`.

## Details

KL and effective fraction are invariant to equal-mass subdivision of
every pixel. They are not invariant to arbitrary resampling or
smoothing.

## Examples

``` r
saliency_location_entropy(matrix(1, 8, 8), "weights")
#>   H_location KL_uniform effective_fraction status
#> 1          6          0                  1     ok
```
