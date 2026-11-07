# Check batch invariance and row-permutation equivariance

Check batch invariance and row-permutation equivariance

## Usage

``` r
check_transform_invariance(fit, blocks, idx = NULL, tolerance = 1e-08)
```

## Arguments

- fit:

  A `hierarchy_transform`.

- blocks:

  Blocks to transform.

- idx:

  Optional row index used for the batch-invariance check. Defaults to a
  deterministic subset.

- tolerance:

  Absolute tolerance for `all.equal`.

## Value

A list with logical flags and maximum absolute differences.
