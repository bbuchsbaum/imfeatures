# Stimulus-only diagnostics for a fitted hierarchy transform

Stimulus-only diagnostics for a fitted hierarchy transform

## Usage

``` r
hierarchy_diagnostics(fit, blocks, heldout = NULL)
```

## Arguments

- fit:

  A `hierarchy_transform`.

- blocks:

  Blocks used to compute in-sample or supplied-sample diagnostics.
  Typically the training blocks.

- heldout:

  Optional held-out blocks for residual CKA and residual energy.

## Value

An object of class `hierarchy_diagnostics`.
