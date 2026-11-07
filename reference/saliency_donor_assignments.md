# Reproducible donor-map permutations for saliency controls

Produces one-to-one assignments within declared strata, excluding the
same stimulus (including alternate versions). A stratum should encode
aspect ratio, saliency model/version, and preprocessing. No pixel
shuffling is used. These strata do not automatically match
concentration; include a declared concentration stratum if the study
requires it, and inspect achieved matches after resizing.

## Usage

``` r
saliency_donor_assignments(
  ids,
  strata,
  stimulus_ids = ids,
  replicates = 10L,
  seed = 1L
)
```

## Arguments

- ids:

  Unique image/map identifiers.

- strata:

  Stratum identifier for each image.

- stimulus_ids:

  Underlying stimulus identifiers; shared values exclude alternate
  versions of the same stimulus from being each other's donors.

- replicates:

  Number of randomized assignments per stratum.

- seed:

  Local random seed; the caller's random state is restored.

## Value

A data frame with target/donor IDs, stimulus IDs, replicate, stratum,
seed, eligible donor count, and status. A stratum without a complete
valid assignment returns unavailable rows, not an invalid partial
permutation.

## Details

Randomized matching does not sample all valid permutations uniformly.
These assignments support descriptive controls, not calibrated
permutation p-values. Repeated assignments are possible, especially in
small strata.

## Examples

``` r
saliency_donor_assignments(c("a", "b", "c"), rep("same-model-square", 3), seed = 3)
#>    target_id target_stimulus_id donor_id donor_stimulus_id           stratum
#> 1          a                  a        b                 b same-model-square
#> 2          b                  b        c                 c same-model-square
#> 3          c                  c        a                 a same-model-square
#> 4          a                  a        c                 c same-model-square
#> 5          b                  b        a                 a same-model-square
#> 6          c                  c        b                 b same-model-square
#> 7          a                  a        b                 b same-model-square
#> 8          b                  b        c                 c same-model-square
#> 9          c                  c        a                 a same-model-square
#> 10         a                  a        c                 c same-model-square
#> 11         b                  b        a                 a same-model-square
#> 12         c                  c        b                 b same-model-square
#> 13         a                  a        c                 c same-model-square
#> 14         b                  b        a                 a same-model-square
#> 15         c                  c        b                 b same-model-square
#> 16         a                  a        c                 c same-model-square
#> 17         b                  b        a                 a same-model-square
#> 18         c                  c        b                 b same-model-square
#> 19         a                  a        c                 c same-model-square
#> 20         b                  b        a                 a same-model-square
#> 21         c                  c        b                 b same-model-square
#> 22         a                  a        c                 c same-model-square
#> 23         b                  b        a                 a same-model-square
#> 24         c                  c        b                 b same-model-square
#> 25         a                  a        b                 b same-model-square
#> 26         b                  b        c                 c same-model-square
#> 27         c                  c        a                 a same-model-square
#> 28         a                  a        c                 c same-model-square
#> 29         b                  b        a                 a same-model-square
#> 30         c                  c        b                 b same-model-square
#>    replicate seed requested_replicates eligible_donors status
#> 1          1    3                   10               2     ok
#> 2          1    3                   10               2     ok
#> 3          1    3                   10               2     ok
#> 4          2    3                   10               2     ok
#> 5          2    3                   10               2     ok
#> 6          2    3                   10               2     ok
#> 7          3    3                   10               2     ok
#> 8          3    3                   10               2     ok
#> 9          3    3                   10               2     ok
#> 10         4    3                   10               2     ok
#> 11         4    3                   10               2     ok
#> 12         4    3                   10               2     ok
#> 13         5    3                   10               2     ok
#> 14         5    3                   10               2     ok
#> 15         5    3                   10               2     ok
#> 16         6    3                   10               2     ok
#> 17         6    3                   10               2     ok
#> 18         6    3                   10               2     ok
#> 19         7    3                   10               2     ok
#> 20         7    3                   10               2     ok
#> 21         7    3                   10               2     ok
#> 22         8    3                   10               2     ok
#> 23         8    3                   10               2     ok
#> 24         8    3                   10               2     ok
#> 25         9    3                   10               2     ok
#> 26         9    3                   10               2     ok
#> 27         9    3                   10               2     ok
#> 28        10    3                   10               2     ok
#> 29        10    3                   10               2     ok
#> 30        10    3                   10               2     ok
```
