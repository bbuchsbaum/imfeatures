# Radial centre/periphery pooling

Averages each channel within concentric rings around the centre of the
activation map. Use it when distance from the image centre matters and
direction does not.

## Usage

``` r
radial_pool(breaks = c(0, 0.25, 0.5, 1))
```

## Arguments

- breaks:

  Strictly increasing numeric breakpoints in \\\[0, 1\]\\ for normalized
  radius from the spatial centre. The default `c(0, 0.25, 0.5, 1)`
  yields three rings.

## Value

An `imfeatures_pooler` that yields `length(breaks) - 1` columns per
channel.

## Details

Radius is measured from the geometric centre of the map and divided by
the distance to the corners, so 0 is the centre and 1 is a corner. On a
square map the midpoint of each edge lies at about 0.71. Ring `k` holds
positions with `breaks[k] <= radius < breaks[k + 1]`; the last ring also
includes its upper break. Positions outside the range of `breaks` are
left out, so `breaks = c(0, 0.5)` pools the central region only.

Rings cover unequal numbers of positions. With the default breaks on a
14 x 14 map, the three rings hold 16, 52, and 128 positions. On small
maps a ring can hold no positions at all; its columns are then 0 and a
warning is raised. The default breaks need a map of about 5 x 5 or
larger: on a 4 x 4 map no position lies within 0.25 of the centre,
because the centre falls between positions.

Columns are ordered by channel, then ring (centre outwards). Names take
the form `stage::channel_001::radial_r1`.

## See also

[`poolers`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md)
for the other pooling options.

## Examples

``` r
# A channel that is active only in the central 2 x 2 positions
a <- array(0, dim = c(1, 8, 8, 1))
a[1, 4:5, 4:5, 1] <- 1
out <- pool_activations(radial_pool(), a)
colnames(out) <- pooler_feature_names(radial_pool(), "b4", 1)
round(out, 3)
#>      b4::channel_001::radial_r1 b4::channel_001::radial_r2
#> [1,]                          1                          0
#>      b4::channel_001::radial_r3
#> [1,]                          0
```
