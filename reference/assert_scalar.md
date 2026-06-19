# Internal assertion helpers using checkmate

These small wrappers standardize argument checks across the package and
yield clear error messages.

## Usage

``` r
assert_scalar(
  x,
  type = c("character", "numeric", "integer", "logical"),
  na.ok = FALSE,
  .var.name = deparse(substitute(x))
)
```
