
<!-- README.md is generated from README.Rmd. Please edit that file -->

# imfeatures

<!-- badges: start -->
[![R-CMD-check](https://github.com/bbuchsbaum/imfeatures/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bbuchsbaum/imfeatures/actions/workflows/R-CMD-check.yaml)
[![Codecov coverage](https://codecov.io/gh/bbuchsbaum/imfeatures/branch/main/graph/badge.svg)](https://app.codecov.io/gh/bbuchsbaum/imfeatures)
<!-- badges: end -->

The goal of imfeatures is to …

## Get started

Install the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("bbuchsbaum/imfeatures")
```

## Cookbook

This is a basic example which shows you how to solve a common problem:

``` r
library(imfeatures)
## basic example code
im1 = "testdata/1_A_1.jpeg"
im2 = "testdata/1_B_1.jpeg"

im_feature_sim(c(im1,im2), layers=c(1,2,3))
#> [========================================================================================] 100%
#> $layer_1
#>            1_A_1.jpeg 1_B_1.jpeg
#> 1_A_1.jpeg  0.0000000  0.3326229
#> 1_B_1.jpeg  0.3326229  0.0000000
#> 
#> $layer_2
#>            1_A_1.jpeg 1_B_1.jpeg
#> 1_A_1.jpeg    0.00000    0.37037
#> 1_B_1.jpeg    0.37037    0.00000
#> 
#> $layer_3
#>            1_A_1.jpeg 1_B_1.jpeg
#> 1_A_1.jpeg  0.0000000  0.4695186
#> 1_B_1.jpeg  0.4695186  0.0000000
```
