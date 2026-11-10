# VGG-16 block-end hierarchy with depth-adaptive DCT pooling

One tap at the end of each convolutional block, plus `fc1` and
pre-softmax logits. Convolutional blocks use
[`dct_pool`](https://bbuchsbaum.github.io/imfeatures/reference/dct_pool.md),
and the number of spatial modes decreases from 6 x 6 in block 1 to 2 x 2
in block 5. Early blocks have few channels and fine spatial structure,
so they keep more spatial modes; late blocks have many channels and
coarse maps, so they keep fewer. This keeps the stages similar in width.

## Usage

``` r
vgg16_block_end_spec(target_size = c(224, 224))

vgg19_block_end_spec(target_size = c(224, 224))
```

## Arguments

- target_size:

  Image resize dimensions passed to Keras.

## Value

A `vgg_hierarchy_spec`; see
[`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md).

## Details

With the default 224 x 224 input the VGG-16 stages are:

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| **Stage** | **Layer** | **Pooler** | **Map** | **Channels** | **Columns** |
| `b1` | `block1_conv2` | `dct_pool(6, 6)` | 224 x 224 | 64 | 2304 |
| `b2` | `block2_conv2` | `dct_pool(5, 5)` | 112 x 112 | 128 | 3200 |
| `b3` | `block3_conv3` | `dct_pool(4, 4)` | 56 x 56 | 256 | 4096 |
| `b4` | `block4_conv3` | `dct_pool(3, 3)` | 28 x 28 | 512 | 4608 |
| `b5` | `block5_conv3` | `dct_pool(2, 2)` | 14 x 14 | 512 | 2048 |
| `fc1` | `fc1` | [`identity_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) | – | 4096 | 4096 |
| `logits` | `classifier_logits` | [`identity_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/poolers.md) | – | 1000 | 1000 |

The first DCT column of every channel is its global average, so each
convolutional block contains the usual average-pooled features plus
low-frequency layout. DCT columns are not on a common scale across modes
or stages (see
[`dct_pool`](https://bbuchsbaum.github.io/imfeatures/reference/dct_pool.md));
standardize them before combining, as
[`block_pca`](https://bbuchsbaum.github.io/imfeatures/reference/block_pca.md)
does by default.

`vgg19_block_end_spec()` uses the same poolers on the VGG-19 block-end
layers (`block3_conv4`, `block4_conv4`, `block5_conv4`); map sizes and
column counts are identical.

## See also

[`vgg_hierarchy_spec`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md)
to choose other taps or poolers.

## Examples

``` r
spec <- vgg16_block_end_spec()
spec
#> VGG hierarchy spec (vgg16)
#>   Stages: b1, b2, b3, b4, b5, fc1, logits
#>   Taps:   block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3, fc1, classifier_logits
#>   Pooling:
#>     b1      DCT pool 6 x 6 (zigzag)
#>     b2      DCT pool 5 x 5 (zigzag)
#>     b3      DCT pool 4 x 4 (zigzag)
#>     b4      DCT pool 3 x 3 (zigzag)
#>     b5      DCT pool 2 x 2 (zigzag)
#>     fc1     Identity pool
#>     logits  Identity pool
spec$pooling$b1
#> DCT pool 6 x 6 (zigzag)
```
