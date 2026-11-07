# VGG-16 block-end hierarchy with depth-adaptive DCT pooling

One tap at the end of each convolutional block, plus `fc1` and
pre-softmax logits. Spatial modes decrease from 6 x 6 in block 1 to 2 x
2 in block 5.

## Usage

``` r
vgg16_block_end_spec(target_size = c(224, 224))
```

## Arguments

- target_size:

  Image resize dimensions passed to Keras.
