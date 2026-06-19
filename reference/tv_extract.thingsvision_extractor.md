# Extract features using a thingsvision_extractor object

This method uses the configured extractor to extract features from the
provided data loader.

## Usage

``` r
# S3 method for class 'thingsvision_extractor'
tv_extract(
  object,
  dataloader,
  module_name,
  flatten_acts = FALSE,
  output_type = "ndarray",
  output_dir = NULL,
  step_size = NULL,
  ...
)
```

## Arguments

- object:

  An object of class \`thingsvision_extractor\`.

- dataloader:

  A \`reticulate\` reference to a Python \`thingsvision.DataLoader\`
  object, typically created using
  [`tv_create_dataloader`](https://bbuchsbaum.github.io/imfeatures/reference/tv_create_dataloader.md).

- module_name:

  Character string. The layer/module name to extract from.

- flatten_acts:

  Logical. Flatten activations?

- output_type:

  Character string ("ndarray" or "tensor"). The desired Python output
  type before conversion to R. Defaults to "ndarray".

- output_dir:

  Character string (optional). Directory to save features iteratively.

- step_size:

  Integer (optional). Step size for saving if \`output_dir\` is used.
  Must be a finite numeric scalar.

- ...:

  Additional arguments (currently ignored).

## Value

An R matrix or array containing the features, or \`NULL\` invisibly if
\`output_dir\` is specified.
