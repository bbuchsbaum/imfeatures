# Align features using a thingsvision_extractor object

Applies alignment transformations (e.g., gLocal) to features using the
extractor's \`align\` method.

## Usage

``` r
# S3 method for class 'thingsvision_extractor'
tv_align(object, features, module_name, alignment_type = "gLocal", ...)
```

## Arguments

- object:

  An object of class \`thingsvision_extractor\`.

- features:

  An R matrix or array of features (will be converted to Python).

- module_name:

  The module name corresponding to the features being aligned.

- alignment_type:

  Character string. The alignment method (e.g., "gLocal"). Must be a
  character scalar. A warning is issued if the type is not recognized.

- ...:

  Additional arguments (currently ignored).

## Value

Aligned features as an R matrix or array.
