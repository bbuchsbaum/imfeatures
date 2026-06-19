# Create a thingsvision ImageDataset

Create a thingsvision ImageDataset

## Usage

``` r
tv_create_dataset(root, out_path, extractor, transforms = NULL, ...)
```

## Arguments

- root:

  Path to the common root directory for images. File names passed to the
  Python dataset should be relative to this directory.

- out_path:

  Path for storing file order list

- extractor:

  An R object of class \`thingsvision_extractor\`. \# MODIFIED

- transforms:

  Optional Python transforms object (usually get from extractor)

- ...:

  Additional arguments for ImageDataset (e.g., class_names, file_names)

## Value

A reticulate Python object reference to the ImageDataset
