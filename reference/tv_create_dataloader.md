# Create a thingsvision DataLoader

Create a thingsvision DataLoader

## Usage

``` r
tv_create_dataloader(dataset, batch_size, extractor, ...)
```

## Arguments

- dataset:

  A thingsvision ImageDataset object

- batch_size:

  Integer batch size

- extractor:

  An R object of class \`thingsvision_extractor\`

- ...:

  Additional arguments for DataLoader (e.g., shuffle, num_workers)

## Value

A reticulate Python object reference to the DataLoader
