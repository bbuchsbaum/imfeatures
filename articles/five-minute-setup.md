# Five-minute set-up

This short guide shows how to configure **imfeatures**.

## GPU installation

``` r

imfeatures_config()
```

If CUDA is available, the output will highlight the GPU being used.

## CPU only

``` r

imfeatures_config(reset = TRUE)
```

The function recreates the environment and reports that CPU is used.

## Offline

Place the wheels listed in `requirements.txt` in a directory and pass
the path via `RETICULATE_PYTHON` before calling
[`imfeatures_config()`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md).
