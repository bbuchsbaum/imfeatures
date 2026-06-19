# Configure the Python environment for imfeatures

Sets up or activates the Python environment required by the package. The
environment is detected once per R session and cached. Users can force a
reinstall with \`reset = TRUE\`.

## Usage

``` r
imfeatures_config(
  envname = "r-imfeatures",
  method = c("auto", "conda", "virtualenv", "existing"),
  reset = FALSE,
  create = TRUE
)

install_imfeatures_python(...)
```

## Arguments

- envname:

  Name of the environment to create/use. Defaults to "r-imfeatures".

- method:

  Installation method. Options: "auto", "conda", "virtualenv", or
  "existing" (to use current Python without modification).

- reset:

  Force reinstallation of the environment.

- create:

  Whether to create the environment if it doesn't exist. Set to FALSE on
  HPC systems where you want to use a pre-configured environment.

- ...:

  Arguments passed to imfeatures_config

## Value

Invisible path to the configured Python binary.

## Functions

- `install_imfeatures_python()`: Deprecated alias for imfeatures_config

## HPC recommendation

On HPC systems, prefer using an existing Python (module + virtualenv)
and avoid auto environment creation. Set \`IMFEATURES_SKIP_PYTHON=TRUE\`
to skip auto-configuration on package load and call
\[use_existing_python()\] to point to your Python. You can also set
\`RETICULATE_PYTHON\` in \`~/.Renviron\` so it is used automatically. If
you do use \`imfeatures_config()\` on HPC, set
\`IMFEATURES_METHOD=virtualenv\` to avoid Conda.
