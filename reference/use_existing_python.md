# Use existing Python installation for imfeatures

Configures imfeatures to use an existing Python installation without
attempting to create or modify environments. This is particularly useful
on HPC systems where Python environments are pre-configured.

## Usage

``` r
use_existing_python(python_path = NULL, check_modules = TRUE, force = FALSE)
```

## Arguments

- python_path:

  Optional path to Python executable. If NULL, uses the current
  reticulate Python configuration.

- check_modules:

  Whether to check for required Python modules and provide informative
  messages about missing dependencies.

- force:

  If TRUE, prioritize system Python over any existing virtualenvs.
  Useful on HPC systems where module-loaded Python should be used.

## Value

Invisible path to the configured Python binary.

## Details

This function is designed for HPC and other restricted environments
where: - Conda/virtualenv creation may not be allowed - Python modules
are installed via module load systems - Custom Python paths need to be
specified

## Examples

``` r
if (FALSE) { # \dontrun{
# On HPC after loading Python module
# module load python/3.9
use_existing_python()

# With specific Python path
use_existing_python("/usr/local/bin/python3")

# Skip module checking
use_existing_python(check_modules = FALSE)
} # }
```
