# Install Python dependencies for imfeatures

Convenience function to install required and optional Python packages
for imfeatures using pip. If the target \`python_cmd\` belongs to a
virtualenv/venv, the \`–user\` flag is automatically disabled as it is
not supported inside virtual environments.

## Usage

``` r
install_python_deps(
  python_cmd = "python3",
  user = TRUE,
  optional = FALSE,
  upgrade = FALSE
)
```

## Arguments

- python_cmd:

  Python command or path (default: "python3")

- user:

  Whether to use –user flag (default: TRUE for HPC). Ignored when
  \`python_cmd\` is a virtualenv/venv.

- optional:

  Whether to install optional packages (default: FALSE)

- upgrade:

  Whether to upgrade existing packages (default: FALSE)

## Examples

``` r
if (FALSE) { # \dontrun{
# Install only required packages
install_python_deps()

# Install all packages including optional
install_python_deps(optional = TRUE)

# Use specific Python
install_python_deps(python_cmd = "/apps/python/3.9/bin/python3")
} # }
```
