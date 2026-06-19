# Configure Python for HPC systems

Helper function specifically designed for HPC systems where Python is
loaded via module systems or custom installations.

## Usage

``` r
configure_hpc_python(
  module_cmd = NULL,
  python_cmd = "python3",
  install_deps = FALSE,
  pip_user = TRUE
)
```

## Arguments

- module_cmd:

  Optional module load command to run (e.g., "module load python/3.9")

- python_cmd:

  Python command to use (default: "python3")

- install_deps:

  Whether to attempt installing missing dependencies

- pip_user:

  Whether to use –user flag for pip installs. Automatically disabled
  when targeting a virtualenv/venv, where –user is not allowed.

## Value

Invisible path to configured Python

## Details

This function is tailored for HPC environments and will: 1. Run the
module load command if provided 2. Find and configure the appropriate
Python 3. Check for required modules 4. Optionally install missing
dependencies with pip –user

## Examples

``` r
if (FALSE) { # \dontrun{
# Load Python module and configure
configure_hpc_python("module load python/3.9")

# Use specific Python without module load
configure_hpc_python(python_cmd = "/apps/python/3.9/bin/python3")

# Auto-install missing dependencies
configure_hpc_python("module load python/3.9", install_deps = TRUE)
} # }
```
