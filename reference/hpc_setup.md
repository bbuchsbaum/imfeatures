# HPC Setup for imfeatures

A single, recommended path to configure Python on HPC systems where
automatic Conda setup may fail or be undesirable.

## Quickstart

1\. Disable automatic Python setup on package load:

In R: \`Sys.setenv(IMFEATURES_SKIP_PYTHON = "TRUE")\`

To make permanent, add \`IMFEATURES_SKIP_PYTHON=TRUE\` to
\`~/.Renviron\`.

2\. Create or choose a Python environment (module + venv is typical):

\- Load Python module (if applicable): \`module load python/3.9\` -
Create venv: \`python -m venv \$WORK/venvs/imfeatures\` - Activate:
\`source \$WORK/venvs/imfeatures/bin/activate\` - Upgrade tooling: \`pip
install –upgrade pip wheel setuptools\` - Install minimal required
packages: \`pip install Pillow numpy\` - Optional full features: \`pip
install thingsvision resmem open-clip-torch\` - Install PyTorch via your
cluster module, or with pip: - CPU: \`pip install torch torchvision
–index-url https://download.pytorch.org/whl/cpu\` - CUDA example: \`pip
install torch torchvision –index-url
https://download.pytorch.org/whl/cu118\`

3\. Point imfeatures at that Python in R:

\- Session-only: \`library(imfeatures);
use_existing_python("\$WORK/venvs/imfeatures/bin/python")\` -
Persistent: add to \`~/.Renviron\`:

“\` RETICULATE_PYTHON=\$WORK/venvs/imfeatures/bin/python
IMFEATURES_SKIP_PYTHON=TRUE “\`

4\. Verify:

\`reticulate::py_config()\` should report the selected Python.

## Environment variables

\- \`IMFEATURES_SKIP_PYTHON\`: If \`TRUE\`, skip auto configuration on
package load. - \`RETICULATE_PYTHON\`: Absolute path to the Python
binary reticulate should use. - \`IMFEATURES_METHOD\`: One of \`auto\`,
\`conda\`, \`virtualenv\`, \`existing\` to control
\`imfeatures_config()\` behavior. On HPC, \`virtualenv\` or \`existing\`
is recommended.

## Troubleshooting

\- If you see a Conda error like "bad interpreter" while loading the
package, you likely have a broken R-miniconda on the shared filesystem.
Use the steps above or set \`IMFEATURES_SKIP_PYTHON=TRUE\` and call
\`use_existing_python()\`. - To avoid Conda entirely during
configuration, set \`IMFEATURES_METHOD=virtualenv\`. -
\`RETICULATE_PYTHON\` is respected on load; set it to point at your
Python to bypass any auto-detection.

## See also

\[use_existing_python()\], \[configure_hpc_python()\],
\[install_python_deps()\], and the vignette: \`vignette("hpc-setup")\`.

## Examples

``` r
# See the vignette for a complete walkthrough:
# vignette("hpc-setup")
```
