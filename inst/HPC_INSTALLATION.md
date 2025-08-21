# Installing imfeatures on HPC Systems

This guide explains how to install and configure the `imfeatures` R package on HPC (High Performance Computing) systems where automatic Python environment creation may fail or be restricted.

## Quick Start

### 1. Install the R package with Python setup disabled

```bash
# Set environment variable to skip Python setup during installation
export IMFEATURES_SKIP_PYTHON=TRUE

# Install the package in R
R -e "install.packages('imfeatures_0.1.0.tar.gz', repos = NULL, type = 'source')"
```

### 2. Configure Python after installation

After the R package is installed, you can configure Python in several ways:

#### Option A: Use HPC module system
```r
# In R, after loading your HPC Python module
library(imfeatures)

# If your HPC provides Python via modules
system("module load python/3.9")  # Example - adjust for your HPC

# Configure imfeatures to use the loaded Python
use_existing_python()
```

#### Option B: Specify custom Python path
```r
library(imfeatures)

# Point to your Python installation
use_existing_python("/path/to/your/python3")
```

#### Option C: Use existing conda environment
```r
library(imfeatures)

# If you have a conda environment already set up
reticulate::use_condaenv("your-env-name")
use_existing_python()
```

## Environment Variables

You can control the package behavior using environment variables:

- `IMFEATURES_SKIP_PYTHON`: Set to `TRUE` to skip automatic Python configuration
- `IMFEATURES_PYTHON_PATH`: Specify a custom Python executable path

Example `.Renviron` file:
```
IMFEATURES_SKIP_PYTHON=TRUE
IMFEATURES_PYTHON_PATH=/usr/local/bin/python3
```

## Manual Python Setup

If automatic setup fails, you can manually install Python dependencies:

### 1. Basic dependencies (required)
```bash
pip install Pillow numpy
```

### 2. Deep learning features (optional)
```bash
# For thingsvision features
pip install thingsvision torch torchvision

# For CLIP support
pip install git+https://github.com/openai/CLIP.git

# For resmem features
pip install resmem
```

### 3. Full installation from requirements file
```bash
# Get the requirements file from the package
R -e "cat(readLines(system.file('requirements.txt', package='imfeatures')), sep='\n')" > requirements.txt

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Problem: "Error creating conda environment"
**Solution**: Set `IMFEATURES_SKIP_PYTHON=TRUE` before installing the package.

### Problem: "Python executable not found"
**Solution**: Load your HPC Python module first, then use `use_existing_python()`.

### Problem: "Module 'thingsvision' not found"
**Solution**: Install Python dependencies manually using pip as shown above.

### Problem: Package won't install due to Python errors
**Solution**: 
1. Set `IMFEATURES_SKIP_PYTHON=TRUE`
2. Install the R package
3. Configure Python after installation using `use_existing_python()`

## Example HPC Workflow

```bash
# 1. Load required modules (example - adjust for your HPC)
module load gcc/9.3.0
module load R/4.3.0
module load python/3.9
module load cuda/11.8  # If using GPU features

# 2. Set environment variable
export IMFEATURES_SKIP_PYTHON=TRUE

# 3. Install R package
R CMD INSTALL imfeatures_0.1.0.tar.gz

# 4. Install Python dependencies
pip install --user Pillow numpy thingsvision torch torchvision

# 5. Use in R
R
```

```r
library(imfeatures)
use_existing_python()  # Will use the module-loaded Python

# Check what's available
use_existing_python(check_modules = TRUE)

# Now you can use the package
# ... your code here ...
```

## Support

For issues specific to HPC installation, please include:
1. Your HPC system details
2. Output of `module list` (if using module system)
3. Output of `which python` and `python --version`
4. The specific error messages you encounter