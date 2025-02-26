#!/usr/bin/env python3

import sys
import warnings

# Display deprecation warning
warnings.warn(
    "This script is deprecated. Please install the package with 'pip install .' "
    "and use the 'slurm-gpus' command instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the package
try:
    from slurm_gpus.core import main
except ImportError:
    print("Error: Could not import from slurm_gpus package.")
    print("Please install the package with 'pip install .' first.")
    sys.exit(1)

if __name__ == "__main__":
    main()