# slurm-gpus

A simple tool to list availability of GPUs on your SLURM cluster.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/rexhaif/slurm-gpus.git
```

Or if you have downloaded the source code:

```bash
pip install .
```

## Usage

After installation, you can run the tool directly from the command line:

```bash
slurm-gpus
```

This will display a summary of available GPUs on your SLURM cluster, including:
- GPU types and models
- Memory per GPU
- Total, allocated, and available GPUs
- Detailed allocation information per node

## Requirements

- Python 3.6 or higher
- SLURM workload manager
- `scontrol` command available in PATH

## License

This project is licensed under the AGPL License - see the LICENSE file for details.
