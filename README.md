
# GlyCompute

GlyCompute is a Python package designed to generate the results found in the paper: [GlyCompute: towards the automated analysis of protein N-linked glycosylation kinetics via an open-source computational framework].

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Modules](#modules)
- [Examples](#examples)
- [License](#license)

## Features

- **Feature 1:** Subgraph generation from glycosylation reaction networks (GRN)
- **Feature 2:** Automated assembly of kinetic model of protein N-linked glycosylation
- **Feature 3:** Parameter estimation approach based on Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC)

## Installation

### Downloading the Project

First, download the project from GitHub. You can do this by cloning the repository:

```bash
git clone https://github.com/kf120/GlyCompute_paper.git
```

Alternatively, you can download the project as a ZIP file and extract it.

### Setting Up the Environment with Anaconda

1. Enter the Repository

```bash
cd GlyCompute_paper
```

2. Create and Activate the Conda Environment
```bash
conda env create -f environment.yml
conda activate glycompute_env
```

This command will install the package and all its dependencies.

### Verifying the Installation

To verify the installation, open a Python session and try importing the package:

```python
import glycompute
print(glycompute)
```

If you can import the package without any errors, the installation was successful.

## Modules

### `glycompute.abc`
Contains functions for ABC-SMC.

### `glycompute.graph`
Contains functions for graph operations.

### `glycompute.model`
Contains functions for model assembly.

### `glycompute.pathway`
Contains functions for automated glycosylation pathway extraction.

### `glycompute.simulation`
Contains functions for simulation activities.

### `glycompute.strategy`
Contains functions for the design of a stage-wise parameter estimation strategy based on graph topology and domain knowledge.

### `glycompute.utils`
Contains utility functions for ancillary tasks across other modules. 

## Examples

You can find example scripts in the `case_study` directory. These scripts demonstrate how to use the different functions provided by the package and can be used to reproduce the results in the paper.

### Running Examples

After installing the package, you can run the example scripts directly from the command line:

#### Example SPF

Navigate to the `case_study` directory and run:

```bash
python example_SPF.py
```

#### Example MFR

Navigate to the `case_study` directory and run:

```bash
python example_MFR.py
```

#### Example GLW

Navigate to the `case_study` directory and run:

```bash
python example_GLW.py
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
