# Mixed Resolution Contact Data Modelling

## Setup

### Prerequisites

- Python 3.9+
- R 4.x (for the R-based experiments, e.g. comparisons against the `ungroup`/`pclm` reference implementation)

### Clone the repository

```bash
git clone https://github.com/ShozenD/mixres.git
cd mixres
```

### Python

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

Each time you start a new terminal session, activate the virtual environment before running any code:

```bash
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### R

R dependencies are managed with [`renv`](https://rstudio.github.io/renv/), which keeps a
project-local package library pinned to the versions recorded in `renv.lock`.

1. From the repository root, restore the locked package library (this installs `renv`
   itself on first run if needed, then reproduces the recorded packages, e.g. `ungroup`
   for the PCLM reference implementation):
   ```bash
   Rscript -e 'renv::restore()'
   ```

`renv` activates itself automatically for any R session started from the project root
(via `.Rprofile`), so no further activation step is needed.

