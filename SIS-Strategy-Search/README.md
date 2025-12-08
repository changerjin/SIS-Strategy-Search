

# Heuristic Optimal Strategy Finder for Lattice Problems 🧮

A framework for finding heuristically optimal strategies for lattice problems, consisting of two core components: `SIS-estimator` for strategy search and `G6K-GPU-Tensor-python3` for practical experiments solving Lattice Challenges.


## ✨ Features

### 1. SIS-estimator (Strategy Search Component)
- `Strategy_search.py`: Uses the **CN-11 deterministic simulator** for precise bound evaluation of Lattice Challenge strategies.
- `Strategy_search_probab.py`: Uses the **BSW-18 probabilistic simulator** for analyzing strategy success probabilities.
- `NIST_estimator.py`: Searches for solving strategies for the Falcon post-quantum signature scheme.
- `NIST_estimator_inf_norm.py`: Searches for solving strategies for the Dilithium post-quantum signature scheme.

### 2. G6K-GPU-Tensor-python3 (Experimental Component)
- **Consistent Installation Requirements**: Follows the same installation prerequisites as [G6K-GPU-Tensor](https://github.com/WvanWoerden/G6K-GPU-Tensor).
-  Run `run_sis.sh` directly to replicate experimental results for Lattice Challenges.

## 📋 Prerequisites

### Common Dependencies
- Python 3.8+
- NumPy, SciPy (for numerical computations in strategy search)
- Git (for repository management and dependency cloning)

### SIS-estimator Specific
- Standard Python scientific libraries (no additional specialized dependencies)

### G6K-GPU-Tensor-python3 Specific
- **Same as G6K-GPU-Tensor**: Refer to the [official installation guide](https://github.com/WvanWoerden/G6K-GPU-Tensor) for full requirements, including:
  - NVIDIA Turing architecture GPU (tested and validated)
  - CUDA Toolkit (compatible version with GPU and project dependencies)
  - Compilation tools: `gcc`, `g++`, `make`, `autogen`, `libtool`
  - Core libraries: FPLLL (lattice reduction), FPyLLL (Python binding for FPLLL)
  - Python 3 dependencies: `Cython`, `cysignals`, `numpy`, `scipy`, `pytest`, etc. (listed in `requirements.txt`)
  - `parallel-hashmap` (automatically cloned via installation scripts)

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/changerjin/SIS-Strategy-Search.git
cd SIS-Strategy-Search
