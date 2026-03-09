# Quantized Linformer for Jet Tagging

This repository implements a quantised Linformer neural network for high-energy physics jet classification using HGQ (Heterogeneous Graph Quantisation) and hls4ml.

## Overview

The project trains a quantised attention-based model (`QLinformer`) to classify jets into five categories (gluon, quark, top, W boson, Z boson) with hardware efficiency constraints. The model targets ~200k EBOps (Binary Operations) for deployment on FPGA/ASIC backends.

## Key Features

- **Quantised Linformer Attention**: O(N) linear complexity attention mechanism with learned projection dimension
- **HGQ Integration**: Hardware-aware quantisation with automatic EBOps tracking and beta-PID optimisation
- **HLS4ML Conversion**: Automatic synthesis to C++ for Intel oneAPI/HLS backends
- **Milestone Tracking**: Saves model checkpoints when crossing predefined EBOps thresholds

## Requirements

```bash
pip install -r requirements.txt
```

## Setup

1. **Data Preparation**
   - Training data can be found 
   - Download LHCjet 150p dataset into `archive/hls4ml_LHCjet_150p_train/`
   - Expected structure:
     ```
     archive/hls4ml_LHCjet_150p_train/
     ├── train/
     │   └── jetImage_*.h5
     └── val/
         └── jetImage_*.h5
     ```

2. **Configure JAX Backend**
   - The notebooks set `KERAS_BACKEND='jax'` for HGQ compatibility
   - To use GPU: `export JAX_PLATFORMS='cuda'`

## Usage

### Training

Run the main notebook:
```bash
jupyter notebook QLinformer2.ipynb
```

The notebook covers:
1. **Config**: Model hyperparameters and data paths
2. **Data Processing**: Loading and normalising jet constituent data
3. **Architecture**: QLinformer model with quantisation scopes
4. **Training**: HGQ training with EBOps-aware callbacks
5. **Synthesis**: Converting trained models to HLS4ML C++ projects

### Outputs

- **Models**: Saved in `saved_models/` as `.keras` files
- **HLS Projects**: Generated in `hls_projects/linformer_ebops_*/` for synthesis
- **Evaluation**: ROC curves and sparsity analysis in the notebook

## Model Architecture

- **Input**: 64 jet constituents × 3 features (pT, η, φ) after 2 GeV pT cut
- **Embedding**: 2×QDense layers to FF_DIM=16 dimensions
- **Attention**: QLinformerAttention with proj_k=2 linear projection
- **Residual Blocks**: 2×(FFN + residual connection)
- **Output**: 5-way classification (per-jet flavour tags)

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `FF_DIM` | 16 | Embedding dimension |
| `NUM_PARTICLES` | 64 | Jet constituents |
| `PROJ_K` | 2 | Linformer projection dimension |
| `TARGET_EBOPS` | 200,000 | Hardware constraint |
| `BATCH_SIZE` | 128 | Training batch size |

## Hardware Synthesis

Generated HLS projects can be synthesised using Intel Quartus or HLS tools:

```bash
./build_report.sh linformer_ebops_200000
```

### Cross-Platform Synthesis (Apple Silicon - M Series)

Running Intel HLS tools on Apple Silicon requires a dual-container workflow due to architecture conflicts. Rosetta 2 cannot translate AVX instructions used by TensorFlow's x86 build.

**See [Synthesis_On_ARM/](Synthesis_On_ARM/) for detailed setup:**
- Dockerfile for Intel oneAPI container (x86_64 emulation via Rosetta 2)
- Automation script to link ARM64 dev environment with x86_64 build container
- Daily workflow for model training → C++ generation → synthesis 
- this allows creation of report files for estimates latency and resource usage

This approach allows you to develop on native ARM64 Python/TensorFlow while delegating compilation to an emulated Intel build server.

## References

- [HGQ Documentation](https://github.com/calad0i/HGQ2)
- [hls4ml](http://hls4ml.ics.uci.edu/)
- LHCjet dataset from high-energy physics benchmarks




- Large model files and data are excluded from git (see `.gitignore`)


#
