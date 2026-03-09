# HLS4ML Cross-Compilation Workflow For Apple Silicon (M-Series)

**Author:** Peter Dunniece  
**Date:** March 9, 2026

---

## 1. Overview

Running Intel HLS tools (OneAPI/Quartus) on Apple Silicon creates an architecture conflict.

### The Problem
- Intel tools require x86_64. macOS emulates this via Rosetta 2, but the x86 version of TensorFlow (required for hls4ml) relies on AVX instructions.
- Rosetta 2 cannot translate AVX, causing Illegal Instruction crashes in a single container.

### The Solution
A dual-container workflow sharing a single file system:

1. **Dev Container (ARM64)**: Runs native Python/TensorFlow. Used for model training and generating C++ files.
2. **Build Server (x86_64)**: Background Intel container used only for synthesis (cmake, make).

---

## 2. Setup: Build Server

Build the Intel Docker image locally for platform consistency.

### 2.1 Dockerfile

Create a Dockerfile using the Intel OneAPI Basekit:

```dockerfile
# Base Image: Official Intel OneAPI (x86 only)
FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu22.04

# Install basic build tools and Python prerequisites
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install HLS4ML dependencies (for verification scripts)
RUN pip3 install numpy tensorflow hls4ml[profiling]

# Set the working directory
WORKDIR /workspace
```

### 2.2 Build the Image

Force the `linux/amd64` platform to trigger Rosetta 2 emulation. Run this in the folder containing your Dockerfile:

```bash
docker build --platform linux/amd64 -t hls4ml-intel-local .
```

**Note:** This pulls large dependencies and takes 10–20 minutes.

---

## 3. Setup: Automation Script

Create `build_report.sh` in your project root to link the development environment and the build container.

```bash
#!/bin/bash

# Default project folder name
PROJECT_NAME=${1:-my_project}

echo "Launching build for: $PROJECT_NAME"

# Execute build commands inside the background 'intel_builder' container
# CRITICAL: We source oneapi-vars.sh to load the compiler environment
docker exec -t intel_builder bash -c "
    source /opt/intel/oneapi/2025.0/oneapi-vars.sh --force && \
    cd /workspace/hls4ml-tutorial/$PROJECT_NAME && \
    mkdir -p build && cd build && \
    cmake .. -DTARGET_DEVICE='Intel' && \
    make report
"

echo "Build complete. Check reports/report.html for results."
```

Make it executable:

```bash
chmod +x build_report.sh
```

---

## 4. Daily Workflow

### 4.1 Start the Environment (Mac Terminal)

Start the Intel container after a reboot or Docker restart:

```bash
# 1. Clean up any stopped instance first
docker rm -f intel_builder

# 2. Navigate to your project root (The "Parent" Folder)
cd "/Users/peterdunniece/Imperial Masters/Self Study Project/HLS4ML Tutorials"

# 3. Mount the current directory ($(pwd)) to /workspace inside the container
docker run -d -t --name intel_builder \
    --platform linux/amd64 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    hls4ml-intel-local bash
```

### 4.2 Generate C++ Files (VS Code)

Run your HLS4ML script in the ARM64 Python container. Use `write()`, not `build()`:

```python
# ... Define and Train Model ...

# Generate Project (Do not compile here)
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my_project',
    backend='oneAPI'
)

# Write only - Compilation happens in the next step
hls_model.write()
print("C++ generated successfully.")
```

### 4.3 Synthesise (Mac Terminal)

Run the script from your Mac terminal to generate resource reports (ALMs, DSPs, Latency) in `build/reports/`:

```bash
./build_report.sh my_project
```

---
