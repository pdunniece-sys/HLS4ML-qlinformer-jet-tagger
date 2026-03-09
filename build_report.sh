#!/bin/bash

# Usage: ./build_report.sh [project_folder_name]
PROJECT_NAME=${1:-manual_intel_project}

echo "Sending build command to Intel Container for: $PROJECT_NAME"

# We use /workspace because that is where your Mac folder is mounted INSIDE the container
docker exec -t intel_builder bash -c "
    source /opt/intel/oneapi/2025.0/oneapi-vars.sh --force && \
    cd \"/workspace/$PROJECT_NAME\" && \
    mkdir -p build && cd build && \
    cmake .. && \
    make report
"

echo "Build command finished."