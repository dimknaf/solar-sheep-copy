#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ci_physics_nightly.sh — CI physics-only nightly runner (python.sh / Kit CLI).
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI physics-only nightly run via python.sh
#
# Usage (from _build/linux-x86_64/release):
#   ./python.sh path/to/ci_physics_nightly.sh [SCRIPT] [SCRIPT_ARGS...]
#
# The headless physics kit file bakes in renderer=false and window=false,
# so python.sh needs no extra flags beyond pointing at the kit file.
# For explicit CLI override (without the kit file), pass:
#   --/renderer/enabled=false --/app/window/enabled=false --/app/livestream/enabled=false

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-"${SCRIPT_DIR}/../../../_build/linux-x86_64/release"}"

# --------------------------------------------------------------------------
# Option 1: Kit CLI with the headless physics kit file (preferred for CI)
# The kit file disables renderer, window, UI extensions, and rendering
# subsystems. Flags on the CLI override settings in the kit file if needed.
# --------------------------------------------------------------------------
run_kit_physics_headless() {
    local test_script="${1:?Usage: run_kit_physics_headless SCRIPT [ARGS...]}"
    shift

    "${ISAAC_SIM_DIR}/kit/kit" \
        "${ISAAC_SIM_DIR}/apps/isaacsim.exp.base.python.physics.headless.kit" \
        --no-window \
        --/app/renderer/enabled=false \
        --exec "${test_script}" \
        "$@"
}

# --------------------------------------------------------------------------
# Option 2: python.sh with inline headless flags (no kit file dependency)
# Use when the test script creates SimulationApp itself.
# python.sh sets up CARB_APP_PATH, PYTHONPATH, LD_LIBRARY_PATH automatically.
# --------------------------------------------------------------------------
run_python_physics_headless() {
    local test_script="${1:?Usage: run_python_physics_headless SCRIPT [ARGS...]}"
    shift

    "${ISAAC_SIM_DIR}/python.sh" \
        "${test_script}" \
        --/renderer/enabled=false \
        --/app/window/enabled=false \
        --/app/livestream/enabled=false \
        "$@"
}

# --------------------------------------------------------------------------
# Dispatch: default to python.sh mode for standalone test scripts
# --------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <test_script.py> [args...]"
    echo ""
    echo "Environment variables:"
    echo "  ISAAC_SIM_DIR  Path to built Isaac Sim (default: _build/linux-x86_64/release)"
    echo "  CI_KIT_MODE    Set to '1' to use kit CLI instead of python.sh"
    exit 1
fi

if [[ "${CI_KIT_MODE:-0}" == "1" ]]; then
    run_kit_physics_headless "$@"
else
    run_python_physics_headless "$@"
fi
