#!/bin/bash
# Case4: Combined IPC Runner
# Runs JIT client and Graph generator server in separate processes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${1:-meta-llama/Llama-2-7b-hf}"

echo "=================================================================================="
echo "Case4: Combined IPC Runner"
echo "=================================================================================="
echo "Model: $MODEL_PATH"
echo "=================================================================================="

# Activate venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: venv not found, using system Python"
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the combined orchestrator
echo ""
echo "🚀 Starting combined IPC inference..."
echo ""

cd "$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python3 case4_combined_ipc_orchestrator.py \
    --model-name "$MODEL_PATH" \
    --prompt "The future of AI is" \
    --max-tokens 20

echo ""
echo "✅ Done!"

