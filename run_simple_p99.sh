#!/bin/bash
# Run Case4 Simple P99 Benchmark with visible output

cd "$(dirname "$0")"
PROJECT_ROOT="$(cd .. && pwd)"

# Activate venv and run
source "$PROJECT_ROOT/venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create output directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/output"

# Run and tee output to both terminal and file
python3 case4_simple_p99_benchmark.py 2>&1 | tee "$PROJECT_ROOT/output/simple_p99_output.log"

echo ""
echo "✅ Benchmark completed!"
