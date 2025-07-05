#!/bin/bash

# Run Swift generation script
echo "Running Swift generation script..."

# Change to the Swift directory and run generate.sh directly
# This allows output to stream in real-time
cd gemma3n-lib-swift

# Run the generation script and capture the exit code
./generate.sh
exit_code=$?

# Check the exit code and provide appropriate feedback
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ TEST FAILED: Swift generation exited with code $exit_code"
    echo ""
    exit $exit_code
else
    echo ""
    echo "✅ Swift generation completed successfully"
    echo ""
    exit 0
fi