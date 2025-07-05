#!/bin/bash

# Simple script to run generation test  
# Usage: ./generate.sh

echo "Running generation test..."
echo ""

# Run the LanguageGenerationTest through xcodebuild
xcodebuild -scheme gemma3n-lib-swift \
    -destination 'platform=macOS,arch=arm64' \
    -configuration Debug \
    OTHER_CFLAGS="-Wunused-const-variable" \
    test -only-testing:gemma3n-lib-swiftTests/LanguageGenerationTest