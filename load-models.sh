#!/usr/bin/env bash
#
# pull_gemma3n_models.sh
# Clone MLX-community Gemma-3n weights into gemma3n-lib-swift/models
# without leaving behind any .git cruft.
#
# Usage:
#   ./pull_gemma3n_models.sh                # clones under ./gemma3n-lib-swift/models
#   ./pull_gemma3n_models.sh /tmp/models    # clones under /tmp/models
#
set -euo pipefail

DEST_ROOT="${1:-gemma3n-lib-swift/models}"

declare -A HF_REPOS=(
  [gemma-3n-E2B-it-bf16]="https://huggingface.co/mlx-community/gemma-3n-E2B-it-bf16.git"
  [gemma-3n-E4B-it-bf16]="https://huggingface.co/mlx-community/gemma-3n-E4B-it-bf16.git"
)

echo ">>> Creating/refreshing   $DEST_ROOT"
mkdir -p "$DEST_ROOT"

for DIR in "${!HF_REPOS[@]}"; do
  URL="${HF_REPOS[$DIR]}"
  OUTDIR="$DEST_ROOT/$DIR"

  echo "── Cloning $URL → $OUTDIR"
  rm -rf "$OUTDIR"
  git clone --depth 1 "$URL" "$OUTDIR"

  # If you only need the pointer files (fast), uncomment the next line:
  # GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$URL" "$OUTDIR"

  rm -rf "$OUTDIR/.git" "$OUTDIR/.gitattributes" "$OUTDIR/.github"
done

echo "✔  Done. Models live in:  $DEST_ROOT"