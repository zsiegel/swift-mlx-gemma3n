# swift-mlx-gemma3n

A swift-mlx port of Gemma3n. This was hacked together over a few days of work and is currently a WIP. The model could certainly be optimized further. I used Claude and o3 to help me through some tricky bits where MLX differs with Swift or is behind the Python lib.

## Instructions

- use `load-models.sh` to download the weights via HF
- update the 2B model path at the top of the `LanguageGenerationTest.swift`
- run `run-swift-generate.sh` to see language generation!

## Notes

- I had on idea people were already hacking on this is mlx-swift-examples when I started this
  - this was really just a fun side project I wanted to attempt myself for putting into a small MacOS app I am building
- This was tested primarily on Mac OS Tahoe 26.0 (Beta)
  - This means its using the newer Swift toolchain but should be easy to set back to pre-beta toolchain
- Update the path to your models in `LanguageGenerationTest.swift` before running the generation script
  - Figuring out paths in test was getting annoying so it is hardcoded for now
- Note that you can not use `swift test` you must use `xcodebuild` and there is a simple helper `run-swift-generate.sh` script to run the test to do generation.
- There is a helper script to download both the `2B` and `4B` models called `load-models.sh` from HuggingFace
- ONLY the language model is supported current - both `2B` and `4B`
- Video/Audio support to follow hopefully soon!

Special thanks to the following repos

- https://github.com/Blaizzy/mlx-vlm
- https://github.com/ml-explore/mlx-swift
- https://github.com/ml-explore/mlx
- https://github.com/ml-explore/mlx-swift-examples
