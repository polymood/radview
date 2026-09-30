# Changelog

All important changes to this project are in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-30

First release.

### Added

- GPU viewer for SAR and ISAR images, with wgpu (Vulkan, Metal, DirectX 12, OpenGL) and egui.
- NITF 2.1 and NSIF 1.0: SICD (complex I/Q), SIDD, uncompressed (NC) and masked (NM) blocks, IMODE B, P and S. Image segments of one image (files larger than 10 GB) show as one image.
- TIFF, BigTIFF and COG: strips and tiles, LZW, Deflate, Zstd and PackBits compression, predictors 2 and 3, both byte orders.
- Pixel types: 8, 16 and 32-bit integers, 32 and 64-bit floats, complex integers and floats.
- Band selection: amplitude, phase, I and Q for complex data, and the band for multi-band data.
- Ten color map presets (Gray, Viridis, Magma, Inferno, Plasma, Cividis, Turbo, Jet, Hot, Sepia). You can edit, add and remove the color stops.
- Stretch: minimum, maximum, gamma, linear or dB scale, automatic clip percentage, invert, value 0 as no data.
- Open a file from the command line, with a file dialog, with a path field, or with drag and drop.
- Background 2x2 mean pyramid on all CPU cores, with a fast preview.
- Pixel position and value under the cursor.

[1.0.0]: https://github.com/polymood/radview/releases/tag/v1.0.0
