# radview

Fast viewer for SAR and ISAR images. Rust, wgpu (WebGPU API), egui.

## Formats

| Format | Supported |
|---|---|
| NITF 2.1 / NSIF 1.0 | SICD (complex I/Q), SIDD, uncompressed (NC) and masked (NM) blocks, all image segments of one image (for files larger than 10 GB) |
| TIFF / BigTIFF / COG | Strips and tiles. None, LZW, Deflate, Zstd and PackBits compression. Predictors 2 and 3 |
| Pixel types | 8, 16 and 32-bit integers, 32 and 64-bit floats, complex integers and floats |

For complex data, you can show amplitude, phase, I or Q. For multi-band data, you can select the band.

JPEG and JPEG 2000 compression are not supported.

## Use

```
radview [file]
```

You can also open a file with the **Open** button, with **Ctrl+O**, or by dropping the file on the window.

| Input | Action |
|---|---|
| Mouse wheel | Zoom at the cursor |
| Drag | Pan |
| Double-click, F | Fit the image to the window |
| 1 | Zoom 1:1 |
| C | Next color map |
| I | Invert the color map |
| H | Show or hide the side panel |

The side panel has these controls:

- Color map presets. You can edit, add or remove the color stops.
- Stretch: minimum, maximum, gamma, linear or dB scale, automatic clip percentage.
- Value 0 as no data.
- Pixel position and value under the cursor.

## Build

Linux:

```
cargo build --release
```

Windows executable, from Linux or WSL (you must have the `gcc-mingw-w64-x86-64-posix` package):

```
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

The build uses `target-cpu=x86-64-v3` (AVX2). To run on older CPUs, remove this flag from `.cargo/config.toml`.

## Operation

The file is memory-mapped. Uncompressed data is not copied. The viewer builds a 2x2 mean pyramid in the background on all CPU cores. A preview of the lowest resolution level shows first.

The GPU keeps 512 x 512 tiles in one texture array (256 MB). All visible tiles draw in one instanced draw call. The shader applies the stretch and the color map, so a change of these settings does not load the data again. 8-bit data stays 8-bit. Other data is stored as 16-bit floats.

In WSL, Vulkan uses only the CPU. Thus the viewer uses the Mesa d3d12 OpenGL driver over X11, which uses the GPU.

## Test

```
cargo test --release
```

To also compare decoded values with files from another TIFF writer, set `TEST_IMAGES` to a list file. Each line has these tab-separated fields: `path[#band]`, width, height, then `x y value` for each point.

## License

You can use this software under the terms of one of these licenses:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
