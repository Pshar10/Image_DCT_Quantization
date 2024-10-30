
# Image Discrete Cosine Transform (DCT) Quantization

This repository demonstrates image compression using Discrete Cosine Transform (DCT) and quantization with different quality matrices. The `quant.py` file provides an interactive program to apply DCT-based compression on a grayscale image and observe the effects of different quantization matrices.

## Overview

The DCT is a commonly used transform in image compression, particularly in JPEG compression. By transforming the image into frequency space and then quantizing it, we can achieve high levels of compression. This program allows you to apply different quantization levels, each trading off compression ratio and image quality.

## Files in Repository

- `quant.py`: Main script that loads an image, performs DCT-based quantization, and displays the compressed and decompressed images.
- **Image File**: The image file `i.jpg` should be located in the same directory as `quant.py` for processing.

## Requirements

- **Python Libraries**:
  - `numpy`
  - `opencv-python`
  - `matplotlib`

Install dependencies using:
```bash
pip install numpy opencv-python matplotlib
```

## Usage

1. Place a grayscale image named `i.jpg` in the project directory.
2. Run the `quant.py` file:
   ```bash
   python quant.py
   ```
3. Follow the prompts to select different quantization matrices:

   - **a**: Low-quality, high-compression (Q10)
   - **b**: Balanced quality and compression (Q50)
   - **c**: High-quality, low-compression (Q90)
   - **q**: Quit the program

## Quantization Matrices

The following quantization matrices are used in this program:

- **Q10**: Low-quality, higher compression.
- **Q50**: Balanced quality and compression, commonly used in JPEG compression.
- **Q90**: High-quality, lower compression.

## Key Functions

- **Quant_matrix**: Selects the quantization matrix based on user input.
- **Quantization**: Applies DCT to 8x8 blocks of the image, quantizes using the selected matrix, and then reconstructs the image using the inverse DCT.

## Example Output

The program displays the compressed (frequency domain) and decompressed images with dimensions of the reconstructed image.

---

This project provides a practical demonstration of DCT and quantization, highlighting the trade-off between image quality and compression.

