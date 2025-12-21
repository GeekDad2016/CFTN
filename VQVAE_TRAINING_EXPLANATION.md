# VQ-VAE Training Explanation

This document explains how the Vector-Quantized Variational Autoencoder (VQ-VAE) in this project learns to represent and reconstruct images.

## 1. Does it learn the "Whole Image" or "Features"?
The VQ-VAE learns **spatial features**, not just a single global representation of the whole image. 

Because the Encoder is built using Convolutional layers, it processes the image patch-by-patch. Instead of compressing the entire 128x128 image into one single vector, it compresses it into a **grid of discrete codes**. 

*   **Local Representation:** Each point in the latent grid corresponds to a specific area (receptive field) in the original image.
*   **Feature Learning:** The model learns a "vocabulary" of visual features (edges, textures, colors) in its codebook and describes the image as a sequence/grid of these vocabulary items.

## 2. Feature and Latent Dimensions
Based on your current configuration (`config/vqvae_naruto.yaml`):

*   **Input Image:** $128 \times 128$ pixels with 3 channels (RGB).
*   **Downsampling:** The encoder uses 2 downsampling layers (each with stride 2).
    *   $128 \rightarrow 64 \rightarrow 32$.
*   **Latent Grid Size:** The resulting feature map is **$32 \times 32$**.
*   **Feature Depth (Embedding Dim):** Each of the $32 \times 32$ cells is represented by a vector of size **256**.
*   **Codebook Size:** There are **4096** possible vectors in the "vocabulary".

**In summary:** The "feature" is a $32 \times 32$ grid. Each cell in that grid is assigned an index (from 0 to 4095) pointing to a 256-dimensional vector in the codebook.

## 3. The Training Process
Training a VQ-VAE involves balancing three distinct goals:

### A. Reconstruction Loss (MSE or L1)
This measures how similar the output image is to the input image. It forces the **Encoder** to extract useful information and the **Decoder** to interpret the quantized vectors correctly.

### B. Codebook Loss (handled by EMA in this project)
The codebook vectors need to move closer to the encoder outputs so they can accurately represent the data. In your project, we use **Exponential Moving Averages (EMA)**. This is a more stable way to update the codebook than using standard backpropagation. It calculates the average position of all encoder outputs assigned to a specific codebook entry and moves that entry there.

### C. Commitment Loss
The encoder outputs could theoretically have any value, but they need to "commit" to the vectors available in the codebook. The commitment loss prevents the encoder's output from growing too far away from the nearest codebook vector.

## 4. Key Training Metrics
When monitoring your logs or WandB, look for these:
1.  **Recon Loss:** Should steadily decrease. If it stays high, the model capacity (hiddens/residual layers) might be too low.
2.  **Perplexity:** This measures how many of your 4096 codebook entries are actually being used. 
    *   If Perplexity is very low (e.g., < 50), you have **Codebook Collapse**. The model is only using a few "words" to describe everything.
    *   If Perplexity is high (e.g., > 1000), the model is utilizing its vocabulary well.
3.  **VQ Loss:** Represents the commitment and codebook alignment. It should be small but stable.
