# Comma Video Compression Evaluation Pipeline Audit

This document details the evaluation pipeline in the `comma-vc-fork` repository. It provides a frame-level analysis of how video sequences are selected, loaded, and preprocessed, along with the precise mathematics governing the **SegNet** and **PoseNet** distortion metrics.

---

## 1. Pipeline Overview & Global Score Formula

The evaluation pipeline evaluates submission files against ground truth videos by comparing reconstructed frames to original frames. The final objective is to minimize a combined score representing both visual quality preservation and data compression performance:

$$\text{Score} = 100 \cdot D_{\text{SegNet}} + \sqrt{10 \cdot D_{\text{PoseNet}}} + 25 \cdot R$$

Where:
*   $D_{\text{SegNet}}$: Average semantic distortion (accuracy mismatch) computed across the dataset.
*   $D_{\text{PoseNet}}$: Average 3D ego-motion prediction distortion (Mean Squared Error) computed across the dataset.
*   $R$: The compression rate, computed as the total file size of the compressed archive (`archive.zip`) divided by the total size of the original uncompressed video files:
    $$R = \frac{\text{Compressed Size (bytes)}}{\text{Original Size (bytes)}}$$

---

## 2. Frame Selection and Loading Nuances

The dataset partitions video files into non-overlapping frame sequences of length $S = 2$ (`seq_len = 2`).

### 2.1 Dataset Classes Comparison
Three dataset classes inherit from the base `VideoDataset` class in [frame_utils.py](file:///home/iantitor/Documents/projects/professional/comma-vc/comma-vc-fork/frame_utils.py):

| Dataset Class | Input Source / Format | Execution Target | Under-the-hood Mechanism |
| :--- | :--- | :--- | :--- |
| **`DaliVideoDataset`** | Original Video (e.g., `.mp4`, `.mkv`, `.hevc`) | CUDA (GPU) | Uses NVIDIA DALI video reader input pipeline (`fn.experimental.inputs.video`). |
| **`AVVideoDataset`** | Original Video (e.g., `.mp4`, `.mkv`, `.hevc`) | CPU / MPS | Uses PyAV to decode video streams sequentially. |
| **`TensorVideoDataset`** | Decompressed raw frame files (`.raw`) | CPU / GPU / MPS | Maps flat, binary `uint8` raw arrays of shape $(N, H, W, 3)$ using `np.memmap`. |

### 2.2 Frame Index Selection and Discarding
During iteration, frames are batched into disjoint sequences of length $S = 2$ starting from index 0:
*   **Sequence 0:** Frame 0, Frame 1
*   **Sequence 1:** Frame 2, Frame 3
*   ...
*   **Sequence $k$:** Frame $2k$, Frame $2k+1$

**Odd-Frame Discarding:**
If a video contains an odd number of frames $N = 2k + 1$, the last frame (index $2k$) is appended to the internal buffer (`seq_buf`), but the dataset loop terminates without yielding this partial sequence of length 1.
*   In `AVVideoDataset` and `TensorVideoDataset`, this buffer is discarded.
*   In `DaliVideoDataset`, the length of sequences is computed as `frames_per_file // seq_len`, which mathematically truncates any odd frames.

Thus, the final frame of any video with an odd frame count is completely ignored in the evaluation.

### 2.3 Colorspace Consistency
To ensure identical evaluation across GPU (via DALI) and CPU (via PyAV), the pipeline normalizes video frames to a standard RGB space. `AVVideoDataset` uses `yuv420_to_rgb` in [frame_utils.py](file:///home/iantitor/Documents/projects/professional/comma-vc/comma-vc-fork/frame_utils.py#L159-L183) to match NVDEC's hardware decoder colorspace conversion (BT.601 limited-range):

$$\begin{aligned}
Y(y, x) &= \text{plane}_{0}(y, x) \\
U_{\text{up}}(y, x) &= \text{BilinearUpsample}(\text{plane}_{1})(y, x) \\
V_{\text{up}}(y, x) &= \text{BilinearUpsample}(\text{plane}_{2})(y, x)
\end{aligned}$$

The floating-point YUV values are rescaled and converted to RGB:

$$\begin{aligned}
Y_{\text{scaled}} &= (Y - 16) \cdot \frac{255}{219} \\
U_{\text{scaled}} &= (U_{\text{up}} - 128) \cdot \frac{255}{224} \\
V_{\text{scaled}} &= (V_{\text{up}} - 128) \cdot \frac{255}{224}
\end{aligned}$$

$$\begin{aligned}
R &= \text{clamp}(Y_{\text{scaled}} + 1.402 \cdot V_{\text{scaled}}, 0, 255) \\
G &= \text{clamp}(Y_{\text{scaled}} - 0.344136 \cdot U_{\text{scaled}} - 0.714136 \cdot V_{\text{scaled}}, 0, 255) \\
B &= \text{clamp}(Y_{\text{scaled}} + 1.772 \cdot U_{\text{scaled}}, 0, 255)
\end{aligned}$$

---

## 3. SegNet Semantic Distortion Metric

The SegNet model assesses structural semantic discrepancy. It is implemented in [modules.py](file:///home/iantitor/Documents/projects/professional/comma-vc/comma-vc-fork/modules.py#L103-L129).

### 3.1 Preprocessing and Frame Slicing
For each sequence $(f_{2k}, f_{2k+1})$:
1.  **Frame Selection:** SegNet processes **only the second frame** of the sequence ($f_{2k+1}$):
    ```python
    x = x[:, -1, ...] # Slice last frame in batch
    ```
    This means even-indexed frames ($0, 2, 4, \dots, 2k$) are completely ignored by the semantic metric.
2.  **Resizing:** The frame is resized from its native camera size $(874 \times 1164)$ to the network input size $(384 \times 512)$ using bilinear interpolation:
    ```python
    torch.nn.functional.interpolate(x, size=(384, 512), mode='bilinear')
    ```

### 3.2 Metric Mathematics
The network produces logits $P \in \mathbb{R}^{B \times 5 \times 384 \times 512}$ corresponding to 5 class channels.
The class index for each pixel $(y, x)$ is determined by taking the argmax across the channel dimension:

$$\hat{C}(y, x) = \arg\max_{c \in \{0, 1, 2, 3, 4\}} P(c, y, x)$$

Let $\hat{C}_{\text{GT}}$ be the predicted classes for the ground truth frame, and $\hat{C}_{\text{Comp}}$ be the predicted classes for the compressed frame. The distortion for a single sample is the spatial mean of their class disagreements:

$$D_{\text{SegNet}} = \frac{1}{384 \cdot 512} \sum_{y=0}^{383} \sum_{x=0}^{511} \mathbb{I}\left( \hat{C}_{\text{GT}}(y, x) \neq \hat{C}_{\text{Comp}}(y, x) \right)$$

Where $\mathbb{I}$ is the indicator function. The dataset-wide SegNet distortion is the average of these sample distortions.

---

## 4. PoseNet 3D Ego-Motion Metric

The PoseNet model evaluates the preservation of temporal driving dynamics. It is implemented in [modules.py](file:///home/iantitor/Documents/projects/professional/comma-vc/comma-vc-fork/modules.py#L61-L102).

### 4.1 Input Preprocessing (YUV6 Formulation)
For each sequence containing the pair $(f_{2k}, f_{2k+1})$, the pipeline processes both frames:
1.  **Resizing:** Both frames are resized from $(874 \times 1164)$ to $(384 \times 512)$ using bilinear interpolation.
2.  **RGB to YUV6 conversion:** Each frame is converted into a 6-channel YUV6 tensor representation of spatial size $(192 \dots 256)$ via `rgb_to_yuv6` in [frame_utils.py](file:///home/iantitor/Documents/projects/professional/comma-vc/comma-vc-fork/frame_utils.py#L51-L78):
    *   Compute BT.601 limited-range Y, U, V channels clamped to $[0.0, 255.0]$.
    *   Subsample the U and V channels to half-resolution $(192 \times 256)$ using a $2 \times 2$ block average:
        $$U_{\text{sub}}(y, x) = \frac{1}{4} \sum_{m=0}^{1} \sum_{n=0}^{1} U(2y+m, 2x+n)$$
    *   Slice the Y channel into 4 half-resolution channels representing the spatial offsets of the $2 \times 2$ grid:
        $$\begin{aligned}
        Y_{00}(y, x) &= Y(2y, 2x) \\
        Y_{10}(y, x) &= Y(2y+1, 2x) \\
        Y_{01}(y, x) &= Y(2y, 2x+1) \\
        Y_{11}(y, x) &= Y(2y+1, 2x+1)
        \end{aligned}$$
    *   Stack these elements to form a 6-channel tensor: $[Y_{00}, Y_{10}, Y_{01}, Y_{11}, U_{\text{sub}}, V_{\text{sub}}]$.
3.  **Temporal Concatenation:** The two 6-channel tensors are concatenated along the channel dimension. The final PoseNet input tensor has shape $(B, 12, 192, 256)$.
4.  **Normalization:** The input is normalized using mean 127.5 and standard deviation 63.75:
    $$x_{\text{normalized}} = \frac{x - 127.5}{63.75}$$

### 4.2 Metric Mathematics
The normalized input is passed through PoseNet (`fastvit_t12` backbone + summarizer + Hydra head), outputting a 12-dimensional vector $\mathbf{p} \in \mathbb{R}^{12}$ per sequence. 
*   The first 6 elements of $\mathbf{p}$ represent the 6-Degrees-of-Freedom (DoF) camera ego-motion parameters (3 translation, 3 rotation). Let this subvector be $\mathbf{v} = \mathbf{p}[:6] \in \mathbb{R}^6$.
*   The remaining 6 elements represent log-variance/uncertainty parameters and are completely ignored during distortion computation.

The distortion metric is the Mean Squared Error (MSE) computed over the 6 ego-motion parameters:

$$D_{\text{PoseNet}} = \frac{1}{6} \sum_{i=0}^{5} \left( v^{\text{GT}}_i - v^{\text{Comp}}_i \right)^2$$

Where $\mathbf{v}^{\text{GT}}$ and $\mathbf{v}^{\text{Comp}}$ represent the ego-motion vectors predicted from the ground truth sequence and the compressed sequence, respectively. The dataset-wide PoseNet distortion is the average of these sample MSE values.
