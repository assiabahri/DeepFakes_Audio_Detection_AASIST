### AASIST

### New words :

**Spoofing :** fake voice or synthetic audio pretends to a real person.
**Amplitude :** loudness/air pressure height in an instant of time.
**Frequency :** How many full wave cycles happen in 1 second (measured Hz)
**Magnitude :** loudness of a frequency when plotted as positive value in a 2D spectrogram.
**Phase :** The timing angle (from 0° to 360°) of a wave at a specific millisecond.

### AASIST: Anti-Spoofing with Audio Spectro-Temporal Graph Attention Networks

AASIST is an ai model designed to detect audio deepfakes, in this tutorial we will explore how AASIST works step by step in a very simple way

### Global Idea :

Those are in general the global steps of AASIST :
![aasist_steps](aasist_data_flow_step_by_step.svg)
![aasist_steps](aasist_data_flow_step_by_step.png)

Now let's go through them stage by stage :

### 1.Raw Audio :

The input to AASIST model is a raw, unprocessed audio so the model sees it as a 1D vector of numbers representing amplitude over time.

note : some models need to receive modified audios (for example spectograms(2D)...) to predict if they're fake or not but when we modify the audio (from 1D to 2D) we risk to lose some features that help the model to predict in a correct way.

### WHY 2D SPECTOGRAM LOSES A FEATURE AND WHAT IS THAT FEATURE ?

When converting 1D audio into 2D spectogram the feature phase will be lost so we need to avoid this because AI voice generators struggle to reconstruct natural phase (they leave errors).
Raw audio preserves phase so AASIST can catch these deepfake clues directly.

## 2. Encoder :

At the encoder stage the raw 1D waveform will be transformed to a deep 3D feature tensor $F$ via those steps :

### 2.1 Sinc-Convolution Layer :

At this step the 1D raw wave passes through SincNet filterbank containing 70 parallel bandpass filters.
Each filter isolates a specific frequency band (low, mid, or high) so :
Input : Raw Audio waveform [B, 1, 64600]
Output : 2D Tensor [B, 70, 64600]
(where B is Batch size, 70 is frequency Channels, and 64600 is Raw Time Samples)

### 2.2 The 6 Residual Blocks :

This stage performs two functions :

1.  **Feature Extraction(Expanding Channels):** Expands frequency representations from $1 \rightarrow 32 \rightarrow 64 \rightarrow 128$ channels.
2.  **Time Compression (Max Pooling):** Downsamples the length dimension from 64,600 down to 200 time steps.

- **What is a ResBlock?** A block of convolutional layers with a **shortcut connection** that adds the input directly to the output ($x + f(x)$), preventing vanishing gradients as the image shows :
  ![Residual Block](image.png)
- **What is Max Pooling?** A operation that reduces spatial dimensions by selecting only the maximum value within a moving window as the image shows :
  ![Max Pooling](image-1.png)

### 2.3 Encoder Output Tensor :

The final output is a 3D feature tensor $F \in \mathbb{R}^{C \times S \times T}$:

- **$C = 128$:** Feature Channels
- **$S = 23$:** Spectral (Frequency) Bins
- **$T = 200$:** Compressed Time Steps

### Resumé of Encoder stage :

Raw waveform (1D, 64600)
|
|
Sinc_Conv Layer (70 frequency Bands)
|
|
Stack of 6 ResBlocks (128 channels, Max Pooling)
|
|
Feature Tensor F (128*23*200)

## 3. Graph construction :

AASIST converts the 3D tensor $F$ into graph representations to analyze dependencies across time and frequency.

### 3.1 Spectral and Temporal Graphs

- **$G_s$ (Spectral Graph):** 23 Spectral Nodes (analyzes relationships across **Frequency**).
- **$G_t$ (Temporal Graph):** 200 Temporal Nodes (analyzes relationships across **Time**).

### 3.2 Combined Graph ($G_{st}$)

Combines $G_s$ and $G_t$ into a single **Heterogeneous Graph** containing $223$ nodes ($23 + 200$).
An extra **Stack Node** is added and connected to every single node to act as a central information hub.

### 3.3 Max Graph Operation (MGO) :

The $G_{st}$ graph passes through two parallel attention and pooling branches. MGO takes the element-wise maximum ($\max(A, B)$) between both branches to retain the strongest feature activations while discarding weak noise.

## 4. Readout and Classification

### 4.1 & 4.2 Feature Pooling

The updated graph features are aggregated into vector representations:

1. **Max Pooling ($e_{\max}$):** Keeps the highest activation per channel ($128$-dimensional vector).
2. **Average Pooling ($e_{\text{avg}}$):** Keeps the average value per channel ($128$-dimensional vector).
3. **Stack Node Extraction ($e_{\text{stack}}$):** Extracts the final embedded state of the central Stack Node ($128$-dimensional vector).

### 4.3 Vector Concatenation & Linear Classifier

The three vectors are concatenated into a single output embedding $e_{\text{out}}$:

$$e_{\text{out}} = [e_{\max} \parallel e_{\text{avg}} \parallel e_{\text{stack}}] \quad \in \mathbb{R}^{384} \quad (128 + 128 + 128 = 384)$$

#### Linear Classification Layer:

$$z = W \cdot e_{\text{out}} + b$$

- **Weight Matrix ($W$):** Shape $[2 \times 384]$. It maps the 384 features into 2 output logits:
  - $z_0$: Score for **Bona-fide (Real)**
  - $z_1$: Score for **Spoof (Fake)**
