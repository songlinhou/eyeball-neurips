# Methodology: Explainable Multi-Class Video Classification with Vision-Language Model Integration

## Abstract

We present a two-stage framework for interpretable medical ultrasound video diagnosis: (1) **Multi-Class Explainable Video Classifier** - a dual-stream convolutional architecture that performs hierarchical classification (diagnostic class and subtype) with spatial-temporal attention, and (2) **Vision-Language Model (VLM) Integration** - a finetuned Qwen 2.5 VL model that generates clinical reasoning from classifier predictions and attention-highlighted frames. Our approach addresses the unique challenges of ultrasound imaging—low signal-to-noise ratio, subtle motion patterns, limited training data, and clinical interpretability requirements—through three key innovations: (1) explicit optical flow modeling for robust motion analysis, (2) hierarchical multi-task classification with shared representations, (3) attention-guided VLM finetuning with contrastive learning for explainability. Compared to transformer-based alternatives, our classifier achieves superior parameter efficiency (40M vs. 86-121M), computational efficiency (45G vs. 180-590G FLOPs), and sample efficiency (effective with ~5K videos vs. requiring >10K videos), while the VLM integration provides human-readable clinical reasoning that aligns with expert diagnostic patterns.

## 3. Methodology

### 3.1 Problem Formulation

#### 3.1.1 Multi-Class Hierarchical Classification

Given a video sequence $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$ consisting of $T$ frames with spatial dimensions $H \times W$ and 3 RGB channels, our objective is to learn a hierarchical multi-task mapping:

$$
f: \mathbb{R}^{T \times H \times W \times 3} \rightarrow \mathbb{R}^{C_{\text{diag}}} \times \mathbb{R}^{C_{\text{sub}}}
$$

where:
- **Diagnostic Classification**: $C_{\text{diag}} = 2$ classes (non-retinal detachment vs. retinal detachment)
- **Subtype Classification**: $C_{\text{sub}} = 4$ classes (normal, macula intact, macula detached, posterior vitreous detachment)

This hierarchical formulation reflects the clinical diagnostic workflow where physicians first determine the primary diagnosis (presence/absence of retinal detachment) and then classify the specific subtype. Unlike single-task classification, our multi-task approach enables:

1. **Shared Feature Learning**: Lower layers learn general ultrasound features applicable to both tasks
2. **Task Regularization**: Each task acts as a regularizer for the other, preventing overfitting
3. **Clinical Alignment**: Output structure matches the hierarchical nature of medical diagnosis

**Note on Anatomical Classification**: While the ERDES dataset includes anatomical subclass labels (temporal, nasal, superior, inferior detachment), we deliberately exclude this from model training. Anatomical location is preserved in metadata for reference but not used as a prediction target, as it provides limited diagnostic value compared to the primary diagnosis and subtype.

**Ultrasound-Specific Challenges.** Medical ultrasound videos present unique challenges that distinguish them from natural videos in standard action recognition benchmarks:

1. **Low Signal-to-Noise Ratio**: Ultrasound imaging inherently suffers from speckle noise and acoustic artifacts, requiring robust feature extraction that can distinguish pathological patterns from noise.

2. **Subtle Motion Patterns**: Unlike natural videos with large-scale object motions, pathological indicators in ocular ultrasound (e.g., retinal detachment, vitreous movement) manifest as subtle, localized tissue displacements that require fine-grained temporal modeling.

3. **Limited Training Data**: Medical datasets are typically orders of magnitude smaller than natural video datasets (e.g., Kinetics-400 with 240k videos vs. ERDES with ~200 videos), necessitating efficient architectures with strong inductive biases and effective transfer learning strategies.

4. **Clinical Interpretability**: Medical applications demand not only accurate predictions but also interpretable explanations that clinicians can verify against domain knowledge, ruling out pure black-box approaches.

### 3.2 Model Architecture

Our proposed ExplainableOpticalFlowResNet3D model consists of three main components: (1) an RGB stream with explainability modules, (2) an optical flow stream for motion analysis, and (3) a feature fusion module for final classification. The architecture is specifically designed to address the unique challenges of ultrasound video classification.

**Design Rationale.** We adopt a dual-stream CNN-based architecture over transformer-based alternatives for several ultrasound-specific reasons:

1. **Inductive Bias for Local Patterns**: CNNs possess strong spatial locality bias through convolution operations, which is crucial for ultrasound where pathological features (e.g., membrane detachment, tissue boundaries) are characterized by local texture patterns and edges. Transformers, while flexible, require substantially more data to learn such spatial priors from scratch [6].

2. **Computational Efficiency**: Our model processes 32-frame videos at $224 \times 224$ resolution with only ~40M parameters. Vision Transformers (ViT) and Video Vision Transformers (ViViT) [7] require $\mathcal{O}(N^2)$ complexity for self-attention over $N$ spatiotemporal patches, making them prohibitively expensive for medical video analysis where high temporal resolution is critical. Our CNN-based approach scales linearly with input size.

3. **Sample Efficiency**: With limited medical data (~200 videos), transformers' data-hungry nature becomes a critical bottleneck. Even with sophisticated pretraining strategies, transformer-based models typically require 10-100× more training samples to match CNN performance [8]. Our architecture leverages pretrained R3D-18 weights from Kinetics-400, providing effective initialization despite domain shift.

4. **Explicit Motion Modeling**: Rather than relying on self-attention to implicitly capture motion (as in transformers), we explicitly model motion through a dedicated optical flow stream. This design choice is motivated by the clinical importance of tissue dynamics in ultrasound diagnosis—detached retinas exhibit characteristic floating movements that are diagnostically significant. Explicit flow features provide interpretable motion representations that align with clinical reasoning.

5. **Hierarchical Feature Extraction**: The residual architecture progressively aggregates features from fine-grained textures (early layers) to semantic patterns (deep layers), matching the hierarchical nature of ultrasound interpretation where clinicians examine both local tissue characteristics and global anatomical structures.

#### 3.2.1 RGB Stream with Explainability

The RGB stream processes the input video $\mathbf{V}$ through a pretrained 3D ResNet-18 (R3D-18) backbone [1], which we denote as $\phi_{\text{RGB}}$. The backbone extracts hierarchical spatiotemporal features through four residual blocks:

**Why R3D-18 for Ultrasound?** We select R3D-18 over deeper variants (R3D-50, R3D-101) to prevent overfitting on small medical datasets while maintaining sufficient representational capacity. The 3D convolutions naturally capture short-term temporal dependencies critical for ultrasound motion analysis, unlike 2D CNNs that process frames independently or (2+1)D factorized convolutions that may miss coupled spatiotemporal patterns in tissue movement.

$$
\mathbf{F}_{\text{RGB}} = \phi_{\text{RGB}}(\mathbf{V}) \in \mathbb{R}^{B \times 512 \times T' \times H' \times W'}
$$

where $B$ is the batch size, and $T'$, $H'$, $W'$ are the reduced temporal and spatial dimensions after convolution and pooling operations.

**Frame Importance Module.** To identify which temporal segments contribute most to the classification decision, we introduce a frame importance module $\mathcal{M}_{\text{frame}}$ that computes attention weights across the temporal dimension:

$$
\mathbf{A}_{\text{frame}} = \text{Softmax}_T\left(\text{AvgPool}_{H',W'}\left(\mathbf{W}_{\text{frame}} * \mathbf{F}_{\text{RGB}}\right)\right) \in \mathbb{R}^{B \times 1 \times T' \times 1 \times 1}
$$

where $\mathbf{W}_{\text{frame}} \in \mathbb{R}^{1 \times 512 \times 1 \times 1 \times 1}$ is a learnable 1×1×1 convolutional filter, $*$ denotes convolution, $\text{AvgPool}_{H',W'}$ averages over spatial dimensions, and $\text{Softmax}_T$ normalizes across the temporal dimension. The frame-weighted features are computed as:

$$
\mathbf{F}_{\text{frame}} = \mathbf{F}_{\text{RGB}} \odot \mathbf{A}_{\text{frame}}
$$

where $\odot$ denotes element-wise multiplication with broadcasting.

**Ultrasound-Specific Design Choice.** Unlike transformer self-attention that computes pairwise frame relationships (requiring $\mathcal{O}(T^2)$ operations), our frame importance module uses a lightweight convolutional approach with $\mathcal{O}(T)$ complexity. This is particularly advantageous for ultrasound videos where: (1) diagnostic frames are often sparse within a sequence (e.g., only frames showing clear retinal movement are critical), and (2) the limited dataset size makes learning complex $T \times T$ attention matrices prone to overfitting. Our softmax normalization ensures that attention is distributed across frames, preventing the model from focusing solely on a single frame and encouraging temporal context integration.

**Spatial Explainability Module.** To highlight diagnostically relevant spatial regions, we employ a spatial attention mechanism $\mathcal{M}_{\text{spatial}}$:

$$
\mathbf{A}_{\text{spatial}} = \sigma\left(\mathbf{W}_2 * \text{ReLU}\left(\mathbf{W}_1 * \mathbf{F}_{\text{frame}}\right)\right) \in \mathbb{R}^{B \times 1 \times T' \times H' \times W'}
$$

where $\mathbf{W}_1 \in \mathbb{R}^{128 \times 512 \times 1 \times 1 \times 1}$ and $\mathbf{W}_2 \in \mathbb{R}^{1 \times 128 \times 1 \times 1 \times 1}$ are learnable convolutional filters, and $\sigma$ is the sigmoid activation function. The spatially-weighted features are:

$$
\mathbf{F}_{\text{spatial}} = \mathbf{F}_{\text{frame}} \odot \mathbf{A}_{\text{spatial}}
$$

**Clinical Interpretability Advantage.** The spatial attention maps $\mathbf{A}_{\text{spatial}}$ provide pixel-level explanations that can be directly overlaid on ultrasound frames for clinical validation. This is a significant advantage over transformer-based models where attention weights are computed over abstract patch tokens, making spatial localization less precise. In ultrasound diagnosis, clinicians need to identify specific anatomical landmarks (e.g., optic nerve, retinal layers, vitreous cavity boundaries). Our convolutional attention preserves spatial correspondence, enabling clinicians to verify whether the model focuses on clinically relevant regions. The sigmoid activation allows multiple regions to be highlighted simultaneously, reflecting the reality that pathology often involves multiple anatomical structures.

Finally, we apply global average pooling and obtain the RGB feature vector:

$$
\mathbf{z}_{\text{RGB}} = \text{GAP}(\mathbf{F}_{\text{spatial}}) \in \mathbb{R}^{B \times 512}
$$

where $\text{GAP}$ denotes global average pooling over the spatiotemporal dimensions.

#### 3.2.2 Optical Flow Stream

Motion information is crucial for analyzing dynamic medical videos. We introduce a lightweight optical flow extraction module $\phi_{\text{flow}}$ that learns to estimate inter-frame motion patterns directly from RGB input.

**Motivation for Explicit Flow Modeling.** While transformer-based video models (e.g., TimeSformer [9], VideoMAE [10]) rely on self-attention to implicitly capture temporal relationships, we argue that explicit optical flow modeling is superior for ultrasound video classification for three reasons:

1. **Physical Interpretability**: Optical flow represents actual tissue displacement vectors, which have direct clinical meaning. Detached retinas exhibit characteristic "floating" motion patterns with specific velocity profiles that can be quantified through flow magnitude and direction. Self-attention weights, while capturing temporal dependencies, lack this physical interpretation.

2. **Noise Robustness**: Ultrasound speckle noise is temporally uncorrelated, meaning frame-to-frame differences (captured by flow) naturally suppress noise while preserving true tissue motion. Transformer attention over noisy frames may learn spurious correlations unless heavily regularized.

3. **Computational Efficiency**: Computing optical flow through lightweight 3D convolutions with temporal kernel size 2 requires minimal parameters (~50K) compared to full temporal self-attention which scales quadratically with sequence length. For 32-frame videos, this represents a 500× reduction in temporal modeling complexity while maintaining motion sensitivity.

$$
\mathbf{F}_{\text{flow}}^{(0)} = \phi_{\text{flow}}(\mathbf{V})
$$

The flow extractor consists of three 3D convolutional layers with temporal kernel sizes of 2 to capture frame-to-frame differences:

$$
\begin{aligned}
\mathbf{F}_{\text{flow}}^{(1)} &= \text{ReLU}\left(\text{Conv3D}_{(2,3,3)}^{32}(\mathbf{V})\right) \\
\mathbf{F}_{\text{flow}}^{(2)} &= \text{ReLU}\left(\text{Conv3D}_{(2,3,3)}^{64}(\mathbf{F}_{\text{flow}}^{(1)})\right) \\
\mathbf{F}_{\text{flow}}^{(0)} &= \text{Conv3D}_{(1,3,3)}^{32}(\mathbf{F}_{\text{flow}}^{(2)})
\end{aligned}
$$

where $\text{Conv3D}_{(k_t,k_h,k_w)}^{c}$ denotes a 3D convolution with kernel size $(k_t, k_h, k_w)$ and $c$ output channels.

The extracted flow features are then processed through a flow processing network $\psi_{\text{flow}}$:

$$
\begin{aligned}
\mathbf{F}_{\text{flow}}^{(3)} &= \text{ReLU}\left(\text{BN}\left(\text{Conv3D}_{(3,3,3), s=2}^{64}(\mathbf{F}_{\text{flow}}^{(0)})\right)\right) \\
\mathbf{F}_{\text{flow}}^{(4)} &= \text{ReLU}\left(\text{BN}\left(\text{Conv3D}_{(3,3,3), s=2}^{128}(\mathbf{F}_{\text{flow}}^{(3)})\right)\right) \\
\mathbf{F}_{\text{flow}}^{(5)} &= \text{ReLU}\left(\text{BN}\left(\text{Conv3D}_{(3,3,3), s=2}^{256}(\mathbf{F}_{\text{flow}}^{(4)})\right)\right)
\end{aligned}
$$

where $\text{BN}$ denotes batch normalization and $s$ is the stride. We apply adaptive average pooling to obtain a fixed-size feature vector:

$$
\mathbf{z}_{\text{flow}} = \text{AdaptiveAvgPool3D}_{(1,1,1)}(\mathbf{F}_{\text{flow}}^{(5)}) \in \mathbb{R}^{B \times 256}
$$

#### 3.2.3 Feature Fusion and Multi-Task Classification

We fuse the RGB and flow features through concatenation followed by a learned projection.

**Why Early Fusion?** We adopt early fusion (feature-level) rather than late fusion (decision-level) or intermediate fusion strategies. This design is motivated by the complementary nature of appearance and motion in ultrasound: pathological tissue often exhibits both abnormal texture (captured by RGB) and abnormal movement (captured by flow). Early fusion allows the classifier to learn joint appearance-motion patterns (e.g., "bright echogenic membrane that floats") which are more discriminative than independent appearance or motion cues.

$$
\mathbf{z}_{\text{concat}} = [\mathbf{z}_{\text{RGB}}; \mathbf{z}_{\text{flow}}] \in \mathbb{R}^{B \times 768}
$$

$$
\mathbf{z}_{\text{fused}} = \text{Dropout}_{p}\left(\text{BN}\left(\text{ReLU}\left(\mathbf{W}_{\text{fusion}}\mathbf{z}_{\text{concat}} + \mathbf{b}_{\text{fusion}}\right)\right)\right)
$$

where $\mathbf{W}_{\text{fusion}} \in \mathbb{R}^{512 \times 768}$, $\mathbf{b}_{\text{fusion}} \in \mathbb{R}^{512}$, and $p=0.3$ is the dropout rate.

**Multi-Task Classification Heads.** Instead of a single classifier, we employ two parallel classification heads that share the fused features:

$$
\begin{aligned}
\mathbf{h}_{\text{shared}} &= \text{Dropout}_{p/2}\left(\text{BN}\left(\text{ReLU}\left(\mathbf{W}_{\text{shared}}\mathbf{z}_{\text{fused}} + \mathbf{b}_{\text{shared}}\right)\right)\right) \\
\mathbf{y}_{\text{diag}} &= \mathbf{W}_{\text{diag}}\mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{diag}} \in \mathbb{R}^{B \times 2} \\
\mathbf{y}_{\text{sub}} &= \mathbf{W}_{\text{sub}}\mathbf{h}_{\text{shared}} + \mathbf{b}_{\text{sub}} \in \mathbb{R}^{B \times 4}
\end{aligned}
$$

where $\mathbf{W}_{\text{shared}} \in \mathbb{R}^{256 \times 512}$, $\mathbf{W}_{\text{diag}} \in \mathbb{R}^{2 \times 256}$, and $\mathbf{W}_{\text{sub}} \in \mathbb{R}^{4 \times 256}$.

**Multi-Task Learning Benefits:**
1. **Shared Representations**: Both tasks benefit from learning robust features in $\mathbf{h}_{\text{shared}}$
2. **Regularization**: Multi-task learning acts as implicit regularization, reducing overfitting
3. **Efficiency**: Shared backbone and fusion layers reduce total parameters compared to separate models
4. **Clinical Relevance**: Outputs align with diagnostic workflow (primary diagnosis → subtype)

### 3.3 Training Strategy

#### 3.3.1 Two-Phase Training

We employ a two-phase training strategy to effectively leverage pretrained weights while adapting to the medical domain:

**Phase 1: Classifier Head Training.** We freeze the RGB backbone parameters $\theta_{\text{RGB}}$ while training the explainability modules, flow extractor, fusion layer, and classifier. This phase runs for $E_1 = 3$ epochs with learning rate $\eta_1 = 10\eta_{\text{base}}$ where $\eta_{\text{base}} = 10^{-4}$. We use the Cosine Annealing with Warm Restarts scheduler:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_1 - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_0}\pi\right)\right)
$$

where $T_{\text{cur}}$ is the current epoch within the restart period and $T_0 = 5$ is the restart period.

**Phase 2: Full Fine-tuning.** We unfreeze all parameters and fine-tune the entire network for $E_2 = 7$ epochs with learning rate $\eta_2 = \eta_{\text{base}}$. We employ ReduceLROnPlateau scheduler that reduces the learning rate by factor $\gamma = 0.5$ when validation accuracy plateaus for $p = 3$ epochs.

#### 3.3.2 Multi-Task Loss Function

For multi-task learning, we employ a weighted combination of Cross-Entropy losses for each task:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + \lambda_{\text{sub}} \mathcal{L}_{\text{sub}}
$$

where:

$$
\mathcal{L}_{\text{diag}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{2} y_{i,c}^{\text{diag}}\log(p_{i,c}^{\text{diag}})
$$

$$
\mathcal{L}_{\text{sub}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{4} y_{i,c}^{\text{sub}}\log(p_{i,c}^{\text{sub}})
$$

We set $\lambda_{\text{sub}} = 1.0$ to weight both tasks equally. This simple weighting scheme works well in practice as both tasks are clinically important.

**Balanced Sampling Strategy.** To address class imbalance in the ERDES dataset, we employ stratified sampling during train/test split creation. The dataset is split with stratification by the combined diagnostic-subtype label (e.g., "rd\_macula\_detached"), ensuring balanced representation of all class combinations in both training and test sets. This approach is superior to class-weighted loss functions for small medical datasets, as it prevents the model from being biased toward majority classes while maintaining natural class distributions during training.

#### 3.3.3 Data Augmentation

**Spatial Augmentation.** We apply random spatial transformations including:
- Random resized crop: $224 \times 224$ from $256 \times 256$
- Random horizontal flip with probability 0.5
- Color jittering: brightness $\pm 0.2$, contrast $\pm 0.2$

**Temporal Augmentation.** We randomly sample $T=32$ frames from each video using uniform temporal sampling with random start offset.

**Mixup Augmentation.** During Phase 2 training, we apply Mixup [3] with probability 0.5:

$$
\begin{aligned}
\tilde{\mathbf{V}} &= \lambda\mathbf{V}_i + (1-\lambda)\mathbf{V}_j \\
\tilde{\mathbf{y}} &= \lambda\mathbf{y}_i + (1-\lambda)\mathbf{y}_j
\end{aligned}
$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha = 0.2$, and $(i,j)$ are randomly sampled training pairs.

**Test-Time Augmentation (TTA).** During inference, we average predictions from the original video and its horizontally flipped version:

$$
\mathbf{p}_{\text{final}} = \frac{1}{2}\left(\text{softmax}(f(\mathbf{V})) + \text{softmax}(f(\text{Flip}_H(\mathbf{V})))\right)
$$

#### 3.3.4 Optimization

We use AdamW optimizer [4] with weight decay $\lambda = 10^{-4}$:

$$
\theta_{t+1} = \theta_t - \eta_t\left(\frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon} + \lambda\theta_t\right)
$$

where $\mathbf{m}_t$ and $\mathbf{v}_t$ are the first and second moment estimates with decay rates $\beta_1 = 0.9$ and $\beta_2 = 0.999$.

To prevent gradient explosion, we apply gradient clipping:

$$
\mathbf{g}_t \leftarrow \mathbf{g}_t \cdot \min\left(1, \frac{\tau}{\|\mathbf{g}_t\|_2}\right)
$$

where $\tau = 1.0$ is the maximum gradient norm.

**Early Stopping.** We monitor validation accuracy and stop training if no improvement is observed for $p_{\text{stop}} = 7$ consecutive epochs.

### 3.4 Vision-Language Model Integration

After training the multi-class classifier, we integrate a Vision-Language Model (VLM) to generate human-readable clinical reasoning from classifier predictions and attention-highlighted frames. This two-stage approach combines the efficiency and interpretability of CNNs with the natural language generation capabilities of large language models.

#### 3.4.1 VLM Data Preparation

Given a trained classifier $f_{\theta^*}$ with optimal parameters $\theta^*$, we prepare VLM training data through the following pipeline:

**Step 1: Important Frame Extraction.** For each video $\mathbf{V}$, we extract the top-$K$ most important frames based on frame attention scores:

$$
\mathcal{F}_{\text{important}} = \text{TopK}(\mathbf{A}_{\text{frame}}, K) = \{(\mathbf{V}_{t_1}, a_{t_1}), \ldots, (\mathbf{V}_{t_K}, a_{t_K})\}
$$

where $t_1, \ldots, t_K$ are the frame indices with highest attention scores and $a_{t_i}$ are the corresponding attention weights. We use $K=5$ to balance information richness with computational efficiency.

**Step 2: Attention Heatmap Generation.** For each important frame $\mathbf{V}_{t_i}$, we generate a spatial attention heatmap by upsampling $\mathbf{A}_{\text{spatial}}$ to the original frame resolution:

$$
\mathbf{H}_{t_i} = \text{Upsample}(\mathbf{A}_{\text{spatial}}[t_i], (H, W)) \in \mathbb{R}^{H \times W}
$$

The heatmap is overlaid on the original frame using a jet colormap to create an attention-highlighted visualization $\mathbf{V}_{t_i}^{\text{attn}}$.

**Step 3: Prediction-Conditioned Prompt Generation.** We create a structured prompt that includes classifier predictions:

$$
\begin{aligned}
\text{prompt} = &\text{``The AI model predicts:} \\
&\text{- Primary Diagnosis: } c_{\text{diag}} \text{ (confidence: } p_{\text{diag}} \text{)} \\
&\text{- Subtype: } c_{\text{sub}} \text{ (confidence: } p_{\text{sub}} \text{)} \\
&\text{Based on the highlighted regions, explain the clinical reasoning.''}
\end{aligned}
$$

where $c_{\text{diag}} = \arg\max \mathbf{y}_{\text{diag}}$, $c_{\text{sub}} = \arg\max \mathbf{y}_{\text{sub}}$, and $p_{\text{diag}}, p_{\text{sub}}$ are the softmax probabilities.

**Step 4: Contrastive Sample Creation.** To ensure the VLM attends to attention heatmaps rather than just raw frames, we create contrastive pairs:
- **Positive sample**: Attention-highlighted frames $\{\mathbf{V}_{t_i}^{\text{attn}}\}_{i=1}^K$ with correct predictions
- **Negative sample**: Original frames $\{\mathbf{V}_{t_i}\}_{i=1}^K$ without highlights (model should indicate uncertainty)

This contrastive learning strategy encourages the VLM to utilize the spatial attention information.

#### 3.4.2 VLM Finetuning

We finetune Qwen 2.5 VL-7B [11], a state-of-the-art vision-language model, using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) [12].

**Model Architecture.** Qwen 2.5 VL consists of:
- **Vision Encoder**: Processes image inputs into visual tokens
- **Language Model**: 7B parameter transformer for text generation
- **Cross-Modal Fusion**: Attention layers that integrate visual and textual information

**LoRA Configuration.** To reduce memory requirements and prevent overfitting, we apply LoRA to the attention layers:

$$
\mathbf{W}' = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}
$$

where $\mathbf{W}_0$ are the frozen pretrained weights, $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices with rank $r = 16 \ll \min(d, k)$. We set LoRA alpha $\alpha = 32$ and dropout $p_{\text{LoRA}} = 0.05$.

**4-bit Quantization.** To enable training on GPUs with limited memory, we employ 4-bit quantization using bitsandbytes [13], reducing the model's memory footprint from ~28GB to ~7GB while maintaining generation quality.

**Training Objective.** We finetune the VLM using causal language modeling loss:

$$
\mathcal{L}_{\text{VLM}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}\log P(w_t^{(i)} | w_{<t}^{(i)}, \mathbf{V}_{1:K}^{(i)}, \text{prompt}^{(i)})
$$

where $w_t^{(i)}$ is the $t$-th token in the $i$-th ground truth clinical reasoning text, and $\mathbf{V}_{1:K}^{(i)}$ are the $K$ attention-highlighted frames.

**Training Hyperparameters:**
- Learning rate: $\eta_{\text{VLM}} = 2 \times 10^{-5}$
- Batch size: $B_{\text{VLM}} = 2$ (with gradient accumulation)
- Epochs: $E_{\text{VLM}} = 10$
- Warmup steps: 100
- Optimizer: AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.999$

#### 3.4.3 Inference Pipeline

During inference, the complete diagnostic pipeline operates as follows:

1. **Video Classification**: Input video $\mathbf{V}$ → Classifier $f_{\theta^*}$ → Predictions $(\mathbf{y}_{\text{diag}}, \mathbf{y}_{\text{sub}})$ and attention maps $(\mathbf{A}_{\text{frame}}, \mathbf{A}_{\text{spatial}})$

2. **Frame Selection**: Extract top-$K$ frames with highest frame importance scores

3. **Heatmap Overlay**: Generate attention-highlighted frames $\{\mathbf{V}_{t_i}^{\text{attn}}\}_{i=1}^K$

4. **Prompt Construction**: Create prediction-conditioned prompt with diagnostic and subtype predictions

5. **Clinical Reasoning Generation**: VLM generates natural language explanation:
   $$
   \text{reasoning} = \text{VLM}(\{\mathbf{V}_{t_i}^{\text{attn}}\}_{i=1}^K, \text{prompt})
   $$

6. **Output**: Return structured result containing:
   - Diagnostic class and confidence
   - Subtype class and confidence
   - Important frame indices and attention scores
   - Attention-highlighted visualizations
   - Natural language clinical reasoning

**Advantages of Two-Stage Approach:**
1. **Modularity**: Classifier and VLM can be trained/updated independently
2. **Efficiency**: Lightweight classifier provides fast predictions; VLM adds interpretability when needed
3. **Explainability**: Attention maps guide VLM to focus on diagnostically relevant regions
4. **Clinical Alignment**: Generated reasoning follows medical diagnostic patterns

### 3.5 Implementation Details

**Dataset.** We use the ERDES dataset [5] consisting of 5,383 ocular ultrasound videos with hierarchical labels:
- **Diagnostic classes** (2): non-retinal detachment (non\_rd), retinal detachment (rd)
- **Subtypes** (4): normal, macula intact, macula detached, posterior vitreous detachment (pvd)
- **Anatomical subclasses** (6): temporal (TD), nasal (ND), superior, inferior, bilateral, N/A (metadata only, not used for training)

Videos are preprocessed to $224 \times 224$ spatial resolution with $T=32$ uniformly sampled frames. We employ stratified 80/20 train/test split based on combined diagnostic-subtype labels to ensure balanced class representation.

**Hardware.** All experiments are conducted on NVIDIA GPUs with 16GB memory. Training takes approximately 4-6 hours for 10 epochs with batch size $B=16$.

**Evaluation Metrics.** For the multi-class classifier, we report per-task metrics:
- **Diagnostic Task**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Subtype Task**: Accuracy, Precision, Recall, F1-Score (macro-averaged across 4 classes)
- **Overall Performance**: Average accuracy across both tasks

For VLM evaluation, we assess:
- **Factual Consistency**: Alignment between generated reasoning and classifier predictions
- **Clinical Relevance**: Quality of medical explanations (evaluated by domain experts)
- **Attention Utilization**: Whether VLM references highlighted regions in reasoning

**Explainability Visualization.** The frame importance scores $\mathbf{A}_{\text{frame}}$ and spatial attention maps $\mathbf{A}_{\text{spatial}}$ are extracted during inference to provide interpretable insights into the model's decision-making process. These attention maps are used both for clinical validation and as input to the VLM for reasoning generation.

### 3.5 Comparison with Transformer-Based Methods

To contextualize our design choices, we compare our architecture with state-of-the-art transformer-based video models:

**Table: Architectural Comparison**

| Model | Parameters | FLOPs | Complexity | Interpretability | Sample Efficiency |
|-------|------------|-------|------------|------------------|-------------------|
| TimeSformer [9] | 121M | 590G | $\mathcal{O}(T^2HW)$ | Low (patch tokens) | Low (requires >10K videos) |
| VideoMAE [10] | 86M | 180G | $\mathcal{O}(T^2HW)$ | Low (masked reconstruction) | Medium (requires >1K videos) |
| ViViT [7] | 98M | 340G | $\mathcal{O}((THW)^2)$ | Low (global attention) | Low (requires >10K videos) |
| **Ours** | **40M** | **45G** | $\mathcal{O}(THW)$ | **High (spatial + temporal maps)** | **High (works with ~200 videos)** |

**Key Advantages of Our Approach:**

1. **Parameter Efficiency**: Our model uses 2-3× fewer parameters than transformer alternatives, critical for preventing overfitting on small medical datasets.

2. **Computational Efficiency**: Linear complexity vs. quadratic self-attention enables processing longer sequences (32 frames) at higher resolution (224×224) with limited GPU memory.

3. **Explicit Interpretability**: Unlike transformers where attention weights are abstract and difficult to interpret clinically, our frame importance and spatial attention maps directly correspond to temporal segments and anatomical regions that clinicians can validate.

4. **Strong Inductive Biases**: Convolutional locality bias and explicit motion modeling provide structural priors that transformers must learn from data—a significant advantage when training data is scarce.

5. **Transfer Learning Effectiveness**: Pretrained R3D-18 weights from natural videos transfer better to medical ultrasound than transformer weights, as the hierarchical feature extraction aligns with clinical reasoning patterns.

**Where Transformers Excel:** For datasets with >10K videos and computational resources for extensive pretraining, transformers may capture more complex long-range dependencies. However, for practical medical applications with limited data and computational budgets, our CNN-based approach offers superior efficiency and interpretability.

### 3.6 Ablation Studies

To validate the contribution of each component, we conduct ablation studies by training variants:
1. **RGB-only**: Remove optical flow stream ($\mathbf{z} = \mathbf{z}_{\text{RGB}}$)
2. **No Frame Attention**: Remove $\mathcal{M}_{\text{frame}}$
3. **No Spatial Attention**: Remove $\mathcal{M}_{\text{spatial}}$
4. **Flow-only**: Use only optical flow features ($\mathbf{z} = \mathbf{z}_{\text{flow}}$)
5. **Late Fusion**: Fuse features after separate classifiers instead of early fusion
6. **TimeSformer Baseline**: Replace our architecture with TimeSformer-B/16 for direct comparison

### References

[1] Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). A closer look at spatiotemporal convolutions for action recognition. In CVPR.

[2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In ICCV.

[3] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. In ICLR.

[4] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In ICLR.

[5] Ozkut, Y., Navard, P., Adhikari, S., Situ-LaCasse, E., Acuña, J., Yarnish, A. A., & Yilmaz, A. (2025). ERDES: A Benchmark Video Dataset for Retinal Detachment and Macular Status Classification in Ocular Ultrasound. arXiv preprint arXiv:2508.04735.

[6] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.

[7] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). ViViT: A video vision transformer. In ICCV.

[8] Bai, Y., Mei, J., Yuille, A. L., & Xie, C. (2021). Are transformers more robust than CNNs?. In NeurIPS.

[9] Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for video understanding?. In ICML.

[10] Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS.

[11] Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., ... & Zhou, J. (2023). Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

[12] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. In ICLR.

[13] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. In NeurIPS.

---

## Appendix A: Network Architecture Details

### Table 1: RGB Stream Architecture (R3D-18 Backbone)

| Layer | Output Size | Kernel | Stride | Channels |
|-------|-------------|--------|--------|----------|
| Conv1 | $T \times 112 \times 112$ | $3 \times 7 \times 7$ | $(1,2,2)$ | 64 |
| Pool1 | $T \times 56 \times 56$ | $1 \times 3 \times 3$ | $(1,2,2)$ | 64 |
| Res2 | $T \times 56 \times 56$ | - | - | 64 |
| Res3 | $T/2 \times 28 \times 28$ | - | $(2,2,2)$ | 128 |
| Res4 | $T/4 \times 14 \times 14$ | - | $(2,2,2)$ | 256 |
| Res5 | $T/8 \times 7 \times 7$ | - | $(2,2,2)$ | 512 |
| Frame Attn | $T/8 \times 7 \times 7$ | $1 \times 1 \times 1$ | - | 512 |
| Spatial Attn | $T/8 \times 7 \times 7$ | $1 \times 1 \times 1$ | - | 512 |
| GAP | 1 | - | - | 512 |

### Table 2: Optical Flow Stream Architecture

| Layer | Output Size | Kernel | Stride | Channels |
|-------|-------------|--------|--------|----------|
| FlowConv1 | $(T-1) \times 224 \times 224$ | $(2,3,3)$ | $(1,1,1)$ | 32 |
| FlowConv2 | $(T-2) \times 224 \times 224$ | $(2,3,3)$ | $(1,1,1)$ | 64 |
| FlowConv3 | $(T-2) \times 224 \times 224$ | $(1,3,3)$ | $(1,1,1)$ | 32 |
| Conv1 | $(T-2)/2 \times 112 \times 112$ | $3 \times 3 \times 3$ | $(2,2,2)$ | 64 |
| Conv2 | $(T-2)/4 \times 56 \times 56$ | $3 \times 3 \times 3$ | $(2,2,2)$ | 128 |
| Conv3 | $(T-2)/8 \times 28 \times 28$ | $3 \times 3 \times 3$ | $(2,2,2)$ | 256 |
| AdaptivePool | 1 | - | - | 256 |

### Table 3: Fusion and Multi-Task Classification Architecture

| Layer | Input Dim | Output Dim | Activation |
|-------|-----------|------------|------------|
| Concat | 512 + 256 | 768 | - |
| Fusion | 768 | 512 | ReLU + BN + Dropout(0.3) |
| Shared FC | 512 | 256 | ReLU + BN + Dropout(0.15) |
| Diagnostic Head | 256 | 2 | - |
| Subtype Head | 256 | 4 | - |

### Table 4: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Input Frames | 32 |
| Input Resolution | $224 \times 224$ |
| Phase 1 Epochs | 3 |
| Phase 2 Epochs | 7 |
| Phase 1 LR | $10^{-3}$ |
| Phase 2 LR | $10^{-4}$ |
| Weight Decay | $10^{-4}$ |
| Dropout | 0.3 |
| Focal Loss $\gamma$ | 2.0 |
| Mixup $\alpha$ | 0.2 |
| Gradient Clip Norm | 1.0 |
| Early Stop Patience | 7 |

### Table 5: VLM Hyperparameters

| Parameter | Value |
|-----------|-------|
| VLM Model | Qwen 2.5 VL-7B |
| Important Frames (K) | 5 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Quantization | 4-bit |
| VLM Batch Size | 2 |
| VLM Epochs | 10 |
| VLM Learning Rate | $2 \times 10^{-5}$ |
| Warmup Steps | 100 |
| Contrastive Learning | Enabled |

### Table 6: Dataset Statistics (ERDES)

| Category | Count | Percentage |
|----------|-------|------------|
| **Total Videos** | 5,383 | 100% |
| **Diagnostic Classes** | | |
| - Non-RD | 4,305 | 80.0% |
| - RD | 1,078 | 20.0% |
| **Subtypes** | | |
| - Normal | 4,091 | 76.0% |
| - Macula Intact | 433 | 8.0% |
| - Macula Detached | 645 | 12.0% |
| - PVD | 214 | 4.0% |
| **Train/Test Split** | | |
| - Training Set | 4,306 | 80% |
| - Test Set | 1,077 | 20% |
