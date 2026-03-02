# GeoLM: Geometry-Aware Language Model Specification

**Author:** Jake Peace  
**Status:** Experimental / Active Implementation  
**Core Thesis:** Language modeling through pure geometry and variance-retaining trees, eliminating backpropagation in the core layers to produce a structurally honest, hallucination-resistant architecture.

---

## 1. Architecture Overview

GeoLM replaces the standard Transformer neural backbone (dense attention and learned FFN weights) with a gradient-free, iterative boosting sequence. It merges the piecewise-linear approximation findings of HVRT with the dynamic routing, scaling, and self-healing mechanics of GeoXGB.

The pipeline consists of four distinct stages:
1.  **Discrete-to-Continuous Front-End:** Standard tokenization and static embeddings.
2.  **Constructed Attention (Geometric Routing):** Dynamic context aggregation via HVRT partitions, Farthest Point Sampling (FPS), and Kernel Density Estimation (KDE).
3.  **The Boosting Backbone:** Iterative residual fitting replacing standard Feed-Forward Networks.
4.  **The Gardener (Inter-layer Regularization):** Dynamic tree surgery to prevent covariate shift cascades.

---

## 2. Component Specifications

### 2.1 The Front-End (Learned/Static)
*Because Layer 0 is uniquely non-linear, the initial discrete-to-continuous mapping remains traditional.*
* **Tokenization:** Standard Byte-Pair Encoding (BPE) or WordPiece.
* **Embedding Table:** A static lookup table mapping tokens to d_model dimensions.
* **Positional Encoding:** Rotary Position Embeddings (RoPE) applied to the token vectors. This injects "time" as a spatial dimension, allowing the downstream HVRT trees to use positional frequency as a native heterogeneity axis for splitting.

### 2.2 Constructed Attention (The Context Engine)
*Replaces W_q, W_k, W_v dense matrix multiplications with dynamic spatial routing.*
* **Query/Key Routing:** The context window is fed into an HVRT decision tree. Tokens that land in the same geometric partition automatically "attend" to each other based on shared local structure.
* **Context Eviction (FPS):** To prevent unbounded context scaling, historical tokens undergo Farthest Point Sampling (FPS). The model retains a geometrically diverse subset of context tokens rather than a dense N x N matrix.
* **Value Retrieval (KDE):** Instead of projecting past tokens through a learned matrix, the model applies Kernel Density Estimation around the current token's partition to blend the outputs of the relevant past sequence.

### 2.3 The Boosting Backbone (Layer Stack)
*Replaces standard transformer FFNs (W_1 GELU W_2) with piecewise-linear maps fitted analytically via OLS.*
* **Sequential Iteration:** Layers are fitted sequentially using iterative residual boosting. Layer L+1 is fitted strictly on the actual, corrected outputs (residuals) of Layer L, ensuring no tree ever trains on an unseen distribution.
* **Partition Constraints:** To prevent rare-token noise from degrading the fit, the vocabulary correction is restricted to the top-K tokens per partition. 

### 2.4 The Gardener (Cascade Regularization)
*A post-hoc/inter-layer surgical module acting as a dynamic safety net to prevent the covariate shift cascade.*
* **Target:** Operates on the outputs of Layer L before they pass to Layer L+1.
* **`heal()` Protocol:** Automatically detects partitions where the representation is drifting out of distribution. Adjusts leaf biases to pull the vector back toward the partition centroid.
* **`prune()` Protocol:** If a representation drifts into a statistically empty leaf (noise), the branch is severed, forcing the routing back to a dense, generalized partition.
* **`graft()` Protocol:** Triggers targeted boosting rounds specifically on highly dense token clusters that fail to approximate linearly, forcing depth only where evidence supports it.

### 2.5 Confidence & Output Generation
*The model is structurally incapable of false high-confidence.*
* **Density-Based Confidence:** The confidence score for any generated token is the literal integer count of training examples present in the final HVRT partition. 
* **Explicit Failure:** If an input representation falls into a leaf with zero or near-zero training examples, the model outputs an explicit "Out of Distribution" error rather than hallucinating.

---

## 3. Implementation Phasing

**Phase 1: The Narrow-Domain Prototype**
* Target a bounded vocabulary task (e.g., API code generation, medical ICD-10 coding).
* Vocabulary size: 500 - 2,000 tokens.
* Implement Front-End + Boosting Backbone. 
* *Goal:* Prove that sequential boosting with OLS solves the layer cascade.

**Phase 2: The Gardener Integration**
* Adapt GeoXGB's Gardener module to sit between the boosting layers.
* Implement `heal()` and `prune()` to track vector drift.
* *Goal:* Stabilize deep layer stacking without exponential variance loss.

**Phase 3: Constructed Attention**
* Implement RoPE embeddings.
* Build the FPS-based dynamic context eviction.
* *Goal:* Enable multi-token context dependencies without quadratic compute.