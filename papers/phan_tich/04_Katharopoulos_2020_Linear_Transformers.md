# Phân Tích Paper: Transformers are RNNs

## Fast Autoregressive Transformers with Linear Attention

**Thông tin:**  
- **arXiv:** [2006.16236](https://arxiv.org/abs/2006.16236) (ICML 2020)  
- **Tác giả:** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret (EPFL, Idiap)  
- **File PDF:** `2006.16236_Katharopoulos_Linear_Transformers.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Standard self-attention có độ phức tạp $O(N^2)$ — không khả thi cho sequences dài. Các giải pháp sparse attention (Child et al.) không tăng tốc autoregressive inference.

**Mục tiêu:** Giảm complexity xuống $O(N)$ cho cả training và inference, đặc biệt cho autoregressive prediction.

**Key insight:** Sử dụng associativity của matrix products với kernel feature maps.

---

## 2. Toán Học Cốt Lõi

### 2.1 Kernel Formulation của Self-Attention

Standard attention:

$$V' = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

Viết dưới dạng kernel:

$$V_i' = \frac{\sum_{j=1}^N \text{sim}(Q_i, K_j) V_j}{\sum_{j=1}^N \text{sim}(Q_i, K_j)}$$

Với $\text{sim}(q, k) = \exp(q^\top k / \sqrt{d})$ cho softmax attention.

### 2.2 Linear Attention Formulation

Chọn feature map $\phi: \mathbb{R}^d \to \mathbb{R}^m$ sao cho:
$$\phi(Q_i)^\top \phi(K_j) \approx \exp(Q_i^\top K_j / \sqrt{d})$$

Khi đó:

$$V_i' = \frac{\sum_{j=1}^N \phi(Q_i)^\top \phi(K_j) V_j}{\sum_{j=1}^N \phi(Q_i)^\top \phi(K_j)}$$

Sử dụng associativity:

$$V' = \phi(Q) \underbrace{(\phi(K)^\top V)}_{O(m \cdot d) \; \text{tính trước}}$$

**Complexity:** $O(N \cdot m)$ thay vì $O(N^2)$.

### 2.3 Causal Masking và RNN Equivalence

Với causal attention:

$$S_i = S_{i-1} + \phi(K_i) \otimes V_i \quad \in \mathbb{R}^{m \times d_v}$$
$$Z_i = Z_{i-1} + \phi(K_i) \quad \in \mathbb{R}^m$$
$$V_i' = \frac{\phi(Q_i) S_i}{\phi(Q_i) Z_i}$$

**Đây chính là RNN!** Với:
- Hidden state: $S_i$ và $Z_i$ (cumulative sums)
- Update: additive (không phải RNN gate)
- Inference: $O(1)$ per token

---

## 3. Feature Map: ELU+1

Trong paper sử dụng:
$$\phi(x) = \text{ELU}(x) + 1$$

Với:
$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ e^x - 1 & \text{if } x \leq 0 \end{cases}$$

**Tính chất:**
- $\phi(x) > 0$: đảm bảo non-negativeness
- $\phi(x) \approx \text{ReLU}(x) + 1$: không có dead neurons
- **Nhược điểm:** Không approximation chính xác của softmax kernel

---

## 4. Thí Nghiệm

### 4.1 Tasks

- **Image generation:** CIFAR-10 (pixel-level autoregressive)
- **Automatic speech recognition:** WSJ, LibriSpeech
- **So sánh:** Transformer vs Linear Transformer

### 4.2 Kết Quả

| Task | Transformer | Linear Transformer | Speedup |
|------|-------------|-------------------|---------|
| Image gen (CIFAR-10) | 2.95 BPD | 2.97 BPD | ~4000x (inference) |
| WSJ | 7.3 PER | 7.8 PER | ~10x |
| LibriSpeech | — | Comparable | ~4000x |

**Key finding:** Performance tương đương với vanilla transformer nhưng nhanh hơn **đến 4000x** cho autoregressive inference trên sequences rất dài.

---

## 5. Liên Quan Đến INLA

### 5.1 Nền Tảng Cho INLA

Paper này là **nền tảng kỹ thuật cốt lõi** cho INLA:

| Khía cạnh | Linear Transformer | INLA |
|-----------|-------------------|------|
| **Cốt lõi** | Kernel attention với ELU+1 | Kernel attention + normalization |
| **Complexity** | $O(N)$ | $O(N)$ |
| **Feature map** | ELU+1 (heuristic) | Information-theoretic optimized |
| **Performance** | Gần softmax | Target: bằng hoặc hơn softmax |

### 5.2 Cải Thiện Cho INLA

1. **Feature map:** Thay thế ELU+1 bằng feature map tốt hơn
2. **Normalization:** Thêm information-theoretic normalization
3. **Re-weighting:** Kết hợp với cơ chế re-weighting (như CosFormer)
4. **Rank preservation:** Đảm bảo effective rank không giảm

---

## 6. Thách Thức

1. **ELU+1 không optimal:** Kernel approximation có sai số lớn trên nhiều tasks
2. **Performance gap:** Linear attention thường thấp hơn softmax attention
3. **Feature map design:** Chưa có principled method để chọn $\phi$
4. **Numerical stability:** Cumulative sum có thể overflow với sequences dài
5. **Transformer equivalence:** Chưa khai thác hết RNN perspective

---

## 7. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | Kernel formulation + RNN equivalence |
| **Toán học** | Associative property: $\phi(Q)(\phi(K)^\top V)$ |
| **Complexity** | $O(N)$ training và $O(1)$ inference per token |
| **Speedup** | Up to 4000x cho sequences rất dài |
| **Với INLA** | Nền tảng kỹ thuật cốt lõi |

---

*Phân tích cho dự án INLA — 06/2026*
