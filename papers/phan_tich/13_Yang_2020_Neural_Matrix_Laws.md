# Phân Tích Paper: Tensor Programs III: Neural Matrix Laws

**Thông tin:**  
- **arXiv:** [2009.10685](https://arxiv.org/abs/2009.10685) (NeurIPS 2020)  
- **Tác giả:** Greg Yang (Microsoft Research)  
- **File PDF:** `2009.10685_Yang_Neural_Matrix_Laws_RMT.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Random Matrix Theory (RMT) đã được dùng để phân tích neural networks (Jacobian spectrum, NTK). Tuy nhiên, thiếu một framework tổng quát.

**Mục tiêu:** Phát triển **Free Independence Principle (FIP)** — (pre-)activations và weights là asymptotically free ở infinite width limit.

---

## 2. Toán Học Cốt Lõi

### 2.1 Free Independence Principle (FIP)

**Theorem (informal):** Khi width → ∞:
1. (Pre-)activations và weights là **asymptotically free** trong sense của free probability
2. Jacobian singular value distribution có thể tính được
3. Gradient independence assumption cho NTK được justified

**Hệ quả:** Cho phép tính toán chính xác spectral distributions của:
- Weight matrices
- Hessian matrices
- Jacobian matrices

### 2.2 Marchenko–Pastur Law (Warmup)

Paper chứng minh lại MP law như một warmup cho Tensor Program framework:

$$\rho(\lambda) = \frac{1}{2\pi\lambda\sigma^2}\sqrt{(b_+ - \lambda)(\lambda - b_-)}$$

Với:
$$b_\pm = \sigma^2(1 \pm \sqrt{\alpha})^2, \quad \alpha = \frac{m}{n}$$

**MP law mô tả:** Empirical spectral distribution của $X^\top X$ với $X \in \mathbb{R}^{m \times n}$ có entries i.i.d.

### 2.3 Tensor Program Master Theorem

Framework tổng quát cho:
- **Input:** Tensor program (neural network computation graph)
- **Output:** Limiting distribution của tensors khi width → ∞
- **Công cụ:** Moment method, free probability, graphon theory

---

## 3. Ứng Dụng Cho Neural Networks

### 3.1 Jacobian Singular Value Distribution

Tính spectral distribution của Jacobian $\frac{\partial h^L}{\partial h^0}$:
- Quan trọng cho gradient flow analysis
- Xác định vanishing/exploding gradient regimes

### 3.2 Neural Tangent Kernel (NTK)

FIP justifies:
$$NTK(x, x') = \lim_{\text{width} \to \infty} \langle \nabla_\theta f(x), \nabla_\theta f(x') \rangle$$

**Gradient independence:** Weights trong forward và backward pass có thể coi là độc lập.

### 3.3 Applications

- **Initialization schemes:** Tìm optimal variance scaling
- **Training dynamics:** NTK regime analysis
- **Capacity analysis:** Spectrum of feature matrices

---

## 4. Liên Quan Đến INLA

### 4.1 Kỹ Thuật Cho INLA

| Khía cạnh | Ứng dụng cho INLA |
|-----------|-------------------|
| **Singular values** | Phân tích effective rank của attention matrices |
| **Spectral entropy** | Đo lường information content qua spectrum |
| **Effective rank** | $\text{rank}_\epsilon = \#\{i: \sigma_i > \epsilon\}$ |
| **Free probability** | Phân tích attention + MLP combinations |

### 4.2 Cụ Thể

1. **Spectral analysis của INLA attention:**
   - Singular value distribution của feature map matrix
   - So sánh với MP law baseline

2. **Effective rank:**
   - Đo rank preservation của INLA
   - So sánh với softmax attention

3. **Spectral entropy:**
   $$H(\sigma) = -\sum_i p_i \log p_i, \quad p_i = \sigma_i^2 / \sum_j \sigma_j^2$$
   - Đánh giá information content

---

## 5. Thách Thức

1. **Infinite width:** Giả định không realistic
2. **Attention complexity:** Attention không phải là entrywise operation
3. **Nonlinearities:** Softmax khó phân tích hơn ReLU
4. **Multi-head:** Tương tác giữa heads phức tạp

---

## 6. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | Free Independence Principle |
| **Toán học** | RMT, free probability, MP law |
| **Ứng dụng** | Jacobian spectrum, NTK |
| **Với INLA** | Công cụ cho spectral analysis |

---

*Phân tích cho dự án INLA — 06/2026*
