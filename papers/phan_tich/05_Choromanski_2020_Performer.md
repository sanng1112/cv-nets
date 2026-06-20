# Phân Tích Paper: Rethinking Attention with Performers

**Thông tin:**  
- **arXiv:** [2009.14794](https://arxiv.org/abs/2009.14794) (ICLR 2021)  
- **Tác giả:** Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, et al. (Google, Cambridge)  
- **File PDF:** `2009.14794_Performer.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Linear attention methods (Katharopoulos et al.) dùng kernel feature maps nhưng không approximating softmax. Các method khác dùng sparse attention.

**Mục tiêu:** Approximate **chính xác** softmax attention với linear complexity — unbiased estimation, provable guarantees.

---

## 2. FAVOR+: Fast Attention Via positive Orthogonal Random features

### 2.1 Random Feature Approximation

Softmax kernel:
$$\exp(q^\top k) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)}[\exp(\omega^\top q - \|\omega\|^2/2) \cdot \exp(\omega^\top k - \|\omega\|^2/2)]$$

Approximation với $m$ random features:
$$\phi(q) = \frac{1}{\sqrt{m}}[\exp(\omega_1^\top q - \|\omega_1\|/2), \ldots, \exp(\omega_m^\top q - \|\omega_m\|/2)]$$

$$\exp(q^\top k) \approx \phi(q)^\top \phi(k)$$

### 2.2 Key Contributions

**1. Positive Features:**
- Standard random features có thể âm → attention weights âm (vô lý)
- FAVOR+ đảm bảo: $\phi(x) > 0$ với mọi $x$
- Dùng $\phi(x) = \exp(-\|\omega\|^2/2) \cdot \exp(\omega^\top x)$

**2. Orthogonal Features:**
- Random features độc lập → high variance
- Orthogonal features: variance giảm $O(1/m^2)$ thay vì $O(1/m)$
- Dùng Gram-Schmidt orthogonalization

### 2.3 Theoretical Guarantees

**Định lý 1 (Unbiased estimation):**
$$\mathbb{E}[\hat{A}] = A$$
Với $\hat{A} = \phi(Q) \phi(K)^\top$ là estimator của $A = \exp(QK^\top)$.

**Định lý 2 (Uniform convergence):**
$$\sup_{x,y \in S^{d-1}} |\phi(x)^\top \phi(y) - \exp(x^\top y)| \leq \epsilon$$
với probability $1-\delta$ khi $m = \Omega(d \log(d/\delta) / \epsilon^2)$.

**Định lý 3 (Variance bound):**
$$\text{Var}(\hat{A}_{ij}) \leq \frac{C}{m}$$

---

## 3. Architecture: Performer

### 3.1 Forward Pass

Regular attention:
$$Attention(Q, K, V) = D^{-1} A V, \quad D = \text{diag}(A 1_n)$$

FAVOR+ attention:
$$\hat{Attention}(Q, K, V) = \hat{D}^{-1} (\phi(Q) \phi(K)^\top V), \quad \hat{D} = \text{diag}(\phi(Q) (\phi(K)^\top 1_n))$$

### 3.2 Causal Masking

Với causal attention:
$$\hat{V}_i = \frac{\phi(Q_i) \sum_{j \leq i} \phi(K_j)^\top V_j}{\phi(Q_i) \sum_{j \leq i} \phi(K_j)^\top}$$

Complexity: $O(N \cdot m)$ — tương tự Linear Transformer.

---

## 4. Thí Nghiệm

### 4.1 Tasks

- **Pixel-prediction:** ImageNet 64x64
- **Text modeling:** LM1B
- **Protein sequence modeling:** UniRef50
- **Long-Range Arena (LRA)**

### 4.2 Kết Quả

| Task | Transformer | Performer (m=256) | Performer (m=512) |
|------|-------------|-------------------|-------------------|
| LM1B perplexity | 30.2 | 30.8 | 30.4 |
| ImageNet BPD | 3.52 | 3.55 | 3.53 |
| Protein (perplexity) | 9.8 | 10.1 | 9.9 |
| LRA avg | 58.2 | — | 57.5 |

**FAVOR+ cần $m \approx 256-512$ để gần bằng softmax accuracy.**

---

## 5. Liên Quan Đến INLA

### 5.1 Kỹ Thuật Feature Maps

| Khía cạnh | Performer | INLA |
|-----------|-----------|------|
| **Feature map** | Random (orthogonal) | Deterministic (learned?) |
| **Kernel** | Softmax approximation | Custom kernel |
| **Biased** | Unbiased | Có thể biased nhưng low variance |
| **Complexity** | $O(N \cdot m)$ | $O(N \cdot m)$ |

### 5.2 Học Hỏi Cho INLA

1. **Random features:**
   - Có thể dùng deterministic features thay vì random
   - INLA cần chứng minh low approximation error
2. **Orthogonalization:**
   - Giảm variance của estimation
   - Có thể áp dụng trong INLA
3. **Positive features:**
   - Đảm bảo attention non-negative
   - INLA cần cùng property

### 5.3 Cải Thiện

- Performer cần $m$ lớn → computational cost cao
- INLA có thể dùng feature maps thông minh hơn với $m$ nhỏ hơn
- Information-theoretic normalization thay vì random projection

---

## 6. Thách Thức

1. **$m$ lớn:** Cần 256-512 features → mất lợi thế về speed
2. **Randomness:** Không deterministic → khó reproducibility
3. **Variance:** Dù có orthogonal, variance vẫn tồn tại
4. **Performance:** Vẫn thấp hơn softmax attention (đặc biệt LRA)
5. **Implementation:** Complex hơn so với Linear Transformer

---

## 7. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | FAVOR+ (positive orthogonal random features) |
| **Toán học** | Unbiased estimation, uniform convergence |
| **Strength** | Provable approximation of softmax |
| **Weakness** | Cần nhiều features, randomness |
| **Với INLA** | Inspiration: deterministic feature maps tốt hơn |

---

*Phân tích cho dự án INLA — 06/2026*
