# Phân Tích Paper: Nyströmformer

## A Nyström-Based Algorithm for Approximating Self-Attention

**Thông tin:**  
- **arXiv:** [2102.03902](https://arxiv.org/abs/2102.03902) (AAAI 2021)  
- **Tác giả:** Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh (UW-Madison, Google Brain)  
- **File PDF:** `2102.03902_Nystromformer.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Standard self-attention $O(N^2)$ → cần approximation cho sequences dài. Các method tồn tại (low-rank, kernel, sparse) có nhiều hạn chế.

**Ý tưởng:** Dùng **Nyström method** — classic low-rank matrix approximation — để xấp xỉ softmax attention matrix với $O(N)$ complexity.

---

## 2. Toán Học Cốt Lõi

### 2.1 Nyström Method

Cho ma trận $P \in \mathbb{R}^{n \times n}$ (attention matrix), xấp xỉ:

$$\tilde{P} = P_{:,M} \cdot P_{M,M}^+ \cdot P_{M,:}$$

Với:
- $M$: tập $k$ landmark indices
- $P_{:,M} \in \mathbb{R}^{n \times k}$: cột ứng với landmarks
- $P_{M,:} \in \mathbb{R}^{k \times n}$: hàng ứng với landmarks
- $P_{M,M}^+$: Moore-Penrose pseudo-inverse của $P_{M,M}$

**Complexity:** $O(nk + k^3)$ — linear khi $k \ll n$.

### 2.2 Nyströmformer Attention

$$\text{NyströmAttention}(Q, K, V) = \text{softmax}\left(\frac{Q\tilde{K}^\top}{\sqrt{d}}\right) \cdot \left[\text{softmax}\left(\frac{\tilde{Q} \tilde{K}^\top}{\sqrt{d}}\right)^+ \cdot \text{softmax}\left(\frac{\tilde{Q} K^\top}{\sqrt{d}}\right) \cdot V\right]$$

Với $\tilde{Q}, \tilde{K} \in \mathbb{R}^{k \times d}$ là landmark queries/keys.

### 2.3 Landmark Selection

- **Segmentation:** Chia sequence thành $k$ segments, lấy segment mean
- **Alternative:** Learned landmarks (nhưng không stable)

### 2.4 Stabilization with LayerNorm

Để tránh pseudo-inverse instability:
- Thêm LayerNorm
- Dùng regularized pseudo-inverse: $(P_{M,M} + \epsilon I)^{-1}$

---

## 3. Thí Nghiệm

### 3.1 Tasks

- **GLUE benchmark** (short sequences)
- **IMDB reviews** (medium sequences)
- **Long-Range Arena (LRA)** (long sequences)

### 3.2 Kết Quả

| Model | GLUE | IMDB | LRA |
|-------|------|------|-----|
| BERT-base | 85.1 | 93.5 | — |
| Nyströmformer ($k=64$) | 84.7 | 93.1 | 55.2 |
| Nyströmformer ($k=128$) | 85.0 | 93.4 | 56.8 |

**Kết luận:** $k = 128$ ≈ full attention với 1/4 memory.

---

## 4. Liên Quan Đến INLA

### 4.1 Low-Rank Structure

| Khía cạnh | Nyströmformer | INLA |
|-----------|--------------|------|
| **Cốt lõi** | Low-rank approximation | Feature map + normalization |
| **Rank control** | Landmark count $k$ | Feature dimension $m$ |
| **Error** | Approximation error | Information-theoretic error |
| **Stability** | Pseudo-inverse issues | Normalization helps |

### 4.2 Học Hỏi

1. **Rank structure quan trọng:**
   - Effective rank của attention matrix có thể thấp
   - INLA cần khai thác điều này

2. **Landmark selection:**
   - Có thể dùng effective rank (từ RMT) để chọn số landmarks
   - Hoặc dùng thông tin mutual information

3. **Stabilization cần thiết:**
   - Normalization giúp ổn định approximation
   - INLA normalization cần đảm bảo numerical stability

---

## 5. Thách Thức

1. **Landmark selection:** Segmentation heuristic — không adaptive
2. **Pseudo-inverse:** Không ổn định, cần regularization
3. **Performance gap:** Vẫn thấp hơn softmax trên một số tasks
4. **Causal attention:** Không trivial để áp dụng Nyström method
5. **Kích thước $k$:** Trade-off giữa accuracy và speed

---

## 6. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Phương pháp** | Nyström low-rank approximation |
| **Complexity** | $O(Nk + k^3)$ |
| **Strength** | Lý thuyết low-rank matrix approximation rõ ràng |
| **Weakness** | Landmark selection heuristic |
| **Với INLA** | Rank structure và stabilization có thể áp dụng |

---

*Phân tích cho dự án INLA — 06/2026*
