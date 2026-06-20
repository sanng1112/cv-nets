# Phân Tích Paper: Graph Neural Networks Exponentially Lose Expressive Power

## For Node Classification

**Thông tin:**  
- **arXiv:** [1905.10947](https://arxiv.org/abs/1905.10947) (ICLR 2020)  
- **Tác giả:** Kenta Oono, Taiji Suzuki (University of Tokyo, RIKEN)  
- **File PDF:** `1905.10947_Oono_Suzuki_GNN_Oversmoothing.pdf`  

---

## 1. Bối Cảnh và Động Cơ

GNNs không cải thiện (hoặc tệ hơn) khi thêm nhiều layers — gọi là **over-smoothing**. Node representations trở nên indistinguishable.

**Mục tiêu:** Phân tích asymptotic behavior của GNNs khi số layers → ∞.

---

## 2. Toán Học Cốt Lõi

### 2.1 GCN as Dynamical System

Forward propagation của Graph Convolutional Network (GCN):

$$X^{(l+1)} = \sigma(\tilde{A} X^{(l)} W^{(l)})$$

Với:
- $\tilde{A} = D^{-1/2} A D^{-1/2}$ (normalized adjacency matrix)
- $\sigma$: ReLU activation
- $W^{(l)}$: weight matrix

### 2.2 Convergence to Invariant Subspace

**Theorem:** Với điều kiện spectral:

$$\text{dist}(X^{(L)}, \mathcal{M}) = O((s\lambda)^L)$$

Trong đó:
- $\mathcal{M}$: **invariant subspace** của dynamics
- $s$: maximum singular value của weights
- $\lambda$: spectral gap của $\tilde{A}$ (phụ thuộc graph)

**Khi $s\lambda < 1$:** Output hội tụ exponentially về $\mathcal{M}$.

### 2.3 Invariant Subspace của GCN

$\mathcal{M}$ chứa các signals:
- **Connected components:** Các nodes trong cùng component có cùng representation
- **Node degrees:** Thông tin về degree distribution

→ **Mất mát thông tin:** Không thể phân biệt nodes khác nhau trong cùng component.

### 2.4 Analysis trên Erdős–Rényi Graph

Với $G_{N,p}$ (N nodes, edge probability p):

$$P(\text{information loss}) \to 1 \quad \text{khi} \quad \frac{\log N}{pN} = o(1)$$

→ Với graph đủ dense và large, GCNs almost surely bị over-smoothing.

### 2.5 Giải Pháp: Weight Normalization

**Principled guideline:** Normalize weights để:
$$s \cdot \lambda < 1$$
- $s = \|W\|_2$ (spectral normalization)
- Hoặc $s = \max_i \sigma_i(W)$

---

## 3. Liên Quan Đến Attention

### 3.1 Tương Đồng với Rank Collapse

| Khía cạnh | GNN Oversmoothing | Attention Rank Collapse |
|-----------|-------------------|------------------------|
| **Cơ chế** | Graph diffusion | Token mixing |
| **Toán học** | $X^{l+1} = \tilde{A} X^l W^l$ | $X^{l+1} = P X^l W$ |
| **Ma trận** | $\tilde{A}$ (adjacency) | $P$ (attention stochastic) |
| **Kết quả** | Node representations → indistinguishable | Token representations → rank-1 |
| **Tốc độ** | Exponential $O((s\lambda)^L)$ | Doubly exponential $O(\gamma^{3^L})$ |

### 3.2 Kỹ Thuật Chuyển Đổi Sang Attention

1. **Invariant subspace analysis:**
   - Tìm subspace mà attention dynamics hội tụ về
   - GNN: connected components + degrees
   - Attention: token uniformity

2. **Spectral analysis:**
   - GNN: spectral gap $\lambda$ của graph Laplacian
   - Attention: spectral properties của stochastic matrix $P$

3. **Weight normalization:**
   - GNN: normalize $W$ để $s\lambda < 1$
   - Attention: normalize để tránh rank collapse

---

## 4. Liên Quan Đến INLA

### 4.1 Học Hỏi Cho INLA

1. **Spectral analysis:**
   - Phân tích spectral properties của attention matrix
   - Xác định invariant subspace

2. **Normalization principled:**
   - Oono & Suzuki: weight normalization
   - INLA: information-theoretic normalization

3. **Tốc độ hội tụ:**
   - GNN: exponential rate $(s\lambda)^L$
   - Attention: doubly exponential $3^L$
   - INLA cần chứng minh tốc độ chậm hơn

### 4.2 Mở Rộng

- Áp dụng invariant subspace analysis cho linear attention
- So sánh spectral gap của softmax vs linear attention
- Thiết kế normalization để ngăn over-smoothing

---

## 5. Thách Thức

1. **GCN-specific:** Kết quả chỉ cho GCN, chưa tổng quát
2. **ReLU là key:** Phân tích dựa vào ReLU activation
3. **Attention phức tạp:** Attention diffusion phức tạp hơn GCN
4. **Input-dependent:** Attention matrix $P$ phụ thuộc input

---

## 6. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | Over-smoothing analysis qua dynamical systems |
| **Toán học** | Invariant subspace, spectral gap |
| **Kết quả** | Weight normalization principled guideline |
| **Với INLA** | Kỹ thuật spectral analysis chuyển đổi được |

---

*Phân tích cho dự án INLA — 06/2026*
