# Phân Tích Paper: Attention Is Not All You Need

## Pure Attention Loses Rank Doubly Exponentially with Depth

**Thông tin:**  
- **arXiv:** [2103.03404](https://arxiv.org/abs/2103.03404) (v2, Aug 2023)  
- **Tác giả:** Yihe Dong (Google), Jean-Baptiste Cordonnier (EPFL), Andreas Loukas (EPFL)  
- **File PDF:** `2103.03404_Dong_Attention_Is_Not_All_You_Need.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Transformer đã trở nên ubiquitous trong ML. Tuy nhiên, hiểu biết về lý do tại sao chúng hoạt động vẫn hạn chế.

**Câu hỏi:** Điều gì xảy ra khi stack nhiều self-attention layers **không** skip connections và MLPs?

**Kết quả chính:** Self-attention thuần túy mất expressive power **doubly exponentially** theo depth → output là rank-1 matrix (mọi token giống hệt nhau).

---

## 2. Path Decomposition

### 2.1 Công thức

Output head $h$ tại layer $l$:

$$SA_h(X) = P_h X W_{V,h} + 1b_{V,h}^\top$$

$P_h$ là **row-stochastic matrix**:

$$P_h = \text{softmax}\left(\frac{X W_{Q,h} W_{K,h}^\top X^\top}{\sqrt{d_{qk}}}\right)$$

### 2.2 Path Decomposition Theorem

**Theorem 2.1:** Output depth-$L$ SAN được phân rã:

$$SAN(X) = \sum_{path \in [H]^L} P_{path} X W_{path} + 1b^\top$$

- $P_{path} = P_{h_L}^L \cdots P_{h_1}^1$ (stochastic matrix)
- $W_{path} = W_{h_1}^1 \cdots W_{h_L}^L$ (weight product)
- **Số paths:** $H^L$ (exponential)

Mỗi path là một single-head network. SAN = **ensemble các single-head networks** độc lập yếu.

---

## 3. Rank Collapse

### 3.1 Định nghĩa

$res(X) = X - 1x^\top$ với $x = \arg\min_x \|X - 1x^\top\|$. Residual đo khoảng cách đến rank-1.

### 3.2 Single-Head Convergence

**Theorem 2.2 (Simplified):** Với $\|W_{QK}^l\|_1 \|W_V^l\|_{1,\infty} \leq \beta$:

$$\|res(SAN(X))\|_{1,\infty} \leq \left(\frac{4\gamma\beta}{\sqrt{d_{qk}}}\right)^{\frac{3^L - 1}{2}} \|res(X)\|^{3^L}_{1,\infty}$$

**Doubly exponential + cubic rate ($3^L$).**

### 3.3 Multi-Head Convergence (No Skip)

**Theorem 2.3 (Simplified):**

$$\|res(SAN(X))\|_{1,\infty} \leq \left(\frac{4\gamma\beta H}{\sqrt{d_{qk}}}\right)^{\frac{3^L - 1}{2}} \|res(X)\|^{3^L}_{1,\infty}$$

**Điều kiện:** $\frac{4\gamma\beta H}{\sqrt{d_{qk}}} < 1$

### 3.4 Proof Sketch

1. $X = 1x^\top + R$ (mean + residual)
2. $P_h = \text{softmax}(R W_{QK} R^\top / \sqrt{d_{qk}} + 1r^\top)$
3. Nếu $E = R W_{QK} R^\top / \sqrt{d_{qk}}$ nhỏ → $P_h \approx 1q^\top$
4. $P_h X = 1x^\top + \text{softmax}(1r^\top + E)R$
5. $\|res(P_h X)\| \leq 2\|D 1_q^\top R\|$ — residual giảm
6. Đệ quy L lần → tốc độ $3^L$

**Cubic rate** vì rank attention matrix phụ thuộc rank input → **cascading effect**.

---

## 4. Cơ Chế Chống Rank Collapse

### 4.1 Skip Connections

Khi thêm skip ($P_0 = I, W_0 = I$):

$$X^L = \sum_{h_1,\ldots,h_L \in ([H] \cup \{0\})^L} (P_{h_L}^L \cdots P_{h_1}^1) X (W_{h_1}^1 \cdots W_{h_L}^L)$$

**Phân bố path lengths:** $|P_l| = \binom{L}{l} H^l$ paths độ dài $l$

Các **short paths** ngăn chặn rank collapse.

**Claim 3.1:** Residual có **lower bound dương** — không hội tụ về 0.

### 4.2 MLPs

MLP làm chậm convergence qua Lipschitz constant nhưng không ngăn hoàn toàn.

### 4.3 Layer Normalization

Chưa được phân tích chi tiết trong paper này.

---

## 5. Thí Nghiệm

- Architectures: Transformer, BERT, GPT-2
- Metrics: effective rank, cosine similarity
- **Kết quả:** Pure SANs → rank giảm doubly exponential. Skip connections → rank maintained. MLP không skip → chậm nhưng collapse.

---

## 6. Liên Quan Đến INLA

| Khía cạnh | Mô tả |
|-----------|-------|
| **Động cơ** | Attention có inductive bias mạnh về token uniformity |
| **Vấn đề** | Linear attention có thể bị rank collapse nhanh hơn |
| **INLA** | Cần normalization để bảo toàn rank structure |
| **Kỹ thuật** | Path decomposition, residual analysis |

---

## 7. Thách Thức

1. Chứng minh chỉ cho SAN không skip connections
2. Điều kiện hội tụ pessimistic
3. Chưa xét LayerNorm, BatchNorm
4. Linear attention chưa được đề cập

---

## 8. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | Path decomposition + rank collapse proof |
| **Toán học** | Doubly exponential, cubic rate $3^L$ |
| **Kết quả** | Skip connections chống collapse, MLP làm chậm |
| **Với INLA** | Rank collapse là vấn đề cần giải quyết |

---

*Phân tích cho dự án INLA — 06/2026*
