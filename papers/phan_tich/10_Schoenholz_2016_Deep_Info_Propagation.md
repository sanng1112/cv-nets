# Phân Tích Paper: Deep Information Propagation

**Thông tin:**  
- **arXiv:** [1611.01232](https://arxiv.org/abs/1611.01232) (ICLR 2017)  
- **Tác giả:** Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, Jascha Sohl-Dickstein (Google Brain, Stanford)  
- **File PDF:** `1611.01232_Schoenholz_Deep_Info_Propagation.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Deep networks khó train vì vanishing/exploding gradients. Cần hiểu signal propagation trong random networks để thiết kế initialization schemes tốt hơn.

**Mục tiêu:** Dùng **mean field theory** để phân tích signal propagation.

---

## 2. Toán Học Cốt Lõi

### 2.1 Mean Field Theory

Với $n$ neurons/layer, weights $w_{ij} \sim \mathcal{N}(0, \sigma_w^2/n)$, biases $b_j \sim \mathcal{N}(0, \sigma_b^2)$.

**Forward propagation:**

$$h_i^{l+1} = \sum_j w_{ij}^{l+1} \phi(h_j^l) + b_i^{l+1}$$

Trong limit $n \to \infty$, theo Central Limit Theorem:
$$h_i^{l+1} \sim \mathcal{N}(0, q^{l+1})$$

Với:
$$q^{l+1} = \sigma_w^2 \int \phi(\sqrt{q^l} z)^2 \mathcal{D}z + \sigma_b^2$$

$z \sim \mathcal{N}(0, 1)$, $\mathcal{D}z = \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz$.

### 2.2 Depth Scales

**Correlation between inputs $x^1, x^2$:**

$$c^{l+1} = \frac{\sigma_w^2}{q^{l+1}} \int \phi(\sqrt{q^l} z_1) \phi(\sqrt{q^l} z_2) \mathcal{D}z_1 \mathcal{D}z_2 + \frac{\sigma_b^2}{q^{l+1}}$$

Với $\begin{bmatrix} z_1 \\ z_2 \end{bmatrix} \sim \mathcal{N}(0, \begin{bmatrix} 1 & c^l \\ c^l & 1 \end{bmatrix})$.

**Fixed point:** $c^*$ khi $l \to \infty$.

**Depth scale $\xi$:**
$$|c^l - c^*| \sim \exp(-l/\xi)$$

- $\xi$ đo tốc độ convergence về fixed point
- $\xi$ diverges tại **order-to-chaos transition**

### 2.3 Pha: Order vs Chaos

| Phase | $\chi_1 = \partial q^{l+1}/\partial q^l$ | Behavior |
|-------|------------------------------------------|----------|
| **Ordered** | $\chi_1 < 1$ | Signals converge, gradients vanish |
| **Chaotic** | $\chi_1 > 1$ | Signals diverge, gradients explode |
| **Edge of chaos** | $\chi_1 = 1$ | **Optimal trainability** |

### 2.4 Backpropagation Mean Field

Gradient backpropagation tương tự:
$$\| \frac{\partial L}{\partial h^l} \| \sim \exp(-l/\xi)$$

- **Ordered phase:** Gradient → 0
- **Chaotic phase:** Gradient → ∞
- **Edge of chaos:** Gradient ổn định

### 2.5 Dropout Effect

Dropout destroys the fixed point structure → không có edge of chaos → depth bị giới hạn.

---

## 3. Liên Quan Đến INLA

### 3.1 Signal Propagation trong Attention

| Khía cạnh | Deep Info Prop | INLA |
|-----------|---------------|------|
| **Kiến trúc** | FC layers | Attention layers |
| **Phân tích** | Mean field theory | Cần mở rộng |
| **Depth scale** | $\xi$ cho signal/correlation | Rank decay rate |
| **Criticality** | Edge of chaos | Optimal normalization |
| **Vấn đề** | Vanishing/exploding gradients | Rank collapse/over-smoothing |

### 3.2 Kỹ Thuật Có Thể Áp Dụng

1. **Mean field theory cho attention:**
   - Phân tích signal propagation qua attention layers
   - Tìm depth scales tương tự

2. **Jacobian spectrum analysis:**
   - INLA có thể phân tích qua Jacobian
   - Xác định regimes: order/chaos

3. **Normalization as criticality:**
   - INLA normalization giống tuning về edge of chaos
   - Optimal signal propagation

### 3.3 Mở Rộng Cho INLA

- Phân tích mean field của linear attention
- Depth scales cho effective rank (thay vì correlation)
- Thiết kế normalization để đạt criticality

---

## 4. Thách Thức

1. **Infinite width:** Mean field giả định $n \to \infty$
2. **Attention phức tạp:** Softmax không phải element-wise nonlinearity
3. **Skip connections:** Cần incorporate vào mean field
4. **Multi-head:** Tương tác giữa các heads phức tạp

---

## 5. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Novelty** | Mean field theory cho DNNs, depth scales |
| **Phát hiện** | Edge of chaos criticality cho trainability |
| **Với INLA** | Framework để phân tích signal propagation |

---

*Phân tích cho dự án INLA — 06/2026*
