# K-Means聚类

_物以类聚，人以群分_

---

## 📖 学习目标

1. 理解无监督学习的概念
2. 掌握 K-Means 的算法流程和推导
3. 理解肘部法则和轮廓系数
4. 掌握 K-Means++ 初始化方法

---

## 第一部分：什么是无监督学习？

### 有监督 vs 无监督

```
有监督学习（有标签）：
输入 (x₁, y₁), (x₂, y₂), ... (xₙ, yₙ)
学习 x → y 的映射
例如：邮件 → 垃圾邮件/正常邮件

无监督学习（无标签）：
输入 x₁, x₂, ... xₙ
学习数据内部的规律
例如：把用户分成不同的群体
```

### K-Means 解决什么问题？

```
给客户分群：
- 我们不知道应该分成几类
- 也不知道每类是什么样的
- K-Means 自动发现数据中的分组
```

---

## 第二部分：K-Means 算法详解

### 算法流程

```
1. 随机选择 K 个中心点（质心）
2. 把每个样本分配给最近的中心点（形成 K 个簇）
3. 重新计算每个簇的中心点
4. 重复 2-3 直到收敛（中心点不再变化）
```

### 图解过程

```
Step 1: 随机选择 K=3 个中心（★）
        
         ★                           ★
                                      
                          ★
         
Step 2: 分配样本给最近的中心
                                   
     ●  ●              ●●  ●        ●  ●
         ★           ●● ★ ●●
                              ●●  ●
                          ★
   
Step 3: 重新计算中心点
                                   
                   ★(新中心)
                                      
         ●  ●  ●            ●  ●  ●      ●  ●
         ★(新中心)                     ★(新中心)
                                      
Step 4: 重复直到收敛
```

---

## 第三部分：数学推导

### 目标函数

**K-Means 的目标：最小化所有样本到其所属簇中心的距离平方和**

```
最小化：J = Σᵢ Σⱼ ||x_ⱼ - μᵢ||²

其中：
- i = 1 到 K（簇的索引）
- j = 1 到 nᵢ（簇 i 中的样本）
- μᵢ = 簇 i 的中心点
- ||x - μ||² = 欧氏距离的平方
```

### 两步交替优化

**第一步：分配样本（固定 μ）**

```
对于每个样本 x_j，找到最近的中心点 μ_i

x_j 被分配到 argmin_i ||x_j - μ_i||²
```

**第二步：更新中心（固定分配）**

```
对于每个簇 i，重新计算中心点

μ_i = (1/nᵢ) × Σⱼ x_j

其中 nᵢ 是簇 i 中的样本数
```

---

## 第四部分：代码实现

```python
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4, verbose=True):
    """
    K-Means 聚类算法

    参数:
        X: 数据矩阵 (n_samples, n_features)
        k: 簇的数量
        max_iters: 最大迭代次数
        tol: 收敛阈值（中心变化小于此值认为收敛）
        verbose: 是否打印进度

    返回:
        labels: 每个样本的簇标签 (n_samples,)
        centroids: 簇中心点 (k, n_features)
        history: 迭代历史（用于可视化）
    """
    n_samples, n_features = X.shape

    # ===== Step 1: 随机选择 K 个中心点 =====
    # 方法1：随机选择 K 个样本作为初始中心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()

    if verbose:
        print("初始中心点:")
        print(centroids)

    history = {'centroids': [centroids.copy()], 'labels': []}

    # ===== 迭代优化 =====
    for iteration in range(max_iters):
        if verbose:
            print(f"\n=== 迭代 {iteration + 1} ===")

        # ===== Step 2: 分配样本到最近的中心 =====
        # 计算每个样本到每个中心的距离
        # distances[i, j] = 样本 i 到 中心 j 的距离平方
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # broadcasting: X - centroids[i]
            distances[:, i] = np.sum((X - centroids[i]) ** 2, axis=1)

        # 分配：选择距离最小的中心
        labels = np.argmin(distances, axis=1)

        if verbose:
            # 统计每个簇的样本数
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  簇 {u}: {c} 个样本")

        # ===== Step 3: 更新中心点 =====
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # 如果某个簇是空的，随机选一个样本作为中心
                new_centroids[i] = X[np.random.choice(n_samples)]

        # ===== 检查收敛 =====
        # 计算中心点变化的大小
        shift = np.linalg.norm(new_centroids - centroids)
        if verbose:
            print(f"中心点移动距离: {shift:.6f}")

        centroids = new_centroids
        history['centroids'].append(centroids.copy())
        history['labels'].append(labels.copy())

        if shift < tol:
            if verbose:
                print(f"\n✓ 在第 {iteration + 1} 次迭代收敛")
            break

    return labels, centroids, history


# ========== 完整示例 ==========
# 生成示例数据：3 个簇
np.random.seed(42)

# 生成 3 个簇的数据
cluster1 = np.random.randn(50, 2) + [2, 2]   # 中心在 (2,2)
cluster2 = np.random.randn(50, 2) + [-2, 1]   # 中心在 (-2,1)
cluster3 = np.random.randn(50, 2) + [0, -2]  # 中心在 (0,-2)

X = np.vstack([cluster1, cluster2, cluster3])
print(f"数据形状: {X.shape}")
print(f"数据范围: x ∈ [{X[:,0].min():.1f}, {X[:,0].max():.1f}], y ∈ [{X[:,1].min():.1f}, {X[:,1].max():.1f}]")

# 运行 K-Means
k = 3
labels, centroids, history = kmeans(X, k, verbose=True)

print(f"\n最终簇中心:")
for i, c in enumerate(centroids):
    print(f"  簇 {i}: ({c[0]:.2f}, {c[1]:.2f})")
```

---

## 第五部分：如何选择 K？

### 方法1：肘部法则（Elbow Method）

**原理：随着 K 增大，簇内误差平方和（SSE）减小，但减小的速度会变化**

```python
def elbow_method(X, k_range):
    """
    肘部法则：绘制不同 K 值对应的 SSE

    SSE = Σᵢ Σⱼ ||x_j - μ_i||²
    """
    results = []

    for k in k_range:
        # 运行 K-Means
        labels, centroids, _ = kmeans(X, k, verbose=False)

        # 计算 SSE（簇内误差平方和）
        sse = 0
        for i in range(k):
            cluster_points = X[labels == i]
            sse += np.sum((cluster_points - centroids[i]) ** 2)

        results.append({'k': k, 'sse': sse})
        print(f"K={k}: SSE={sse:.2f}")

    return results


# 测试肘部法则
k_range = range(1, 10)
results = elbow_method(X, k_range)

# 找拐点（肘部）
# SSE 下降速度明显变缓的点就是最优 K
```

### 肘部法则图解

```
SSE
  │
  │╲
  │  ╲
  │    ╲___
  │       ╲____
  │            ╲___________
  │                         ╲
  └──────────────────────────── K
       ↑
    这里就是"肘部"
    K=3 或 K=4 是比较好的选择
```

### 方法2：轮廓系数（Silhouette Score）

```python
def silhouette_score(X, labels):
    """
    轮廓系数：衡量聚类质量

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    其中：
    a(i) = 样本 i 到同簇其他样本的平均距离
    b(i) = 样本 i 到最近其他簇的平均距离

    轮廓系数范围：[-1, 1]
    - 1 = 完全紧凑，簇分离得好
    - 0 = 簇边界模糊
    - -1 = 分类错误
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    scores = []

    for i in range(n_samples):
        cluster_i = labels[i]

        # a(i): 同簇内其他点到 i 的平均距离
        same_cluster = labels == cluster_i
        same_cluster[i] = False  # 排除自己
        if np.sum(same_cluster) > 0:
            a_i = np.mean([np.linalg.norm(X[i] - X[j])
                           for j in range(n_samples) if same_cluster[j]])
        else:
            a_i = 0

        # b(i): 到最近其他簇的平均距离
        b_i = float('inf')
        for cluster_j in unique_labels:
            if cluster_j != cluster_i:
                other_cluster = labels == cluster_j
                if np.sum(other_cluster) > 0:
                    dist = np.mean([np.linalg.norm(X[i] - X[j])
                                   for j in range(n_samples) if other_cluster[j]])
                    b_i = min(b_i, dist)

        # 轮廓系数
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0

        scores.append(s_i)

    return np.mean(scores)


# 测试不同 K 的轮廓系数
for k in range(2, 6):
    labels, _, _ = kmeans(X, k, verbose=False)
    score = silhouette_score(X, labels)
    print(f"K={k}: 轮廓系数 = {score:.4f}")
```

---

## 第六部分：K-Means++ 初始化

### 问题：随机初始化可能导致局部最优

```
不同的初始化可能产生完全不同的结果：

初始化1：
    ●●●●      ●●●●
        ●●●●      ●●●●

初始化2：
    ●●●●  ●●●●
    ●●●●  ●●●●

初始化3：
    ●●●●●●●
    ●●●●●●●
```

### K-Means++ 的改进

**核心思想：让初始中心点尽可能分散**

```python
def kmeans_plusplus_init(X, k, random_state=None):
    """
    K-Means++ 初始化方法

    步骤：
    1. 随机选一个样本作为第一个中心
    2. 对于每个样本，计算到最近中心的距离
    3. 按距离比例概率抽取下一个中心（距离越大越可能被选）
    4. 重复 2-3 直到选够 K 个中心
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]

    # ===== Step 1: 随机选第一个中心 =====
    indices = []
    first_idx = np.random.randint(n_samples)
    indices.append(first_idx)
    centroids = [X[first_idx].copy()]

    # ===== Step 2-3: 选择剩余的 K-1 个中心 =====
    for _ in range(1, k):
        # 计算每个样本到最近中心的距离平方
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for c in centroids:
                dist = np.sum((X[i] - c) ** 2)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist

        # 按距离比例概率抽取下一个中心
        # 距离越大，被选中的概率越高
        probabilities = distances / distances.sum()
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_idx].copy())
        indices.append(next_idx)

    return np.array(centroids), indices


# 对比：随机初始化 vs K-Means++ 初始化
print("=== 随机初始化 ===")
centroids_random, _ = kmeans_plusplus_init(X, 3, random_state=None)
print(centroids_random)

print("\n=== K-Means++ 初始化 ===")
centroids_pp, _ = kmeans_plusplus_init(X, 3, random_state=42)
print(centroids_pp)
```

---

## 第七部分：sklearn 实现

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ===== 基本使用 =====
kmeans = KMeans(
    n_clusters=3,           # 簇数量
    init='k-means++',       # 初始化方法：'k-means++' 或 'random'
    n_init=10,              # 运行多少次，取最好结果
    max_iter=300,           # 最大迭代次数
    tol=1e-4,              # 收敛阈值
    random_state=42
)

# 训练
kmeans.fit(X)

# 获取结果
labels = kmeans.labels_           # 簇标签
centroids = kmeans.cluster_centers_  # 簇中心
inertia = kmeans.inertia_        # SSE（簇内误差平方和）

# 预测
new_labels = kmeans.predict(X_new)

# ===== 可视化 =====
plt.figure(figsize=(10, 6))

# 绘制数据点，颜色按簇标签
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
               c=colors[i], label=f'簇 {i}', alpha=0.6)

# 绘制中心点
plt.scatter(centroids[:, 0], centroids[:, 1],
           c='black', marker='X', s=200, label='中心点')

plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('K-Means 聚类结果')
plt.legend()
plt.show()
```

---

## 第八部分：实际应用

### 客户分群

```python
# 假设有客户的以下特征：
# - 年龄
# - 年收入
# - 购买频率
# - 上次购买距今

customer_features = np.array([
    [25, 50000, 10, 30],   # 年轻，低收入，高频，上次30天
    [35, 80000, 5, 60],    # 中年，高收入，低频，上次60天
    [45, 120000, 2, 90],   # 年长，高收入，非常低频
    [22, 30000, 20, 7],    # 很年轻，低收入，超高频
    # ... 更多客户
])

# K-Means 聚类
kmeans = KMeans(n_clusters=4, random_state=42)
segments = kmeans.fit_predict(customer_features)

# 分析每个簇的特点
for i in range(4):
    cluster_customers = customer_features[segments == i]
    print(f"\n簇 {i} ({len(cluster_customers)} 人):")
    print(f"  平均年龄: {cluster_customers[:, 0].mean():.1f}")
    print(f"  平均收入: {cluster_customers[:, 1].mean():.0f}")
    print(f"  平均购买频率: {cluster_customers[:, 2].mean():.1f}")
```

---

## 第九部分：K-Means 的局限性

### 问题1：需要预先指定 K

```
解决：肘部法则、轮廓系数、Gap Statistics
```

### 问题2：对初始中心敏感

```
解决：K-Means++ 初始化，多次运行取最优
```

### 问题3：只能发现球形簇

```
问题：K-Means 假设簇是球形的（各向同性）

        ✓ K-Means 擅长          ✗ K-Means 不擅长
     ┌─────────┐              ┌─────────┐
     │ ●●●●●●● │              │ ●●●    │
     │ ●●●●●●● │              │   ●●●● │
     │ ●●●●●●● │              │ ●●●    │
     └─────────┘              └─────────┘
```

### 问题4：对噪声和离群点敏感

```
解决：使用 DBSCAN 等密度-based 算法
```

---

## ✅ 小结

1. **K-Means**是无监督聚类算法
2. 目标：最小化 SSE（簇内误差平方和）
3. 流程：随机初始化 → 分配 → 更新 → 重复
4. K-Means++ 改进初始化，让中心点更分散
5. 选择 K：肘部法则、轮廓系数
6. 优点：简单、快速
7. 缺点：需预设 K、对初始值敏感、只能发现球形簇

---

_继续学习：下一章「梯度下降」_
