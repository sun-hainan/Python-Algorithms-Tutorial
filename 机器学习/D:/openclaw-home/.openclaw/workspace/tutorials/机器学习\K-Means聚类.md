# K-Means聚类

_无监督学习的经典聚类算法_

---

## 📖 学习目标

1. 理解决K-Means的工作原理
2. 掌握聚类的评估指标
3. 能解决无标签数据的聚类问题

---

## 第一部分：什么是K-Means？

### 🎯 核心思想

**把数据分成 K 个簇，每个簇有一个中心点（质心）。**

```
K-Means 过程：
1. 随机选择 K 个中心点
2. 把每个点分配给最近的中心
3. 重新计算中心点
4. 重复 2-3 直到收敛
```

---

## 第二部分：图解过程

```
Step 1: 随机选择K个中心
        ●        ○
    ●       ●
                  ●
        (K=2, ●和○是中心)

Step 2: 分配给最近的中心
        ●●●●     ○○○
        ●●●●     ○○○
        (所有●属于上面的中心，○属于下面)

Step 3: 重新计算中心
        ●●●●●    ○○○○
        ●●●●○    ○○○○

Step 4: 迭代直到收敛
```

---

## 第三部分：代码实现

```python
import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-Means 聚类

    X: 数据 (n_samples, n_features)
    k: 聚类数
    """
    n_samples = X.shape[0]

    # 1. 随机选择 K 个中心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]

    for iteration in range(max_iters):
        # 2. 分配每个点到最近的中心
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)

        # 3. 重新计算中心
        new_centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)

        # 4. 检查收敛
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"收敛于第 {iteration} 次迭代")
            break

        centroids = new_centroids

    return labels, centroids


# 测试
np.random.seed(42)
X1 = np.random.randn(50, 2) + [2, 2]
X2 = np.random.randn(50, 2) + [-2, -1]
X = np.vstack([X1, X2])

labels, centroids = kmeans(X, k=2)
print(f"聚类标签: {np.unique(labels)}")
print(f"中心点: {centroids}")
```

---

## 第四部分：评估指标

### 肘部法则 (Elbow Method)

```python
def elbow_method(X, k_range):
    """用肘部法则确定最优K"""
    inertias = []

    for k in k_range:
        labels, centroids = kmeans(X, k, max_iters=50)
        inertia = 0
        for i in range(k):
            cluster_points = X[labels == i]
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
        inertias.append(inertia)

    # 找到拐点
    # 在 inertias 曲线上找拐点
    return inertias
```

### 轮廓系数 (Silhouette Score)

```python
def silhouette_score(X, labels):
    """
    轮廓系数：衡量聚类质量
    值范围 [-1, 1]，越接近 1 越好
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)

    s = 0
    for i in range(n_samples):
        cluster_i = labels[i]

        # a(i): 同簇内其他点到 i 的平均距离
        a_i = np.mean([np.linalg.norm(X[i] - X[j])
                       for j in range(n_samples) if labels[j] == cluster_i and j != i])

        # b(i): 到最近其他簇的平均距离
        b_i = min([np.mean([np.linalg.norm(X[i] - X[j])
                            for j in range(n_samples) if labels[j] == c])
                   for c in unique_labels if c != cluster_i])

        s += (b_i - a_i) / max(a_i, b_i)

    return s / n_samples
```

---

## 第五部分：sklearn 实现

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='x', s=200)
plt.show()

print(f"聚类中心: {kmeans.cluster_centers_}")
```

---

## 第六部分：K-Means++ 初始化

```python
def kmeans_plusplus(X, k):
    """K-Means++ 初始化：选择距离已选中心最远的点"""
    n_samples = X.shape[0]
    centroids = []

    # 第一个中心：随机选
    centroids.append(X[np.random.randint(n_samples)])

    for _ in range(1, k):
        # 计算每个点到最近中心的距离
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = np.inf
            for c in centroids:
                dist = np.linalg.norm(X[i] - c)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist

        # 按距离概率抽样
        probs = distances / distances.sum()
        next_center = X[np.random.choice(n_samples, p=probs)]
        centroids.append(next_center)

    return np.array(centroids)
```

---

## 第七部分：实际应用场景

### 🎯 场景1：客户分群

```python
# 特征：购买金额、购买频率、活跃天数
# 把客户分成 K 个群体，针对性营销
```

### 🖼️ 场景2：图像压缩

```python
# 把图像中的颜色减少到 K 种
# 每个像素用最近的聚类中心替代
```

### 🏠 场景3：异常检测

```python
# 不属于任何簇的点可能是异常值
```

---

## 第八部分：名词解释

### 质心 (Centroid)

**定义：** 簇内所有点的平均值。

### 惯性 (Inertia)

**定义：** 所有点到各自簇中心的距离平方和。

```python
inertia = Σ ||x_i - μ_cluster(x_i)||²
```

---

## ✅ 小结

1. **K-Means**是无监督聚类算法
2. 步骤：初始化中心 → 分配 → 更新 → 重复
3. K-Means++ 改进了初始化
4. 用肘部法则或轮廓系数选 K
5. 优点：简单、快速
6. 缺点：需要指定 K、对初始值敏感

---

_继续学习：下一章「梯度下降」_
