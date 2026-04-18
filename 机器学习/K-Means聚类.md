# K-Means聚类

_无监督学习的经典聚类算法_

---

## 第一部分：什么是K-Means？

### 核心思想

**把数据分成 K 个簇，每个簇有一个中心点（质心）。**

```
K-Means 过程：
1. 随机选择 K 个中心
2. 把每个点分配给最近的中心
3. 重新计算中心点
4. 重复 2-3 直到收敛
```

---

## 第二部分：代码实现

```python
def kmeans(X, k, max_iters=100):
    n_samples = X.shape[0]

    # 随机选择 K 个中心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        # 分配每个点到最近的中心
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        labels = np.argmin(distances, axis=1)

        # 重新计算中心
        new_centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)

        centroids = new_centroids

    return labels, centroids
```

---

## 第三部分：sklearn 实现

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X)
```

---

## 小结

1. **K-Means**是无监督聚类算法
2. 步骤：初始化中心 → 分配 → 更新 → 重复
3. 用肘部法则或轮廓系数选 K

---

_继续学习：下一章「梯度下降」_
