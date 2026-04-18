# K-Means聚类

_物以类聚，人以群分_

---

## 📖 学习目标

1. 理解决无监督学习的概念
2. 掌握 K-Means 的算法流程
3. 理解肘部法则选择 K

---

## 第一部分：什么是无监督学习？

```
有监督：有标签 x → y
无监督：无标签，学习数据内部规律
```

---

## 第二部分：K-Means 算法

```
1. 随机选择 K 个中心点
2. 把每个样本分配给最近的中心
3. 重新计算中心点
4. 重复 2-3 直到收敛
```

```python
def kmeans(X, k, max_iters=100):
    # 随机选 K 个中心
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        # 分配样本
        distances = np.zeros((len(X), k))
        for i in range(k):
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        labels = np.argmin(distances, axis=1)

        # 更新中心
        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids
```

---

## 第三部分：肘部法则选择 K

```
SSE = Σᵢ Σⱼ ||x_j - μ_i||²

K 增大 → SSE 减小
拐点 = 最优 K
```

---

## ✅ 小结

1. **K-Means**是无监督聚类算法
2. 目标：最小化 SSE（簇内误差平方和）
3. 用**肘部法则**选择最优 K

---

_继续学习：下一章「梯度下降」_
