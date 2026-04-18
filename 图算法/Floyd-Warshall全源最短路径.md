# Floyd-Warshall 算法

_求所有顶点对之间最短路径的动态规划算法_

---

## 📖 学习目标

1. 理解全源最短路径的概念
2. 掌握 Floyd-Warshall 的动态规划思想
3. 能解决任意两点间最短距离问题

---

## 第一部分：什么是 Floyd-Warshall？

### 📝 问题定义

**全源最短路径问题：**
- 给定一个有向带权图
- 求**任意两个顶点**之间的最短路径

### Floyd-Warshall 的核心思想

**动态规划 + 中间顶点：**
- 逐步引入中间顶点
- 更新所有顶点对之间的距离

```
dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            通过顶点 k
```

---

## 第二部分：图解过程

### 示例图

```
    4        2
   →A ─────→ B
   ↑ \      ↗ ↑
  3  1\    / 5
   │    \ /
   D ───→ C
     2

邻接矩阵（∞ 表示无边）：
    A   B   C   D
A   0   4   1   3
B   ∞   0   2   ∞
C   ∞   ∞   0   ∞
D   ∞   5   ∞   0
```

### Floyd-Warshall 过程

```
Step 1: 允许经过顶点 A 作为中间顶点
        检查所有通过 A 的路径是否更短

Step 2: 允许经过顶点 A, B 作为中间顶点
        检查所有通过 A 或 B 的路径是否更短

Step 3: 允许经过顶点 A, B, C 作为中间顶点
        检查所有通过 A, B 或 C 的路径是否更短

Step 4: 最终结果！
```

---

## 第三部分：代码逐行解释

```python
def floyd_warshall(graph):
    """
    Floyd-Warshall 全源最短路径

    参数:
        graph: 邻接矩阵
               graph[i][j] = 从 i 到 j 的权重
               graph[i][j] = ∞ 表示无边
    返回值:
        所有顶点对之间的最短距离矩阵
    """

    n = len(graph)

    # -------- 初始化距离矩阵 --------
    dist = [row[:] for row in graph]

    # -------- Floyd-Warshall 主循环 --------
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
```

---

## 第四部分：实际运行代码

```python
# 定义无穷大
INF = float('inf')

# 示例：城市之间的距离
cities = [
    [0, 8, INF, 5],   # 北京
    [8, 0, 3, INF],   # 上海
    [INF, 3, 0, 2],   # 广州
    [5, INF, 2, 0],   # 成都
]

print("原始距离矩阵：")
for row in cities:
    print(row)

# Floyd-Warshall
shortest = floyd_warshall(cities)

print("\n最短距离矩阵：")
for row in shortest:
    print(row)

print(f"\n北京 → 广州: {shortest[0][2]}")
print(f"成都 → 上海: {shortest[3][1]}")
```

---

## 第五部分：实际应用场景

### 🗺️ 场景1：全国城市距离查询系统

```python
# 预计算后，查询任意两城市距离 O(1)
roads = [
    [0, 100, INF, 200],
    [100, 0, 150, INF],
    [INF, 150, 0, 100],
    [200, INF, 100, 0],
]
```

### 👥 场景2：社交网络人脉分析

```python
# 1 表示认识，∞ 表示不认识
social = [
    [0, 1, INF, 1],
    [1, 0, 1, INF],
    [INF, 1, 0, 1],
    [1, INF, 1, 0],
]
degrees = floyd_warshall(social)
```

---

## 第六部分：算法对比

### 最短路径算法对比

| 算法 | 适用场景 | 时间复杂度 |
|------|----------|-----------|
| BFS | 无权图单源最短 | O(V+E) |
| Dijkstra | 带权图单源最短 | O((V+E)logV) |
| **Floyd-Warshall** | **全源最短** ⭐ | **O(V³)** |

### Floyd-Warshall 的优势

```
✅ 代码简单，矩阵操作
✅ 一次性得到所有顶点对距离
✅ 可以检测负环
✅ 适合 V <= 500 的稠密图
```

---

## 第七部分：名词解释

### 动态规划

**定义：** 保存子问题的解，避免重复计算。

### 负环

**定义：** 环上所有边权重之和为负数。

---

## ✅ 小结

1. **Floyd-Warshall** 求所有顶点对之间的最短路径
2. 使用动态规划，逐步引入中间顶点
3. 时间复杂度 O(V³)，适合小规模图
4. 一次性得到所有距离，方便查询

---

_继续学习：下一章「拓扑排序」_
