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
Step 0: 不允许经过任何中间顶点
        dist[i][j] = 直接边的权重

Step 1: 允许经过顶点 A 作为中间顶点
        检查所有通过 A 的路径是否更短
        
Step 2: 允许经过顶点 A, B 作为中间顶点
        检查所有通过 A 或 B 的路径是否更短

Step 3: 允许经过顶点 A, B, C 作为中间顶点
        检查所有通过 A, B 或 C 的路径是否更短

Step 4: 允许经过顶点 A, B, C, D 作为中间顶点
        最终结果！
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
    # 复制原始图（避免修改原图）
    dist = [row[:] for row in graph]

    # -------- Floyd-Warshall 主循环 --------
    # k 是中间顶点
    for k in range(n):
        # i 是起点
        for i in range(n):
            # j 是终点
            for j in range(n):
                # 如果经过 k 的路径比直接路径更短
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    # 更新最短距离
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def print_dist_matrix(dist, vertex_names=None):
    """打印距离矩阵"""
    n = len(dist)
    names = vertex_names or [str(i) for i in range(n)]

    # 打印表头
    print("     ", end="")
    for name in names:
        print(f"{name:>6}", end="")
    print()

    # 打印每行
    for i in range(n):
        print(f"{names[i]:>4}", end=" ")
        for j in range(n):
            val = dist[i][j]
            if val == float('inf'):
                print("    ∞", end="")
            else:
                print(f"{val:>6}", end="")
        print()
```

---

## 第四部分：实际运行代码

```python
# 定义无穷大
INF = float('inf')

# 示例：城市之间的距离
#    北京  上海  广州  成都
# 北京  0    8    ∞    5
# 上海  8    0    3    ∞
# 广州  ∞    3    0    2
# 成都  5    ∞    2    0

cities = [
    [0, 8, INF, 5],   # 北京
    [8, 0, 3, INF],   # 上海
    [INF, 3, 0, 2],   # 广州
    [5, INF, 2, 0],   # 成都
]

print("原始距离矩阵：")
print_dist_matrix(cities, ["北京", "上海", "广州", "成都"])

# Floyd-Warshall
shortest = floyd_warshall(cities)

print("\n最短距离矩阵：")
print_dist_matrix(shortest, ["北京", "上海", "广州", "成都"])

# 测试任意两点间最短距离
print("\n查询：")
print(f"  北京 → 广州: {shortest[0][2]}")  # 北京到广州的最短距离
print(f"  成都 → 上海: {shortest[3][1]}")  # 成都到上海的最短距离
```

---

## 第五部分：实际应用场景

### 🗺️ 场景1：全国城市距离查询系统

预计算所有城市间的最短距离。

```python
# 城市间公路网络
# 预计算后，查询任意两城市距离 O(1)
roads = [
    [0, 100, INF, 200],   # A
    [100, 0, 150, INF],   # B
    [INF, 150, 0, 100],   # C
    [200, INF, 100, 0],   # D
]

shortest = floyd_warshall(roads)

# 查询
print(f"A 到 C 的最短距离: {shortest[0][2]}")  # 250 (A→B→C)
```

### 👥 场景2：社交网络人脉分析

计算两人之间的"距离"（几度人脉）。

```python
# 社交网络：1 表示认识，∞ 表示不认识
# 1度 = 直接认识，2度 = 朋友的朋友
social = [
    [0, 1, INF, 1],   # 你
    [1, 0, 1, INF],   # A
    [INF, 1, 0, 1],   # B
    [1, INF, 1, 0],   # C
]

degrees = floyd_warshall(social)
print(f"你和 B 之间是 {int(degrees[0][2])} 度人脉")  # 2度
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

```python
# Floyd-Warshall 的子问题：
# dist[i][j][k] = 从 i 到 j，最多经过前 k 个顶点的最短距离

# 状态转移：
# dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1])
```

### 负环

**定义：** 环上所有边权重之和为负数。

```python
# Floyd-Warshall 可以检测负环
# 如果 dist[i][i] < 0，则存在负环
```

---

## ✅ 小结

1. **Floyd-Warshall** 求所有顶点对之间的最短路径
2. 使用动态规划，逐步引入中间顶点
3. 时间复杂度 O(V³)，适合小规模图
4. 一次性得到所有距离，方便查询

---

_继续学习：下一章「Prim 最小生成树」_
