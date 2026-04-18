# Dijkstra 最短路径算法

_带权图最短路径的经典算法_

---

## 📖 学习目标

1. 理解 Dijkstra 的基本思想
2. 掌握最小堆的使用
3. 能解决最短路径问题

---

## 第一部分：什么是 Dijkstra？

### 🎯 Dijkstra 的核心思想

**贪心 + 最小堆：**
1. 从起点开始，逐步扩展
2. 每次选择距离最近的未访问顶点
3. 更新邻居的距离
4. 直到到达终点

### Dijkstra vs BFS

```
BFS：每条边权重相同，找边数最少的路径

Dijkstra：边有权重，找总权重最小的路径
```

### 什么时候用？

- GPS 导航：找最短距离
- 网络路由：找最小延迟
- 航班规划：找最便宜路线

---

## 第二部分：图解 Dijkstra

### 城市地图示例

```
从"家"到"公司"的最短路径：

    家 --10-- 地铁站 --20-- 公司
     |                      ^
     |                      |
     5                      15
     |                      |
     v                      |
    公交站 --3--------- 公司

边上的数字是距离（权重）
```

### Dijkstra 执行过程

```
Step 0: 初始化
  距离表：{家: 0, 地铁站: ∞, 公交站: ∞, 公司: ∞}
  已访问：{}
  优先队列：[(0, 家)]

Step 1: 取出距离最小的顶点（家 = 0）
  更新：地铁站 = min(∞, 0+10) = 10
        公交站 = min(∞, 0+5) = 5
  已访问：{家}
  队列：[(5, 公交站), (10, 地铁站)]

Step 2: 取出距离最小的顶点（公交站 = 5）
  公交站的邻居只有公司：15 → 更新：公司 = min(∞, 5+15) = 20
  已访问：{家, 公交站}
  队列：[(10, 地铁站), (20, 公司)]

Step 3: 取出距离最小的顶点（地铁站 = 10）
  更新：公司 = min(20, 10+20) = 20（不变）
  已访问：{家, 公交站, 地铁站}
  队列：[(20, 公司)]

Step 4: 取出公司，到达终点！
  最短距离 = 20
  路径：家 → 地铁站 → 公司（或者 家 → 公交站 → 公司）
```

---

## 第三部分：代码逐行解释

```python
import heapq

def dijkstra(graph, start, end):
    """
    Dijkstra 最短路径算法

    参数:
        graph: 邻接表 {(顶点): [(邻居, 权重)]}
        start: 起始顶点
        end: 目标顶点
    返回值:
        start 到 end 的最短距离，找不到返回 -1
    """

    # -------- 初始化 --------
    heap = [(0, start)]  # 优先队列：(距离, 顶点)
    visited = set()      # 已访问的顶点

    # -------- Dijkstra 主循环 --------
    while heap:
        # -------- 取出当前距离最小的顶点 --------
        dist, vertex = heapq.heappop(heap)

        # 如果已经访问过，跳过
        if vertex in visited:
            continue

        # 标记为已访问
        visited.add(vertex)

        # -------- 到达终点，返回距离 --------
        if vertex == end:
            return dist

        # -------- 更新邻居的距离 --------
        for neighbor, weight in graph.get(vertex, []):
            if neighbor not in visited:
                # 新距离 = 当前距离 + 边权重
                new_dist = dist + weight
                heapq.heappush(heap, (new_dist, neighbor))

    # 找不到路径
    return -1
```

---

## 第四部分：实际运行代码

```python
import heapq

def dijkstra(graph, start, end):
    heap = [(0, start)]
    visited = set()
    while heap:
        dist, vertex = heapq.heappop(heap)
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex == end:
            return dist
        for neighbor, weight in graph.get(vertex, []):
            if neighbor not in visited:
                heapq.heappush(heap, (dist + weight, neighbor))
    return -1


# ========================== 测试 ==========================

# 城市地图：从家到公司的路线
city_map = {
    "家": [("地铁站", 10), ("公交站", 5)],
    "地铁站": [("家", 10), ("公司", 20), ("公交站", 3)],
    "公交站": [("家", 5), ("地铁站", 3), ("公司", 15)],
    "公司": [("地铁站", 20), ("公交站", 15)],
}

print("最短路径测试：")
print(f"  家 → 公司: {dijkstra(city_map, '家', '公司')} 公里")
# 输出: 15 公里

print(f"  家 → 地铁站: {dijkstra(city_map, '家', '地铁站')} 公里")
# 输出: 10 公里

print(f"  公交站 → 公司: {dijkstra(city_map, '公交站', '公司')} 公里")
# 输出: 15 公里
```

---

## 第五部分：实际应用场景

### 🗺️ 场景1：GPS 导航

计算从起点到终点的最短路线。

```python
# 城市道路网络
roads = {
    "起点": [("A", 5), ("B", 3)],
    "A": [("起点", 5), ("终点", 12)],
    "B": [("起点", 3), ("C", 2), ("终点", 10)],
    "C": [("B", 2), ("终点", 4)],
    "终点": [("A", 12), ("B", 10), ("C", 4)],
}

shortest = dijkstra(roads, "起点", "终点")
print(f"最短距离: {shortest}")
# 输出: 9 (起点 → B → C → 终点)
```

### 🌐 场景2：网络路由

找到服务器之间的最小延迟路径。

```python
# 网络拓扑：服务器之间的延迟
network = {
    "服务器A": [("服务器B", 5), ("服务器C", 10)],
    "服务器B": [("服务器A", 5), ("服务器D", 3)],
    "服务器C": [("服务器A", 10), ("服务器D", 7)],
    "服务器D": [("服务器B", 3), ("服务器C", 7), ("服务器E", 5)],
    "服务器E": [("服务器D", 5)],
}

min_delay = dijkstra(network, "服务器A", "服务器E")
print(f"A 到 E 的最小延迟: {min_delay} ms")
# 输出: 8 ms
```

### ✈️ 场景3：航班规划

找最便宜的中转路线。

```python
# 航线和价格
flights = {
    "北京": [("上海", 500), ("广州", 800)],
    "上海": [("北京", 500), ("东京", 1200), ("广州", 400)],
    "广州": [("北京", 800), ("上海", 400), ("深圳", 100)],
    "东京": [("上海", 1200), ("纽约", 3000)],
    "深圳": [("广州", 100), ("香港", 50)],
    "香港": [("深圳", 50), ("纽约", 2500)],
    "纽约": [("东京", 3000), ("香港", 2500)],
}

min_price = dijkstra(flights, "北京", "纽约")
print(f"北京到纽约最便宜: {min_price} 元")
# 输出: 2900 (北京 → 广州 → 深圳 → 香港 → 纽约)
```

---

## 第六部分：Dijkstra vs 其他算法

### 最短路径算法对比

| 算法 | 适用场景 | 时间复杂度 |
|------|----------|-----------|
| BFS | 无权图最短路径 | O(V + E) |
| **Dijkstra** | 正权图最短路径 ⭐ | O((V+E) log V) |
| Bellman-Ford | 含负权的图 | O(VE) |
| Floyd-Warshall | 全源最短路径 | O(V³) |

### Dijkstra 的限制

❌ **不能处理负权边：**
```
A → B: 5
A → C: 2
C → B: -3

Dijkstra: A → B = 5
实际最短: A → C → B = 2 + (-3) = -1
```

---

## 第七部分：名词解释

### 最小堆 (Min Heap)

**定义：** 每次取出最小值的优先队列。

```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)
heapq.heappop(heap)  # 返回 2
```

### 贪心算法

**定义：** 每一步都选择当前最优解，不考虑全局。

```python
# Dijkstra 的贪心：
# 每次都选择距离最近的未访问顶点
# 期望最终得到全局最短
```

### 松弛操作

**定义：** 通过中间顶点更新距离。

```python
# 如果通过 vertex 能让 neighbor 更近，就更新
if dist[vertex] + weight < dist[neighbor]:
    dist[neighbor] = dist[vertex] + weight
```

### 优先队列

**定义：** 按优先级出队的数据结构。

```python
# Dijkstra 用 (距离, 顶点) 作为元素
# 按距离自动排序，距离小的先出队
heapq.heappush(heap, (10, "A"))
heapq.heappush(heap, (5, "B"))
heapq.heappop(heap)  # 返回 (5, "B")
```

---

## ✅ 小结

1. **Dijkstra** 使用贪心 + 最小堆
2. 每次选择距离最近的未访问顶点
3. **不能处理负权边**
4. 时间复杂度 O((V+E) log V)
5. 适合：GPS导航、网络路由、航线规划

---

_继续学习：下一章「斐波那契数列」_
