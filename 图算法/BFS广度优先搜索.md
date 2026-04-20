# BFS 广度优先搜索

_层层推进的搜索策略_

---

## 📖 学习目标

1. 理解 BFS 的基本思想

![bfs](bfs.png)

2. 掌握队列的使用
3. 能解决最短路径问题

---

## 第一部分：什么是 BFS？

### 🎯 BFS 的核心思想

**广度优先搜索（Breadth-First Search）：**
- 像波浪一样，从起点向外层层扩散
- 先搜索距离近的，再搜索距离远的
- **队列**是 BFS 的核心数据结构

### 🌊 想象一下

想象你往水里扔一颗石子：
- 水波从中心向外扩散
- 第1圈先碰到，然后第2圈、第3圈...
- **先遇到的，就是距离最近的**

### BFS vs DFS

```
BFS（广度优先）：
    层层推进，先远后近
    
DFS（深度优先）：
    一条路走到黑，再回溯
```

---

## 第二部分：图解 BFS

### 迷宫搜索示例

```
从起点 S 出发，找到终点 E

    S ─── A ─── C
    │              │
    B ─── D ───   │
    │              │
    F ─────────── E

邻居关系：
S: [A, B, F]
A: [S, C, D]
B: [S, D]
C: [A]
D: [A, B, E]
F: [S, E]
E: [D, F]
```

### BFS 搜索过程

```
Queue: [S]        已访问: []
Step 1: 取出 S，加入队列
        Queue: [A, B, F]    已访问: [S]

Step 2: 取出 A，加入队列
        Queue: [B, F, C, D] 已访问: [S, A]

Step 3: 取出 B，加入队列
        Queue: [F, C, D]    已访问: [S, A, B]

Step 4: 取出 F，加入队列
        Queue: [C, D, E]    已访问: [S, A, B, F]
                          ↑ 发现终点 E！

路径：S → F → E（最短路径！）
```

---

## 第三部分：代码逐行解释

```python
from collections import deque

def bfs(graph, start):
    """
    广度优先搜索

    参数:
        graph: 图的邻接表表示 {顶点: [邻接顶点]}
        start: 起始顶点
    返回值:
        从 start 可达的所有顶点
    """

    # -------- 初始化 --------
    visited = {start}    # 已访问的顶点集合
    queue = deque([start])  # 用队列存储待访问的顶点
    result = []            # 存储搜索结果

    # -------- BFS 主循环 --------
    while queue:
        # -------- 取出队首顶点 --------
        vertex = queue.popleft()
        result.append(vertex)

        # -------- 遍历所有邻居 --------
        for neighbor in graph.get(vertex, []):
            # 如果邻居还没访问过
            if neighbor not in visited:
                visited.add(neighbor)    # 标记为已访问
                queue.append(neighbor)   # 加入队列等待访问

    return result
```

---

## 第四部分：实际运行代码

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result


# ========================== 测试 ==========================

# 定义一个简单的社交网络
social_network = {
    "你": ["张三", "李四", "王五"],
    "张三": ["你", "赵六", "钱七"],
    "李四": ["你", "孙八"],
    "王五": ["你", "周九"],
    "赵六": ["张三"],
    "钱七": ["张三", "郑十"],
    "孙八": ["李四"],
    "周九": ["王五"],
    "郑十": ["钱七"],
}

print("社交网络 BFS 遍历（从'你'开始）：")
network = bfs(social_network, "你")
print(f"  {network}")
# 输出: ['你', '张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '郑十']

print("\n逐层展开：")
print("  第1层（直接认识）: 张三, 李四, 王五")
print("  第2层（朋友的朋友）: 赵六, 钱七, 孙八, 周九")
print("  第3层: 郑十")
```

---

## 第五部分：实际应用场景

### 👥 场景1：社交网络好友推荐

通过共同好友推荐新朋友。

```python
# 你认识 A，A 认识 B，B 认识 C
# BFS 找到你所有 2 度好友（朋友的朋友）
social = {
    "你": ["A", "B"],
    "A": ["你", "C", "D"],
    "B": ["你", "E", "F"],
    "C": ["A", "G"],
    "D": ["A"],
    "E": ["B", "H"],
}
friends_of_friends = bfs(social, "你")
print(f"  2度好友推荐: C, D, E, F")
```

### 🗺️ 场景2：迷宫最短路径

在无权图中找最短路径。

```python
# 迷宫：找到从起点到终点的最短路径
maze = {
    (0,0): [(0,1), (1,0)],
    (0,1): [(0,0), (0,2), (1,1)],
    # ... 更多格子
}

def bfs_shortest_path(maze, start, end):
    visited = {start}
    queue = deque([(start, [start])])  # (位置, 路径)

    while queue:
        (vertex, path) = queue.popleft()
        if vertex == end:
            return path

        for neighbor in maze.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
```

### 📁 场景3：文件系统层级遍历

按层级遍历文件夹。

```python
file_system = {
    "根目录": ["文件夹A", "文件夹B", "文件1.txt"],
    "文件夹A": ["文件2.txt", "文件3.txt"],
    "文件夹B": ["文件夹C", "文件4.txt"],
    "文件夹C": ["文件5.txt"],
}

print("按层级遍历：")
print("  第1层: 文件夹A, 文件夹B, 文件1.txt")
print("  第2层: 文件2.txt, 文件3.txt, 文件夹C, 文件4.txt")
print("  第3层: 文件5.txt")
```

---

## 第六部分：BFS vs 其他搜索

### BFS 的特点

| 特点 | 说明 |
|------|------|
| 数据结构 | 队列（先进先出）|
| 时间复杂度 | O(V + E) |
| 空间复杂度 | O(V) |
| 特点 | **保证找到最短路径** |

### BFS vs DFS

| 对比 | BFS | DFS |
|------|-----|-----|
| 数据结构 | 队列 | 栈/递归 |
| 搜索顺序 | 层层推进 | 一条路走到黑 |
| 最优性 | **最短路径** ⭐ | 不保证 |
| 内存 | 随层级增长 | 随深度增长 |

### 何时使用 BFS？

✅ **适合使用：**
- 找最短路径 ⭐
- 层次遍历
- 找所有可达节点

---

## 第七部分：名词解释

### 邻接表

**定义：** 用字典表示图的连接关系。

```python
graph = {
    "A": ["B", "C"],  # A 连接 B 和 C
    "B": ["A", "D"],  # B 连接 A 和 D
    "C": ["A"],        # C 只连接 A
    "D": ["B"],        # D 只连接 B
}
```

### 队列 (Queue)

**定义：** 先进先出（FIFO）的数据结构。

```python
from collections import deque

queue = deque()
queue.append("A")  # 入队
queue.append("B")
queue.popleft()    # 出队，返回 "A"
queue.popleft()    # 出队，返回 "B"
```

### 时间复杂度 O(V + E)

**定义：** V 是顶点数，E 是边数。

```
O(V + E) = O(顶点数) + O(边数)
```

### 最短路径

**定义：** 从起点到终点经过边数最少的路径。

```
BFS 保证找到的路径是最短的（边数最少）
```

---

## ✅ 小结

1. **BFS** 使用队列，层层推进
2. **先访问近的，再访问远的**
3. **保证找到最短路径**
4. 时间复杂度 O(V + E)
5. 适合：最短路径、层次遍历

---

_继续学习：下一章「DFS 深度优先搜索」_
