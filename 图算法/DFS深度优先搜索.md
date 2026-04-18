# DFS 深度优先搜索

_一条路走到黑的搜索策略_

---

## 📖 学习目标

1. 理解 DFS 的基本思想
2. 掌握递归和栈的用法
3. 能解决连通分量和路径问题

---

## 第一部分：什么是 DFS？

### 🎯 DFS 的核心思想

**深度优先搜索（Depth-First Search）：**
- 一条路走到黑，走到死胡同再回溯
- 使用**栈**（或递归）实现
- 适合找所有可能的路径

### 🧭 想象一下

像走迷宫：
- 遇到岔路，随便选一条往前走
- 走到死胡同，退回上一个岔路口
- 选另一条路继续走
- **不撞南墙不回头**

### DFS vs BFS

```
DFS（深度优先）：
    一条路走到黑，再回溯
    
BFS（广度优先）：
    层层推进，先远后近
```

---

## 第二部分：图解 DFS

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

### DFS 搜索过程（递归版）

```
从 S 开始，走到死胡同再回溯

Step 1: S → A（选择第一条路）
Step 2: A → C（C 没有未访问的邻居，死胡同！）
Step 3: 回溯到 A
Step 4: A → D（D 没有未访问的邻居？不对，还有 E！）
Step 5: D → E（找到终点！）

路径：S → A → D → E
```

### DFS 搜索过程（栈版）

```
Stack: [S]        已访问: []
Step 1: 取出 S，压入栈，加入已访问
        Stack: [A, B, F]    已访问: [S]
        访问顺序: S

Step 2: 取出 A，压入栈，加入已访问
        Stack: [B, F, C, D] 已访问: [S, A]
        访问顺序: S, A

Step 3: 取出 C，加入已访问（C 没有未访问邻居）
        Stack: [B, F, D]    已访问: [S, A, C]
        访问顺序: S, A, C

Step 4: 取出 D，加入已访问
        Stack: [B, F, E]    已访问: [S, A, C, D]
        访问顺序: S, A, C, D

Step 5: 取出 E，加入已访问（找到终点！）
        Stack: [B, F]       已访问: [S, A, C, D, E]
        访问顺序: S, A, C, D, E

路径：S → A → C → D → E
```

---

## 第三部分：代码逐行解释

```python
def dfs(graph, start, visited=None):
    """
    深度优先搜索（递归版）

    参数:
        graph: 图的邻接表表示 {顶点: [邻接顶点]}
        start: 起始顶点
        visited: 已访问顶点集合（用于去重）
    返回值:
        从 start 可达的所有顶点
    """

    # -------- 第一次调用时初始化 --------
    if visited is None:
        visited = set()

    # -------- 标记为已访问 --------
    visited.add(start)
    result = [start]  # 记录访问顺序

    # -------- 递归访问所有邻居 --------
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            # 递归 DFS
            result.extend(dfs(graph, neighbor, visited))

    return result
```

### 栈版本（不用递归）

```python
def dfs_iterative(graph, start):
    """
    深度优先搜索（迭代版，用栈）
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()  # 弹出栈顶

        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            # 把邻居压入栈（逆序保证顺序）
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result
```

---

## 第四部分：实际运行代码

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result


# ========================== 测试 ==========================

# 定义一个社交网络
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

print("社交网络 DFS 遍历（从'你'开始）：")
network = dfs(social_network, "你")
print(f"  {network}")
# 输出顺序可能不同，取决于邻接表顺序

print("\n深度探索过程：")
print("  你 → 张三 → 赵六（死胡同，回溯）")
print("  张三 → 钱七 → 郑十（死胡同，回溯）")
print("  继续探索其他分支...")
```

---

## 第五部分：实际应用场景

### 🔍 场景1：走迷宫 - 找所有可能的路径

DFS 可以找到从起点到终点的所有路径。

```python
# 迷宫问题
maze = {
    (0,0): [(0,1), (1,0)],
    (0,1): [(0,0), (1,1)],
    (1,0): [(0,0), (1,1)],
    (1,1): [(0,1), (1,0)],
}

def find_all_paths(maze, start, end):
    """找到所有从起点到终点的路径"""
    visited = set()
    paths = []

    def dfs(current, path):
        if current == end:
            paths.append(path)
            return
        visited.add(current)
        for neighbor in maze.get(current, []):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.remove(current)

    dfs(start, [start])
    return paths
```

### 🔢 场景2：全排列枚举

生成数字的所有排列方式。

```python
def permute(nums):
    """生成所有排列"""
    result = []

    def dfs(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i, num in enumerate(nums):
            if not used[i]:
                used[i] = True
                path.append(num)
                dfs(path, used)
                path.pop()
                used[i] = False

    dfs([], [False] * len(nums))
    return result

print("数字 [1,2,3] 的所有排列：")
print(permute([1, 2, 3]))
# 输出: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

### 🧩 场景3：图的连通分量

判断图中有多少个独立的连通区域。

```python
def count_components(graph):
    """统计连通分量数量"""
    visited = set()
    count = 0

    for vertex in graph:
        if vertex not in visited:
            dfs(graph, vertex, visited)
            count += 1

    return count

# 独立的社交圈
social = {
    "张三": ["李四"],
    "李四": ["张三"],
    "王五": ["赵六"],
    "赵六": ["王五"],
}
print(f"连通分量数: {count_components(social)}")  # 2
```

---

## 第六部分：DFS vs BFS

### 对比

| 对比 | DFS | BFS |
|------|-----|-----|
| 数据结构 | 栈/递归 | 队列 |
| 搜索顺序 | 深度优先 | 广度优先 |
| 最优性 | 不保证最短 | **保证最短** |
| 内存 | 随深度增长 | 随层级增长 |

### 何时使用 DFS？

✅ **适合使用：**
- 找所有可能路径 ⭐
- 连通分量检测
- 排列组合
- 递归能解决的问题

❌ **不适合使用：**
- 最短路径（BFS 更好）

---

## 第七部分：名词解释

### 递归

**定义：** 函数调用自己。

```python
def dfs(graph, start, visited):
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)  # 调用自己
```

### 栈 (Stack)

**定义：** 先进后出（LIFO）的数据结构。

```python
stack = []
stack.append("A")  # 入栈
stack.append("B")
stack.pop()        # 出栈，返回 "B"
stack.pop()        # 出栈，返回 "A"
```

### 回溯

**定义：** 当走到死胡同时，退回上一步。

```python
def dfs(current):
    if is_dead_end(current):
        return  # 回溯
    for next_state in successors(current):
        dfs(next_state)
```

### 连通分量

**定义：** 图中相互连通的顶点集合。

```
连通分量 1: {A, B, C, D}
连通分量 2: {E, F}
```

---

## ✅ 小结

1. **DFS** 使用栈（或递归），一条路走到黑
2. **不撞南墙不回头**，死胡同才回溯
3. **不保证最短路径**
4. 适合：找所有路径、连通分量、排列组合
5. 时间复杂度 O(V + E)

---

_继续学习：下一章「Dijkstra 最短路径」_
