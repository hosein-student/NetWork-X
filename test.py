import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

# ساخت گراف جهت‌دار
G = nx.DiGraph()

# افزودن یال‌ها (با جهت)
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'D'),
    ('D', 'E'),
    ('E', 'A')  # یک حلقه برای جذابیت بیشتر
])


# رسم گراف
pos = nx.spring_layout(G)  # موقعیت گره‌ها برای رسم بهتر
nx.draw(G, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', arrowsize=20, font_size=15)
plt.title("lrng")
plt.show()



G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3)])
print("Nodes:", G.nodes())
print("Edges:", G.edges())
nx.draw(G, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', arrowsize=20, font_size=15)
plt.show()




DG = nx.MultiDiGraph()

# افزودن یال‌های چندگانه جهت‌دار
DG.add_edge('A', 'B')
DG.add_edge('A', 'َB')  # یال دوم با همان جهت
DG.add_edge('B', 'A')  # یال در جهت مخالف
DG.add_edge('B', 'C')
DG.add_edge('C', 'A')

# رسم گراف

nx.draw(DG, with_labels=True, node_color='lightcoral', node_size=2000, edge_color='black', arrows=True, arrowsize=20, font_size=15)
plt.title("گراف چندگانه جهت‌دار (MultiDiGraph)")
plt.show()


# ساخت گراف چندگانه بدون جهت
G = nx.MultiGraph()

# افزودن یال‌های چندگانه بین گره‌ها
G.add_edge('A', 'B')
G.add_edge('A', 'A')  # یال دوم بین A و B
G.add_edge('B', 'C')
G.add_edge('C', 'A')
G.add_edge('A', 'D')
G.add_edge('A', 'B')

# رسم گراف

nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, edge_color='gray', font_size=15)
plt.title("گراف چندگانه (MultiGraph)")
plt.show()




import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

G1 = nx.Graph()
G1.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'A')
])
G2 = nx.Graph()

G2.add_edges_from([
    (1, 2),
    (2, 3),
    (3, 1)
])



result = nx.is_isomorphic(G1, G2)
print("آیا دو گراف ایزومورف هستند؟", result)


G = nx.complete_graph(5)  # گراف کامل با 5 گره

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=14)
plt.title("گراف کامل بدون جهت (K5)")
plt.axis('off')
plt.show()



G = nx.Graph()
G.add_edges_from([
    (1, 2), (2, 3), (3, 4)
])

print("آیا گراف همبند است؟", nx.is_connected(G))


DG = nx.DiGraph()
DG.add_edges_from([
    (1, 2), (2, 3), (3, 1)
])

print("آیا گراف قویاً همبند است؟", nx.is_strongly_connected(DG))
print("آیا گراف ضعیفاً همبند است؟", nx.is_weakly_connected(DG))


# گراف اصلی
G = nx.Graph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'E')
])

# ساخت زیرگراف فقط از گره‌های A، B، C
nodes = ['A', 'B', 'C']
subG = G.subgraph(nodes)

# رسم زیرگراف
nx.draw(subG, with_labels=True, node_color='lightblue', node_size=2000)
plt.title("زیرگراف G")
plt.show()

# ساخت گراف بدون جهت
G = nx.Graph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('A', 'D'),
    ('D', 'E'),
    ('E', 'C')
])

# رسم گراف
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000)
plt.title("گراف بدون جهت")
plt.axis('off')
plt.show()

# پیدا کردن کوتاه‌ترین مسیر بین A و C
path = nx.shortest_path(G, source='A', target='C')
length = nx.shortest_path_length(G, source='A', target='C')

print("کوتاه‌ترین مسیر از A به C:", path)
print("طول مسیر:", length)



# ساخت گراف جهت‌دار وزن‌دار
G = nx.DiGraph()
G.add_weighted_edges_from([
    ('A', 'B', 2),
    ('A', 'D', 1),
    ('B', 'C', 3),
    ('D', 'E', 2),
    ('E', 'C', 1)
])

# رسم گراف
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, arrows=True, arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("گراف جهت‌دار وزن‌دار")
plt.axis('off')
plt.show()



# محاسبه کوتاه‌ترین مسیر از A به C
path = nx.shortest_path(G, source='A', target='C', weight='weight')
length = nx.shortest_path_length(G, source='A', target='C', weight='weight')

print("کوتاه‌ترین مسیر از A به C:", path)
print("طول مسیر:", length)


import networkx as nx

G = nx.DiGraph()
G.add_weighted_edges_from([
    ('A', 'B', 2),
    ('B', 'C', 3),
    ('A', 'D', 1),
    ('D', 'C', 1)
])

path = nx.dijkstra_path(G, source='A', target='C')
length = nx.dijkstra_path_length(G, source='A', target='C')

print("مسیر:", path)
print("طول مسیر:", length)



G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'C'),
    ('C', 'A'),
])

# نمایش درجه‌ها
for node in G.nodes:
    print(f"{node}: in-degree = {G.in_degree(node)}, out-degree = {G.out_degree(node)}")
    



# بی جهت
iG = nx.Graph()
iG.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'A'),
    ('A', 'C')
])

print("مدار اویلری؟", nx.is_eulerian(iG))  # مدار؟
print("دارای مسیر اویلری؟", nx.has_eulerian_path(iG))  # مسیر؟





#جهت دار 
G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'A')
])
nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, arrows=True, arrowsize=20)
plt.show()

# بررسی مدار اویلری
if nx.is_eulerian(G):
    print("✅ مدار اویلری دارد")
    for u, v in nx.eulerian_circuit(G):
        print(f"{u} → {v}")
# بررسی مسیر اویلری
elif nx.has_eulerian_path(G):
    print("✅ مسیر اویلری دارد")
    for u, v in nx.eulerian_path(G):
        print(f"{u} → {v}")
else:
    print("❌ نه مسیر و نه مدار اویلری دارد")






    from itertools import permutations

# ساخت گراف
G = nx.Graph()
G.add_edges_from([
    (1, 2), (2, 3), (3, 4),
    (4, 1), (1, 3)
])

nodes = list(G.nodes)

print("بررسی مسیرها و مدارهای هامیلتیونی:\n")

for path in permutations(nodes):
    # بررسی مسیر هامیلتیونی
    is_path = True
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i+1]):
            is_path = False
            break

    if is_path:
        print("✅ مسیر هامیلتیونی:", path)

        # بررسی اینکه آیا مسیر به رأس اول برمی‌گردد → دور هامیلتیونی
        if G.has_edge(path[-1], path[0]):
            print("🔁  ⮕ این مسیر یک **دور هامیلتیونی** است:", path + (path[0],))



# ساخت گراف
G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('C', 'D'),
    ('D', 'A')
])

# گرفتن ماتریس هم‌سایگی به صورت آرایه NumPy
adj_matrix = nx.to_numpy_array(G, dtype=int, nodelist=sorted(G.nodes()))

# چاپ ماتریس
print("ماتریس هم‌سایگی:")
print(adj_matrix)

# ساخت درخت جهت‌دار
G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('B', 'E'),
    ('C', 'F')
])

# رسم درخت
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, arrows=True, arrowsize=20)
plt.title("درخت جهت‌دار (Directed Tree)")
plt.axis('off')
plt.show()




# ساخت درخت دودویی
G = nx.DiGraph()
edges = [
    ('A', 'B'), ('A', 'C'),
    ('B', 'D'),
    ('C', 'E')
]
G.add_edges_from(edges)

# موقعیت دلخواه برای ظاهر دودویی
pos = {
    'A': (0, 3),
    'B': (-1.5, 2), 'C': (1.5, 2),
    'D': (-2, 1), 'E': (2, 1)
}

# رسم درخت
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True, arrowsize=20)
plt.title("درخت دودویی (Binary Tree)")
plt.axis('off')
plt.show()



#
#Ordered tree

G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'), ('A', 'C'),
    ('B', 'D'), ('B', 'E'),
    ('C', 'F')
])

# موقعیت دستی برای حفظ ترتیب
pos = {
    'A': (0, 2),
    'B': (-1, 1), 'C': (1, 1),
    'D': (-1.5, 0), 'E': (-0.5, 0),
    'F': (1, 0)
}

nx.draw(G, pos, with_labels=True, node_color='lightyellow', node_size=2000, arrows=True, arrowsize=20)
plt.title("Ordered Tree (درخت مرتب)")
plt.axis('off')
plt.show()


#Full binary tree
G = nx.DiGraph()
edges = [
    ('A', 'B'), ('A', 'C'),
    ('B', 'D'), ('B', 'E'),
    ('C', 'F'), ('C', 'G')
]
G.add_edges_from(edges)

# موقعیت گره‌ها برای رسم تمیز
pos = {
    'A': (0, 3),
    'B': (-2, 2), 'C': (2, 2),
    'D': (-3, 1), 'E': (-1, 1),
    'F': (1, 1), 'G': (3, 1)
}

# رسم گراف
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, arrows=True, arrowsize=20)
plt.title("Full Binary Tree (درخت دودویی کامل)")
plt.axis('off')
plt.show()


#Positional tree

G = nx.DiGraph()
edges = [
    ('A', 'B'),  # pos 1
    ('A', 'C'),  # pos 2
    ('A', 'D')   # pos 3
]
G.add_edges_from(edges)

pos = {
    'A': (0, 2),
    'B': (-2, 1),
    'C': (0, 1),
    'D': (2, 1)
}

nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=2000, arrows=True, arrowsize=20)
plt.title("Positional Tree (درخت موقعیتی)")
plt.axis('off')
plt.show()







#گراف جهت دار متصل ضعیف
G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('E', 'C'),
    ('D', 'E')
])

# رسم گراف
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2000, arrows=True, arrowsize=20)
plt.title("گراف جهت‌دار متصل ضعیف")
plt.axis('off')
plt.show()

# بررسی اتصال ضعیف
is_weakly = nx.is_weakly_connected(G)
print("آیا گراف متصل ضعیف است؟", is_weakly)



# ساخت دو گراف جهت‌دار
G1 = nx.DiGraph()
G1.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'), ('D', 'E'), ('E', 'C')])

G2 = nx.DiGraph()
G2.add_edges_from([('X', 'Y'), ('Y', 'Z'), ('X', 'W'), ('W', 'Q'), ('Q', 'Z')])

# بررسی یک‌ریختی
GM = isomorphism.DiGraphMatcher(G1, G2)
print("آیا گراف‌ها یک‌ریخت هستند؟", GM.is_isomorphic())




G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C')
])

print("همبند قوی؟", nx.is_strongly_connected(G))   # False
print("همبند ضعیف؟", nx.is_weakly_connected(G))   # True



#الگوریتم کراسکال

edges = [('A', 'B', 1), ('B', 'C', 2), ('A', 'C', 3)]
nodes = ['A', 'B', 'C']
class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru != rv:
            self.parent[rv] = ru
            return True
        return False

def kruskal(nodes, edges):
    uf = UnionFind(nodes)
    edges.sort(key=lambda x: x[2])
    mst = []

    for u, v, w in edges:
        if uf.union(u, v):
            mst.append((u, v, w))
    return mst

mst = kruskal(nodes, edges)
print("درخت پوشای کمینه:", mst)