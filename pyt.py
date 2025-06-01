import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
from itertools import permutations

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