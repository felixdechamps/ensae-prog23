# This will work if ran from the root folder.
import sys 
sys.path.append("delivery_network")

from graph import Graph, graph_from_file,visual_rpz, roads_from_file,UnionFind,kruskal
import time
from time import perf_counter
import unittest   # The test framework

class Test_Kruskal(unittest.TestCase):
    def test_network02(self):
        g = graph_from_file("input/network.02.in")
        g_mst = kruskal(g)
        self.assertEqual(g_mst.graph, {1: [(4, 4, 1)], 2: [(3, 4, 1)], 3: [(2, 4, 1), (4, 4, 1)], 4: [(1, 4, 1), (3, 4, 1)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []} )

    def test_network03(self):
        g = graph_from_file("input/network.03.in")
        g_mst = kruskal(g)
        self.assertEqual(g_mst.graph,{1: [(2, 10, 1)], 2: [(3, 4, 1), (1, 10, 1)], 3: [(2, 4, 1), (4, 4, 1)], 4: [(3, 4, 1)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []} )
    
    def test_network03(self):
        g = graph_from_file("input/network.04.in")
        g_mst = kruskal(g)
        self.assertEqual(g_mst.graph,{1: [(2, 4, 89)], 2: [(3, 4, 3), (1, 4, 89)], 3: [(2, 4, 3), (4, 4, 2)], 4: [(3, 4, 2)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []})