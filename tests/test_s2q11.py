import sys 
sys.path.append("delivery_network/")

import unittest
from graph import Graph, graph_from_file,Union_Find,kruskal

class Test_Minimum_Spanning_Tree(unittest.TestCase):
    def test_network03(self):
        g = graph_from_file("input/network.03.in")
        g_mst = kruskal(g)
        self.assertEqual(g_mst.graph, {1: [(2, 10, 1)], 2: [(3, 4, 1), (1, 10, 1)], 3: [(2, 4, 1), (4, 4, 1)], 4: [(3, 4, 1)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []})
        self.assertEqual(g_mst.nb_edges, 3)
    def test_network02(self):
        g = graph_from_file("input/network.02.in")
        g_mst = kruskal(g)
        self.assertEqual(g_mst.graph, {1: [(4, 4, 1)], 2: [(3, 4, 1)], 3: [(2, 4, 1), (4, 4, 1)], 4: [(3, 4, 1), (1, 4, 1)], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []})
        self.assertEqual(g_mst.nb_edges, 3)

if __name__ == '__main__':
    unittest.main()
