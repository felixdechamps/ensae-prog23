import sys 
sys.path.append("delivery_network/")

import unittest
from graph import Graph, graph_from_file

class Test_graph_from_file_with_distance(unittest.TestCase) :
    def test_network04(self):
        g = graph_from_file("input/network.04.in")
        self.assertEqual(g.graph[2], [(3,4,3),(1,4,89)])

if __name__ == '__main__':
    unittest.main()