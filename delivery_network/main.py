from graph import Graph, graph_from_file


data_path = "input/"
file_name = "network.00.in"

g = graph_from_file(data_path + file_name)
print(g)
print(g.connected_components_set())
print(g.get_path_with_power(6,4,10))
