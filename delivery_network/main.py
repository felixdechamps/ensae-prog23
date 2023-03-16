from graph import Graph, graph_from_file,visual_rpz, roads_from_file,Union_Find,kruskal
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter


data_path = "input/"

file_name_network= "network.2.in"


g = graph_from_file(data_path + file_name_network)

g_mst = kruskal(g)
print("ok")
file_name_roads = "routes.2.in"

t_start = perf_counter()

for road in roads_from_file(data_path + file_name_roads) :
    g_mst.min_power(road[0], road[1])
t_stop = perf_counter()

print("Time to do this", t_stop - t_start,"secondes")

#soit en moyenne 2.6/140 = 19 ms environ par trajet. """
