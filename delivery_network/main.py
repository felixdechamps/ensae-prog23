from graph import Graph, graph_from_file,visual_rpz, roads_from_file,UnionFind,kruskal
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter


data_path = "input/"

file_name_network= "network.2.in"

g = graph_from_file(data_path + file_name_network)

g_mst = kruskal(g)
#####      Question 14  ####
def get_parents_and_depths(g):
    """INPUT : a graph g
    OUTPUT : dictionaries one containing the parent of each node and 
    another called depths containing the depth of each node in the tree"""
    nodes=g.nodes
    parents=dict(((node, (None,None))for node in nodes)) #dictionary of parents
    depths=dict(((node,0)for node in nodes)) #dictionary of the depths of each node
    parents[nodes[0]]=(nodes[0],0) #the first node of the tree is chosen as the root

    def exploration(g,node,depth):
        """ Underfunction implementing a recursive BFS"""
        for edge in g.graph[node]:
            neighbor= edge[0]
            if parents[neighbor] == (None,None): # if the parent is not already visited
                parents[neighbor] = (node,edge[1]) # update of the parent and the depth
                depths[neighbor] = depth
                exploration(g,neighbor,depth+1) # we go explore the neighbors and increase the depth
    
    exploration(g,nodes[0],1)

    return parents, depths

def new_min_power(parents, depths, src,dest): 
    power=0
    depth_src=depths[src]
    depth_dest=depths[dest]
    node_1 = 0
    node_2 = 0
    if depth_src>depth_dest:# this step makes sure that the node_1 is always the deepest 
        node_1,node_2 = src,dest
    else:
        node_1,node_2 = dest,src
    while depths[node_1] != depths[node_2]: #  as long as the depths of node_1 and node_2 aren't equal :
        power = max(parents[node_1][1],power) #   we update the power with the power wich is required to go up from node_1 to his parent (= parents[node_1][1])
        node_1 = parents[node_1][0] # we move up the node_1's parent
    #   Once node_1 reached the depths of node_2, move up in the tree, simultaneously from node_1 and node_2, until node_1 = node_2
    while node_1 != node_2: 
        power = max(parents[node_1][1],power) # We update the power when node_1 moves up to his parent
        power = max(parents[node_2][1],power) # We update the power when node_2 moves up to his parent
        node_1,node_2 = parents[node_1][0],parents[node_2][0] # We move up simultaneously in the graph
    return power

###     Question 15     ###

def routes_min_power(routes_x_in, network_x_in, routes_x_out):
    g= graph_from_file(network_x_in)
    with open(routes_x_in, "r") as file:
        otherfile = open(routes_x_out,"r+")
        nb_roads=list(map(int, file.readline().split()))[0]
        g_mst = kruskal(g)
        parents, depths = dictionnaries(g_mst)
        for _ in range (nb_roads):
            src, dest, cost = list(map(int, file.readline().split()))
            power = new_min_power(parents, depths, src, dest)
            if len(otherfile.readlines())!=nb_roads:
                    otherfile.write(str(power)+"\n")
        otherfile.close()














