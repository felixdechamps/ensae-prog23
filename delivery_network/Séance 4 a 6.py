#Let's define the objective function
from graph import Graph, graph_from_file,visual_rpz, roads_from_file,Union_Find,kruskal
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter
import numpy as np
import itertools as it

data_path = "input/"

file_name_roads= "routes.2.in"
file_name_trucks = "trucks.1.in"

file_name_network= "network.2.in"

g = graph_from_file(data_path + file_name_network)

def trucks_from_file(filename) :
    """ INPUT  : a text file routes.x.in 
        OUTPUT : a list of the roads in the file (source, destination, utility)"""
    lines = []
    with open(filename, encoding="utf-8") as file :
        for line in file :
            line = line.rsplit()
            lines.append(list(map(int,line)))
    nb_models = lines[0][0]
    trucks = []
    for i in range(1,nb_models+1) :
            trucks.append((lines[i][0],lines[i][1])) 
    return(trucks)


R = np.array([road[2] for road in roads_from_file(data_path + file_name_roads)])
Cost_models = np.array([truck[1] for truck in trucks_from_file(data_path + file_name_trucks )])
B = 2.5 * 10**9

def f(X) : 
     C = np.array([x[1] for x in X])
     sum(R-C)

def g(X) : #on note la contrainte g
    C = np.array([x[1] for x in X])
    res = [B-sum(C)]
    for i in range(len(X)):
         src = roads[i][0]
         dest = roads[i][1]
         truck_power = trucks[i][0]
         res.append(g.get_path_with_power(src, dest, truck_power)!=None)
    return(res)


   
     
   
combinations = list(it.product(trucks, repeat = len(R)))
for X in combinations : 
     if g(X)[0] >= 0 and all(g(X)[1:])==True and f(x) > fx:
          fx = f(X)
          sol = X
print(X) 
