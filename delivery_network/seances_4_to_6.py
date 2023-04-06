from graph import Graph, graph_from_file,visual_rpz, roads_from_file,UnionFind,kruskal
from main import get_parents_and_depths,new_min_power
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter
import numpy as np
import itertools as it

###     QUESTION 18     ###

# filnames
data_path = "input/"
file_name_roads= "routes.2.in"
file_name_trucks = "trucks.2.in"
file_name_network= "network.2.in"

def trucks_from_file(filename) :
    """ INPUT  : a text file trucks.x.in 
        OUTPUT : a list of the trucks in the file (power, cost)"""
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

g = graph_from_file(data_path + file_name_network)
roads = roads_from_file(data_path + file_name_roads)
trucks = trucks_from_file(data_path + file_name_trucks)
g_mst = kruskal(g)
parents, depths = get_parents_and_depths(g_mst)[0],get_parents_and_depths(g)[1]

#### BRUTE FORCE ####
R = np.array([road[2] for road in roads_from_file(data_path + file_name_roads)])
Cost_models = np.array([truck[1] for truck in trucks_from_file(data_path + file_name_trucks )])
B = 2.5*10**9

def f(X) : 
     C = np.array([x[1] for x in X])  # C is a vector containing the cost of the trucks 
     sum(R-C)

def h(X) : # we name h the constraint
    C = np.array([x[1] for x in X])
    res = [sum(C)-B]
    for i in range(len(X)):
         src = roads[i][0]
         dest = roads[i][1]
         truck_power = X[i][0]
         res.append(g.get_path_with_power(src, dest, truck_power)!=None)
    return(res)


# res is a vector composed of a first budget constraint function (which must be negative)
# and a sequence of Booleans which must all be True

     
def force_brute() : 
    """the idea of this method is to test all possible solutions and choose the best one only the fact 
    of having to calculate all the combinations means that this method can only work for small only work 
    for small graphs"""
    combinations = list(it.product(trucks, repeat = len(R)))
    fx = 0 
    for X in combinations : 
        #we look at whether the solution satisfies the constraint h
        if h(X)[0] <= 0 and all(h(X)[1:])==True and f(X) > fx:
            fx = f(X)
            sol = X
    return(X)

#####   KNAPSACK PROBLEM    ####

def greedy(trucks, roads,B):
    # greedy algorithm
    sorted_trucks = sorted(trucks, key=lambda x: x[0]) # we sort the trucks by increasing cost
    selected_trucks = [] # list of the trucks selected for each travel
    for road in roads:
        for truck in sorted_trucks:
            #if truck[1] <= B and truck[1] >= road[2] and truck[0]>=new_min_power(parents, depths, road[0], road[1]) :
            if truck[1] <= B and truck[1] >= road[2] :
                selected_trucks.append(trucks.index(truck))
                B -= truck[1]
                break

    # we compute the total profit and the list of the selected trucks
    total_profit = sum([roads[i][2] for i in range(len(selected_trucks))])
    selected_trucks = [(i, trucks[i][1], trucks[i][0]) for i in selected_trucks]
    return (f"total_profit = {total_profit} \n selected_trucks = {selected_trucks}")
    
print(greedy(trucks,roads,B))