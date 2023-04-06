#Let's define the objective function
from graph import Graph, graph_from_file,visual_rpz, roads_from_file,UnionFind,kruskal
from main import get_parents_and_depths,new_min_power
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter
import numpy as np
import itertools as it

data_path = "input/"

file_name_roads= "routes.1.in"
file_name_trucks = "trucks.1.in"

file_name_network= "network.1.in"

g = graph_from_file(data_path + file_name_network)
roads = roads_from_file(data_path + file_name_roads)


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

trucks = trucks_from_file(data_path + file_name_trucks)

R = np.array([road[2] for road in roads_from_file(data_path + file_name_roads)])
Cost_models = np.array([truck[1] for truck in trucks_from_file(data_path + file_name_trucks )])
B = 2.5 * 10**9

def f(X) : 
     C = np.array([x[1] for x in X])  #C est le vecteur avec que les coûts des camions 
     sum(R-C)

def g(X) : #on note la contrainte g
    C = np.array([x[1] for x in X])
    res = [sum(C)-B]
    for i in range(len(X)):
         src = roads[i][0]
         dest = roads[i][1]
         truck_power = X[i][0]
         res.append(g.get_path_with_power(src, dest, truck_power)!=None)
    return(res)


   # res est un vecteur composé d'une première fonction contrainte budgétaire (qui doit être négative) et d'une suite de booléens qui doivent tous être True 

     
def force_brute() : 
    combinations = list(it.product(trucks, repeat = len(R)))
    fx = 0 
    for X in combinations : 
        if g(X)[0] <= 0 and all(g(X)[1:])==True and f(X) > fx:
            fx = f(X)
            sol = X
    return(X)
#print(force_brute())
#####   KNAPSACK PROBLEM    ####
def W(g,trucks,roads):
    g_mst = kruskal(g)
    res = []
    parents = get_parents_and_depths(g_mst)[0]
    depths = get_parents_and_depths(g_mst)[1]

    for truck in trucks :
        res_int = 0
        for road in roads :
            if new_get_power(parents, depths, road[0], road[1])<=truck[0]:
                res_int+=truck[1]
        res.append(res_int)
    return res



def knapsack_greedy(g,trucks,roads, B):
    # Initialisation des variables
    N = len(trucks)
    M = len(roads)
    remaining_budget = B
    W = W(g,trucks,roads)
    P =[road[2] for road in roads]
    assigned_trips = set()

    # Tri des camions par coût croissant
    sorted_trucks = sorted(trucks[1:], key=lambda i: trucks[1:][i][1])

    # Boucle sur les camions
    for i in sorted_trucks:
        # Sélection des trajets non affectés ayant un poids total inférieur ou égal à la capacité restante du camion
        feasible_trips = [(j, P[j]/W[j]) for j in range(M) if j not in assigned_trips and W[j] <= remaining_budget]
        if feasible_trips:
            # Sélection du trajet avec le plus grand rapport qualité-prix
            selected_trip = max(feasible_trips, key=lambda x: x[1])[0]
            # Affectation du trajet au camion i
            assigned_trips.add(selected_trip)
            remaining_budget -= W[selected_trip]

    # Calcul du profit total
    total_profit = sum(P[j] for j in assigned_trips)

    return total_profit
print(knapsack_greedy(g, trucks, roads, B))
