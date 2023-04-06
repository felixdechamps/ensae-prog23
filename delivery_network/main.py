from graph import Graph, graph_from_file,visual_rpz, roads_from_file,UnionFind,kruskal
import graphviz
from graphviz import Graph,Source
import time
from time import perf_counter


data_path = "input/"

file_name_network= "network.1.in"


g = graph_from_file(data_path + file_name_network)
#print(g)
g_mst = kruskal(g)

print(g_mst)

def get_parents_and_depths(g):
    nodes=g.nodes
    parents=dict(((node, (-1,-1))for node in nodes)) #dictionnaire des parents
    print(f"parent = {parents}")
    depths=dict(((node,0)for node in nodes)) #dictionnaire des profondeurs de chaque noeud
    print("depths = ",depths)
    parents[nodes[0]]=(nodes[0],0) #on choisit le premier noeud de l'arbre en guise de racine
    '''
    On définit une sous fonction dico qui permet de construire, par un parcours en profondeur récursif, les 
    dictionnaires parents et depths. Pour chaque noeud, l'on parcourt ses voisins pour lesquels on met à 
    jour le parent et la profondeur et l'on rappelle la fonction sur chaque voisin. 
    '''
    def exploration(g,node,depth):
        for edge in g.graph[node]:
            ngb= edge[0]
            if parents[ngb]==(-1,-1):
                parents[ngb]=(node,edge[1])
                depths[ngb]=depth
                exploration(g,ngb,depth+1)
    
    exploration(g,nodes[0],1)

    return parents, depths
print(get_parents_and_depths(g_mst))
'''
Complexité de l'algorithme:
On note n le nombre de noeud de l'arbre.
Dans cet algorithme on parcourt chaque noeud du graphe, jusqu'à ce qu'ils soient tous marqués. Pour chacun 
des noeuds, on exécute une boucle for sur l'ensemble des voisins des noeuds. On en déduit une complexité 
en O(n*avg(nb_ngb)) où avg(nb_ngb) est le nombre moyen de voisins par noeud du graphe. S'agissant d'un arbre
il est raisonnable de considérer que avg(ng_ngb) est petit, on peut donc approximer la complexité par O(n)
'''
'''
On écrit à présent une fonction retournant la puissance nécessaire pour effectuer le trajet dans l'arbre.
On met en oeuvre la méthode expliquée au début de la question. (NB: s'agissant d'un arbre, le trajet entre
deux noeuds est unique, la puissance renvoyée correspond donc bien ici à la puissance nécessaire pour effectuer
le trajet.)
'''
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


'''
Dans cette fonction, on parcourt au maximum l'ensemble des noeuds de l'arbre, à travers les boucles while.
De plus, on effectue des opérations bornées au sein des boucles. D'où une complexité dans le pire des cas en 
O(n). 
Pour déterminer la puissance minimale pour parcourir un chemin au sein d'un graphe, on effectue successivement
les fonctions dictionnaries et new_get_power, ayant chacune dans le meilleur des cas une complexité en 
O(n). D'où une complexité linéaire en O(n) pour déterminer la puissance minimale d'un trajet. 
'''

'''
Question 15
'''
def test_time(filename1, filename2, filename3):
    g= graph_from_file(filename2)
    with open(filename1, "r") as file:
        nb_trajet=list(map(int, file.readline().split()))[0]
        tot_time=0
        start=time.perf_counter()
        new_g=kruskal(g)
        parents, depths=dictionnaries(new_g)
        for _ in range (nb_trajet):
            src, dest, cost=list(map(int, file.readline().split()))
            pow=new_get_power(parents,depths, src, dest)
        stop=time.perf_counter()
        tot_time+=(stop-start)
    return tot_time

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



file_name_roads = "routes.1.in"
parents = get_parents_and_depths(g_mst)[0]
depths = get_parents_and_depths(g_mst)[1]
t_start = perf_counter()

for road in roads_from_file(data_path + file_name_roads) :
    new_min_power(parents, depths, road[0], road[1])
t_stop = perf_counter()

print("Time to do this", t_stop - t_start,"secondes")

#soit en moyenne 2.6/140 = 19 ms environ par trajet. """













