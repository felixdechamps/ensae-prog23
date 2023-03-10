import numpy as np
import copy
import graphviz
from graphviz import Digraph
class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        self.graph[node1] += [(node2, power_min, dist)]
        self.graph[node2] += [(node1, power_min, dist)]
      
    def path(self,node1,node2):
        visited = []
        visited.append(node1)
        #print(f"visited in exp {visited}")
        neighbors = [self.graph[node][i][0] for i in range(len(self.graph[node]))]
        #print(f"liste des neighbors de {node} = {neighbors}")
        for neighbor in neighbors :
            #print(f"neighbor du node {node} = {neighbor}")
            if neighbor not in visited :
                exploration(self,neighbor)

    def exploration(self,node,visited):
            visited[node-1] = 1
            #print(f"visited in exp {visited}")
            neighbors = [self.graph[node][i][0] for i in range(len(self.graph[node]))]
            #print(f"liste des neighbors de {node} = {neighbors}")
            for neighbor in neighbors :
                #print(f"neighbor du node {node} = {neighbor}")
                if visited[neighbor-1]==0 :
                    self.exploration(neighbor,visited)
    
    def connected_components(self):
        visited = [0]*self.nb_nodes
        components = []
         
        for node in self.nodes :
            #print(f"node ={node}")
            #print(f"visited = {visited}")
            if visited[node-1] ==0 :
                visited = [0]*self.nb_nodes
                self.exploration(node,visited)
                component = []
                for k in range(self.nb_nodes):
                    if visited[k]==1 :
                        component.append(k+1) 
                components.append(component)
                #print(f"MAJ components {components}")
        return components
                
    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def graph_aux(self,power):
        res=copy.deepcopy(self)
        removed_edges=0
        for node in res.nodes :
            edges_list = res.graph[node][:]
            for el in edges_list :
                if el[1] > power :
                    res.graph[node].remove(el)
                    removed_edges+=1/2
        res.nb_edges=res.nb_edges-int(removed_edges)
        return res
    
    def BFS(self,node):
        queue=[node]
        visited = np.array([0]*self.nb_nodes)
        visited[node-1] = 1
        ways = [[node]]
        while len(queue) > 0:
            #print(f"queue begin = {queue}")
            neighbors = np.array([self.graph[queue[0]][i][0] for i in range(len(self.graph[queue[0]]))])
            #print(f"neighbors={neighbors}")
            ways_new = []
            if all(visited[neighbors-1])==1:
                ways_new = ways

            for neighbor in neighbors:
                if visited[neighbor-1]==0 :
                    queue.append(neighbor)
                    visited[neighbor-1]=1
                    for way in ways :
                        if way[-1]==queue[0]:
                            ways_new.append(way+[neighbor])
                        elif way not in ways_new:
                            ways_new.append(way)
            ways=ways_new
            #print(f"ways={ways}")
            del queue[0]
            #print(f"queue end ={queue}")
        return ways

    def get_path_with_power(self, src, dest, power):
        graph_aux=self.graph_aux(power)
        for component in graph_aux.connected_components_set() :
            if {src,dest} & component == {src,dest} :
                ways = self.BFS(src)
                #print(f"ways={ways}")
                ways_to_dest=[]
                for way in ways:
                    #print(f"way ={way}")
                    if dest in way:
                        while way[-1] != dest:
                            del way[-1]
                        ways_to_dest.append(way)
                #print(f"ways_to_dest ={ways_to_dest}")    
                ways_lengths = [len(way) for way in ways_to_dest]
                #print(f"ways_lengths={ways_lengths}")
                return ways_to_dest[np.argmin(ways_lengths)]  
        return None

    def get_paths(self,src, dest):
        for component in self.connected_components_set() :
            if {src,dest} & component == {src,dest} :
                ways = self.BFS(src)
                #print(f"ways={ways}")
                ways_to_dest=[]
                for way in ways:
                    #print(f"way ={way}")
                    if dest in way:
                        while way[-1] != dest:
                            del way[-1]
                        ways_to_dest.append(way)
                return ways_to_dest

    def min_power(self, src, dest):
        #on commence par regarder quelles sont les puissances possibles
        #et on les range par ordre croissant. 
        power_possible = []
        for node in self.nodes :
            for edge in self.graph[node]:
                if edge[1] not in power_possible:
                    power_possible.append(edge[1])            
        power_possible = sorted(power_possible)
        power_min = power_possible[-1]
        res = []
        #on recupere la liste des chemins qui menent de src a dest. 
        ways = self.get_paths(src,dest)
        for way in ways :
            for i in range(len(power_possible)):
                if self.get_path_with_power(src, dest,power_possible[i])==None:
                    i+=1
                else :
                    if power_possible[i]<power_min:
                        res = [way, power_possible[i]]
                        break
        return res
            

        """
        Should return path, min_power. 
        """
    
    def min_power_s2(self, src, dest):
        power_possible = []
        for node in self.nodes :
            for edge in self.graph[node]:
                if edge[1] not in power_possible:
                    power_possible.append(edge[1])            
        power_possible = sorted(power_possible)
        power_min = power_possible[-1]
        res = []
        way = self.get_paths(src,dest)
        for i in range(len(power_possible)):
            if self.get_path_with_power(src, dest,power_possible[i])==None:
                i+=1
            else :
                if power_possible[i]<power_min:
                    res = [way, power_possible[i]]
                    break
        return res



def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    lines = []
    with open(filename, encoding="utf-8") as file :
        for line in file :
            line = line.rsplit()
            lines.append(list(map(int,line)))
    n,m = lines[0][0],lines[0][1]
    nodes = [k for k in range(1,n+1)]
    graph = Graph(nodes)
    graph.nb_edges = m
    for i in range(1,m+1) :
        if len(lines[i]) == 4:
            graph.add_edge(lines[i][0],lines[i][1],lines[i][2], lines[i][3])  
        else :
            graph.add_edge(lines[i][0],lines[i][1],lines[i][2])
    return(graph)

def visual_rpz(graph):
    rpz = graphviz.Digraph(comment=f"Visual display of {graph}", engine='neato')
    for node in graph.nodes :
        rpz.node(str(node))
        print(f"edges to add ={[f'{node}{el[0]}' for el in graph.graph[node]]}")
    for node in graph.nodes :
        neighbors = [el[0] for el in graph.graph[node]]
        for neighbor in neighbors :
            rpz.edge(f'{neighbor}',f'{node}')

        #rpz.edges([f"{node}{el[0]}" for el in graph.graph[node]])
    return rpz

def roads_from_file(filename) :
    lines = []
    with open(filename, encoding="utf-8") as file :
        for line in file :
            line = line.rsplit()
            lines.append(list(map(int,line)))
    nb_roads = lines[0][0]
    roads = []
    #nodes = [k for k in range(1,n+1)]
    #graph = Graph(nodes)
    #graph.nb_edges = m
    for i in range(1,nb_roads+1) :
        #if len(lines[i]) == 4:
            roads.append((lines[i][0],lines[i][1],lines[i][2])) 
    return(roads)

class Union_Find :
    def __init__(self,node):
        self.element = node
        self.parent = self
        self.rank = 0
    def find(self):
        if self.parent == self:
            return self
        else :
            return (self.parent).find()
    @staticmethod
    def union(node1,node2):
        root_node1 = node1.find()
        root_node2 = node2.find()
        if root_node1.rank > root_node2.rank :
            node2.parent = root_node1
        else :
            node1.parent = root_node2
            if root_node1.rank == root_node2.rank :
                root_node2.rank+=1 
def kruskal(g):
    g_mst = Graph(g.nodes)
    edges =[]
    list_set = []
    # on recupere la liste (triee par ordre croissant de puissance) des arrêtes
    for node in g.nodes :
        list_set.append(Union_Find(node))
        for el in g.graph[node] :
            edge=[node]+list(el)
            if edge not in edges and node>el[0]:
                edges.append(edge)
            print(f"edges ={edges}")
    edges = sorted(edges, key=lambda edge : edge[2])
    print(f"sorted edges = {edges}")
    for edge in edges :
        node1,node2 = list_set[edge[0]-1],list_set[edge[1]-1]
        print("ok")
        print(node1.find().element,node2.find().element)
        if node1.find()!=node2.find():
            print("ok")
            print(node1.element,node2.element,edge[2],edge[3])
            g_mst.add_edge(node1.element, node2.element, edge[2], edge[3])
            g_mst.nb_edges+=1
            Union_Find.union(node1, node2)
            print(node1.find().element,node2.find().element)
        print(g_mst)
    return g_mst







    