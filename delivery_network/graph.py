import numpy as np
import copy
import graphviz
from graphviz import Graph
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
        self.edges = []
        self.powers = set()

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
        self.graph[node1] += [(node2, power_min, dist)]# we had the node2 to the neighbors of the node1 and vice versa. 
        self.graph[node2] += [(node1, power_min, dist)]


    def DFS(self,node,visited):
        """Depth-First Search algorithm"""
        visited[node-1] = 1 #this node in now marked as visited
        neighbors = [self.graph[node][i][0] for i in range(len(self.graph[node]))] #list of the node's neighbors
        for neighbor in neighbors :
            if visited[neighbor-1]==0 : # if the note hasn't been visited before we apply the DFS to it
                self.DFS(neighbor,visited)
    
    def connected_components(self):
        """method giving the connected components ef the graph in the format of a list of lists"""
        visited = [0]*self.nb_nodes
        components = []
         
        for node in self.nodes :

            if visited[node-1] == 0 :
                visited = [0]*self.nb_nodes
                self.DFS(node,visited)
                component = []
                for k in range(self.nb_nodes):
                    if visited[k]==1 :
                        component.append(k+1) 
                components.append(component)
        return components
                
    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def graph_aux(self,power):
        """
        Input : a graph
        Output : an auxiliary graph containing only the egdes 
        with a power inferior to the argument "power
        """
        res=copy.deepcopy(self) # we make a copy of the initial graph so that it won't be modified
        removed_edges=0 #stores the number of removed edges
        for node in res.nodes :
            edges_list = res.graph[node][:] #edges_list = list of the edges of the node (neigbors, power, dist)
            for el in edges_list :
                if el[1] > power :
                    res.graph[node].remove(el) #we remove the egdes which requires to much power
                    removed_edges+=1/2
        res.nb_edges=res.nb_edges-int(removed_edges) #update of the number of edges of the auxiliary graph 
        return res
    
    def BFS(self,node):
        """Breadth-First search algorithm
        INPUT : a starting node
        OUTPUT : a list of ways from the node in input to cover the entire graph
        """
        queue=[node]#initialisation of a queue
        visited = np.array([0]*self.nb_nodes)#list of the visited nodes
        visited[node-1] = 1#add the starting node to the list of the nodes already visited
        ways = [[node]]#list of the different ways starting from the input node
        while len(queue) > 0:
            #list of the neighbors of the queue's first element
            neighbors = np.array([self.graph[queue[0]][i][0] for i in range(len(self.graph[queue[0]]))]) 
            ways_new = []# new list to store the ways
            if all(visited[neighbors-1])==1: #if all the neighbor's have already been visited
                ways_new = ways
            for neighbor in neighbors:
                if visited[neighbor-1]==0 : #if the neighbor has not been already visited we add it to the queue
                    queue.append(neighbor)
                    visited[neighbor-1]=1# we mark this neighbor as a visited node
                    for way in ways : #this is for updating the list of the ways 
                        if way[-1]==queue[0]: #if the last element of the way is the current first element of the queue
                            ways_new.append(way+[neighbor]) # we  create a new way finishing by the neighbor (it's like growing a branch with a new leaf).
                        elif way not in ways_new: 
                            ways_new.append(way)
            ways=ways_new
            
            del queue[0] #we update the queue by deleting the first element
            
        return ways

    def get_path_with_power(self, src, dest, power):
        """ INPUT : a graph, une source node, a destination node and the power of the truck which wants 
            to cover the way
            OUTPUT : True and the shortest way between src and dest if it exists or None if it doesn't exists  """
        graph_aux=self.graph_aux(power) #we use an auxiliary graph where all the power superior to 'power' have been removed
        for component in graph_aux.connected_components_set() :
            if {src,dest} & component == {src,dest} : #tests if src and dest belong to the same connected component, ie wheter a path between them exists.
                ways = self.BFS(src)
                ways_to_dest=[] #we are going to select the ways leading to src among the ways starting from src
                for way in ways:
                    if dest in way:
                        while way[-1] != dest: #we shorten the way until the last element is the destination
                            del way[-1]
                        ways_to_dest.append(way)   
                ways_lengths = [len(way) for way in ways_to_dest]
                return ways_to_dest[np.argmin(ways_lengths)]  # we return the shorttest path among the ways to the destination
        return None

    def get_paths(self,src, dest):
        """ Similar to the precedeing function, but it forget the power 
            and only return all the ways between src and dest"""
        for component in self.connected_components_set() :
            if {src,dest} & component == {src,dest} :
                ways = self.BFS(src)
                ways_to_dest=[]
                for way in ways:
                    if dest in way:
                        while way[-1] != dest:
                            del way[-1]
                        ways_to_dest.append(way)
                return ways_to_dest

    def binary_search(self,powers,src,dest) :
        list_powers = sorted(list(powers))
        n = len(list_powers)
        inf = 0
        sup = n
        middle = n//2
        if self.get_path_with_power(src, dest,list_powers[middle]) == None :
            return self.binary_search(list_powers[middle:],src,dest)
        else :
            if self.get_path_with_power(src, dest,list_powers[middle-1])==None :
                return list_powers[middle]
            if len(list_powers)==2 and self.get_path_with_power(src, dest,list_powers[middle-1])!=None:
                return list_powers[middle-1] 
            return self.binary_search(list_powers[:middle+1],src,dest)


    def min_power(self, src, dest):
        """INPUT : Source and destination of the way
            OUTPUT : path and min power to go through that path """
        #on commence par regarder quelles sont les puissances possibles
        #et on les range par ordre croissant. 
        """
        power_possible = []
        for node in self.nodes :
            for edge in self.graph[node]:
                if edge[1] not in power_possible:
                    power_possible.append(edge[1])            
        power_possible = sorted(power_possible)
        """
        
        res = []
        #on recupere la liste des chemins qui menent de src a dest. 
        ways = self.get_paths(src,dest)
        power_min = max(self.powers)
        list_powers = sorted(self.powers)
        for way in ways :
            if self.binary_search(list_powers,src,dest)<=power_min :
                power_min = self.binary_search(list_powers,src,dest)
                res = [way, power_min]
        return res
            

        """
        Should return path, min_power. 
        """
    def min_power_s2(self, src, dest):
        power_possible = []
        for node in self.nodes :
            for edge in self.graph[node] :
                if edge[1] not in power_possible :
                    power_possible.append(edge[1])            
        power_possible = sorted(power_possible)
        res = []
        way = self.get_paths(src,dest)
        for i in range(len(power_possible)):
            if self.get_path_with_power(src, dest,power_possible[i])==None:
                i+=1
            else :
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
    with open(filename, encoding="utf-8") as file : # we open the text file cointaining the network
        for line in file :
            line = line.rsplit()
            lines.append(list(map(int,line)))
    n,m = lines[0][0],lines[0][1] #update of the number of node n and of the number of edges m
    nodes = [k for k in range(1,n+1)]#create the list of nodes
    graph = Graph(nodes)#create an empty graph
    graph.nb_edges = m
    for i in range(1,m+1) : # add the edges to the graph, checking if there is a given distance or not
        if len(lines[i]) == 4:
            graph.add_edge(lines[i][0],lines[i][1],lines[i][2], lines[i][3])
            graph.edges.append((lines[i][0],lines[i][1],lines[i][2], lines[i][3]))
            graph.powers.add(lines[i][2])
        else :
            graph.add_edge(lines[i][0],lines[i][1],lines[i][2],1)
            graph.edges.append((lines[i][0],lines[i][1],lines[i][2],1))
            graph.powers.add(lines[i][2])
    return(graph)

def visual_rpz(g,comment = "Graphe", view = True):
    """ INPUT : a graph g 
        OUTPUT : a graphical representation of the graph g """

    rpz = graphviz.Graph(comment = comment, engine='neato')
    for node in g.nodes :
        rpz.node(f"{node}", str(node))
        
    for source, destinations in g.graph.items() :
        for destination, power, dist in destinations :
            if source < destination :
                rpz.edge(str(source), str(destination), label = f"{power}, {dist}")
    rpz.render(filename = f"doctest-output/{comment}.gv", cleanup = True, view = view )

def roads_from_file(filename) :
    """ INPUT  : a text file routes.x.in 
        OUTPUT : a list of the roads in the file (source, destination, utility)"""
    lines = []
    with open(filename, encoding="utf-8") as file :
        for line in file :
            line = line.rsplit()
            lines.append(list(map(int,line)))
    nb_roads = lines[0][0]
    roads = []
    for i in range(1,nb_roads+1) :
            roads.append((lines[i][0],lines[i][1],lines[i][2])) 
    return(roads)

class Union_Find :
    """ class creating the Union-Find data structure"""
    def __init__(self,node):
        """ Equivalent of the MakeSet function, create a set with 
            a single element pointing on itself, with rank 0"""
        self.element = node
        self.parent = self
        self.rank = 0
    def find(self):
        """ INPUT : an union-find data
            OUTPUT : the root of the union-find data in the input"""
        if self.parent == self:
            return self
        else :
            return (self.parent).find()
    @staticmethod
    def union(node1,node2):
        """ INPUT : two elements with an union-find data type
            OUTPUT : None, but it "joins" the trees containing 
                     node1 and node2 by updating the parents and rank """
        root_node1 = node1.find()
        root_node2 = node2.find()
        if root_node1.rank > root_node2.rank :
            node2.parent = root_node1
        else :
            node1.parent = root_node2
            if root_node1.rank == root_node2.rank :
                root_node2.rank+=1 
def kruskal(g):
    """INPUT :  a graph g
       OUTPUT : a graph g_mst with correspond to the minimal spannig tree of the graph g"""
    g_mst = Graph(g.nodes)
    edges =[]
    list_set = [] # list_set will contain the union-find version of the graph
    # we get the list in the increasing order of the power of the different edges
    for node in g.nodes :
        list_set.append(Union_Find(node))
        """
        for el in g.graph[node] :
            edge=[node]+list(el) # the edges as the format [node1,node2,power,dist]
            if edge not in edges and node>el[0]:
                edges.append(edge)
                """
    #edges = sorted(edges, key=lambda edge : edge[2])
    for edge in g.edges :
        node1,node2 = list_set[edge[0]-1],list_set[edge[1]-1]
        if node1.find()!=node2.find(): # if the roots of the nodes aren't the same, we add the edge to g_mst
            g_mst.add_edge(node1.element, node2.element, edge[2], edge[3])
            g_mst.nb_edges+=1
            g_mst.powers.add(edge[2])
            g_mst.edges.append(edge)
            Union_Find.union(node1, node2) # the union process prevents the formation of a circle in the structure of g_mst
    return g_mst







    