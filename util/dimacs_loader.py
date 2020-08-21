import numpy as np

class DIMACS:
    def __init__(self, filename):
        self.graph, self.edge_count, self.vertex_count = loader(filename)


def loader(filename):
    content = ""

    with open(filename, "r") as file:
        content = file.readlines()
    
    edge_count = 0
    vertex_count = 0
    graph = np.empty(0)

    for line in content:
        if line[0] == "c":
            pass
        elif line[0] == "p":
            splitted = line.split()
            vertex_count = int(splitted[2])
            edge_count = int(splitted[3])
            graph = np.zeros((vertex_count, vertex_count), dtype=np.bool)
        elif line[0] == "e":
            splitted = line.split()
            graph[int(splitted[1])-1, int(splitted[2])-1] = True
            graph[int(splitted[2])-1, int(splitted[1])-1] = True
    
    return graph, edge_count, vertex_count


if __name__ == "__main__":
    instance = DIMACS("dimacs_instances/own_10.col")

    print(instance.graph)
