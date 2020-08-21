import numpy as np


class CHACO:
    def __init__(self, filename):
        self.graph, self.node_count, self.vertex_count = loader(filename)


def loader(filename):
    content = ""

    with open(filename, "r") as file:
        content = file.readlines()

    iterator = iter(content)

    node_count, vertex_count = map(int, next(iterator).split())
    graph = np.zeros((node_count, node_count), dtype=np.bool)

    for n, line in enumerate(iterator):
        connections = np.array(list(map(int, line.split()))) - 1
        graph[n, connections] = True
        # print(f"{n} : {connections}")
    
    return graph, node_count, vertex_count
