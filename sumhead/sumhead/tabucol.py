import numpy as np
from dimacs_loader import DIMACS
from time import time
from numba import njit, prange


def conflicts(solution, graph):
    node_conflicts = np.zeros_like(solution)

    for v_a in range(solution.shape[0]):
        for v_b in range(v_a):
            if graph[v_a, v_b] and solution[v_a] == solution[v_b]:
                node_conflicts[v_a] += 1

    return node_conflicts


def create_gamma(graph, solution, color_count):
    """
    Keeps the color count in the neighborhood for each vertex
    """
    gamma = np.zeros((graph.shape[0], color_count), dtype=np.int32)
    conflict_count = 0

    for x in range(graph.shape[0]):
        for y in range(x):
            if graph[x, y]:
                gamma[x, solution[y]] += 1
                gamma[y, solution[x]] += 1

                if solution[x] == solution[y]:
                    conflict_count += 1

    return gamma, conflict_count


def set_color(graph, solution, gamma, vertex, new_color):
    """
    Sets the enw color for a vertex and update the gamma matrix
    """
    former_color = solution[vertex]
    solution[vertex] = new_color

    for n in range(solution.shape[0]):
        if graph[vertex, n]:
            gamma[n, former_color] -= 1
            gamma[n, new_color] += 1


def delta_improvement(gamma, solution, vertex, color):
    """
    Computes the improvement (delta) for a new color
    """
    return gamma[vertex, color] - gamma[vertex, solution[vertex]]


def tabucol(instance, solution, color_count, max_iter=1):
    best_solution = np.copy(solution)

    gamma, conflict_count = create_gamma(instance.graph, solution, color_count)
    solution_objective = conflict_count
    best_objective = solution_objective
    tabu = np.zeros((instance.vertex_count, color_count), dtype=np.int32)
    graph = instance.graph

    i = 0
    while i < max_iter and best_objective != 0:
        best_update = (0, 0)  # 0 : vertex, 1 : color
        best_improvement = delta_improvement(gamma, solution, best_update[0], best_update[1])

        # search for the best improvement

        # each vertex of the solution
        for v in range(solution.shape[0]):

            # if at least on conflict exists with this vertex
            if gamma[v, solution[v]] > 0:

                # each color available
                for c in range(color_count):
                    # we are looking for a different color and a move which is not tabu
                    if c != solution[v] and not tabu[v, c] > i:
                        improvement = delta_improvement(gamma, solution, v, c)
                        if improvement < best_improvement:
                            best_update = (v, c)
                            best_improvement = improvement
                        elif improvement == best_improvement and np.random.rand() > 0.5:
                            best_update = (v, c)
                            best_improvement = improvement

        # update the solution with the best move found
        solution_objective += best_improvement
        # print(f"objective : {solution_objective}")
        set_color(graph, solution, gamma, best_update[0], best_update[1])

        # update tabu list for the move found
        tabu[best_update[0], best_update[1]] = i + int(0.6 * solution_objective) + np.random.randint(1, 10)

        if solution_objective < best_objective:
            best_objective = solution_objective
            best_solution[:] = solution[:]
            # print(best_objective)

        i += 1

    return best_solution, best_objective


if __name__ == "__main__":
    instance = DIMACS("dimacs_instances/DSJC125.5.col")

    runs = 1
    durations = np.zeros(runs)

    for r in range(runs):
        start = time()
        solution = np.random.randint(0, 1, instance.vertex_count, dtype=np.int32)
        solution, conflict = tabucol(instance, solution, 20, 1_000)
        durations[r] = time() - start
        _, conflict_count = create_gamma(instance.graph, solution, 20)
        print(f"{conflict} | {conflict_count} | {conflicts(solution, instance.graph).sum()}")
        assert(conflict == conflict_count == conflicts(solution, instance.graph).sum())

    print(f"duration : {durations}")
