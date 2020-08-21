import numpy as np
from dimacs_loader import DIMACS
from time import time
from numba import njit, prange
import tabucol


def conflicts(solution, graph):
    node_conflicts = np.zeros_like(solution)

    for v_a in range(solution.shape[0]):
        for v_b in range(v_a):
            if graph[v_a, v_b] and solution[v_a] == solution[v_b]:
                node_conflicts[v_a] += 1

    return node_conflicts


def create_gamma_sum(solution, color_count):
    """
    gives the actual solution cost for each color
    """
    color_counter = np.zeros(color_count, dtype=np.int32)

    for c in solution:
        color_counter[c] += (c + 1)

    return color_counter


def set_color(graph, solution, gamma_sum, gamma_conflict, v, c, w):
    """
    Sets the actual color of v for w and the color c for vertex v and update the gamma_sum array
    """

    # update conflicts
    former_color_w = solution[w]
    new_color_w = solution[v]

    former_color_v = solution[v]
    new_color_v = c

    for n in range(solution.shape[0]):
        if graph[w, n]:
            gamma_conflict[n, former_color_w] -= 1
            gamma_conflict[n, new_color_w] += 1
        if graph[v, n]:
            gamma_conflict[n, former_color_v] -= 1
            gamma_conflict[n, new_color_v] += 1

    # update sum
    # gamma_sum[solution[w]] -= color_costs[solution[w]]
    gamma_sum[solution[w]] -= (solution[w] + 1)
    solution[w] = solution[v]
    solution[v] = c
    # gamma_sum[c] += color_costs[c]
    gamma_sum[c] += (c + 1)


def delta_improvement(solution, gamma_sum, v, c, w):
    """
    Computes the improvement to change the color of w in the actual color of v and v for c
    """
    # return color_costs[solution[w]] - color_costs[c]
    return solution[w] + 1 - c + 1


def tabusum(instance, solution, color_count, max_iter=1, max_iter_without_improvement=10):
    """

    """
    # color_costs = np.arange(1, color_count + 1)

    gamma_sum = create_gamma_sum(solution, color_count)
    solution_cost = np.sum(gamma_sum)
    gamma_conflict, _ = tabucol.create_gamma(instance.graph, solution, color_count)

    best_solution = np.copy(solution)
    best_objective = solution_cost

    tabu = np.zeros((instance.vertex_count, color_count), dtype=np.int32)

    iter_without_improvement = 0

    i = 0
    while i < max_iter and iter_without_improvement < max_iter_without_improvement:
        best_move = (0, 0, 0)
        best_improvement = 0
        updated = False

        # search for vertex v
        for v in range(solution.shape[0]):

            # search for a new color c for v
            for c in range(color_count):

                # check conflict in the neighborhood
                if gamma_conflict[v, c] == 0:

                    # second-move : search for another vertex w which will use the former color of v : c'
                    for w in range(solution.shape[0]):
                        # check conflict with the actual color of v and tabu
                        if gamma_conflict[w, solution[v]] == 0 \
                                and i > tabu[best_move[2], solution[best_move[0]]] \
                                and i > tabu[best_move[0], best_move[1]]:
                            improvement = delta_improvement(solution, gamma_sum, v, c, w)

                            if improvement > best_improvement:
                                best_move = (v, c, w)
                                best_improvement = improvement
                                # print(f"new best improvement : {best_improvement}")
                                updated = True
                            # elif improvement == best_improvement and np.random.rand() > 0.5:
                            #     best_move = (v, c, w)
                            #     best_improvement = improvement
                            #     updated = True

        if updated:
            # update solution
            set_color(instance.graph, solution, gamma_sum, gamma_conflict, best_move[0], best_move[1], best_move[2])
            solution_cost = np.sum(gamma_sum)

            # print("===================")
            # print(f"conflict: {conflicts(solution, instance.graph).sum()}")
            # print("cost :")
            # print(np.sum(create_gamma_sum(solution, color_count, color_costs)))
            # print("===================")
            # print(f"\t\tactual cost : {solution_cost}")

            # update tabu
            tabu[best_move[2], solution[best_move[0]]] = i + np.random.randint(0, 10, 1)
            tabu[best_move[0], best_move[1]] = i + np.random.randint(0, 10, 1)

            # update the best solution found
            if solution_cost < best_objective:
                best_objective = solution_cost
                best_solution[:] = solution[:]
                print(f"\t\tnew best solution {best_objective}")
        else:
            iter_without_improvement += 1

        i += 1

    return best_solution, best_objective


def score(solution):
    total = 0

    for i in range(solution.shape[0]):
        total += solution[i] + 1

    return total

if __name__ == "__main__":
    instance = DIMACS("dimacs_instances/DSJC125.5.col")

    runs = 1
    durations = np.zeros(runs)

    for r in range(runs):
        start = time()
        solution = np.arange(0, instance.vertex_count)
        np.random.shuffle(solution)
        solution, f = tabusum(instance, solution, instance.vertex_count, 100, 2)
        print(f"conflicts : {conflicts(solution, instance.graph).sum()}")
        print(f"score : {score(solution)} / {f}")
        durations[r] = time() - start

    print(f"duration : {durations}")
