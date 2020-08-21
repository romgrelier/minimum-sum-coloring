import numpy as np
from dimacs_loader import DIMACS
from time import time
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba
from tabucol import conflicts

vertex_count = 0
color_count = 0


@numba.cuda.jit(debug=False)
def tabucol(rng_states, graph, solution_host, solution_output, f_solution_output, max_iter, debug):
    thread_id = numba.cuda.grid(1)

    debug[thread_id] = vertex_count

    tabu = numba.cuda.local.array((vertex_count, color_count), dtype=numba.int32)

    # Gamma creation
    gamma = numba.cuda.local.array((vertex_count, color_count), dtype=numba.int32)
    for x in range(vertex_count):
        for y in range(color_count):
            gamma[x, y] = 0
            tabu[x, y] = 0

    # copy actual solution
    solution = numba.cuda.local.array(vertex_count, dtype=numba.int32)
    for i in range(vertex_count):
        solution[i] = solution_host[thread_id, i]

    # init gamma
    conflict_count = 0
    for x in range(vertex_count):
        for y in range(x):
            if graph[x, y]:
                gamma[x, solution[y]] += 1
                gamma[y, solution[x]] += 1

                if solution[x] == solution[y]:
                    conflict_count += 1

    # actual solution
    solution_objective = conflict_count

    # best solution
    best_solution = numba.cuda.local.array(vertex_count, dtype=numba.int32)
    for i in range(vertex_count):
        best_solution[i] = solution[i]
    best_objective = solution_objective

    i = 0
    while i < max_iter and best_objective != 0:
        best_update = (0, 0)  # 0 : vertex, 1 : color
        best_improvement = gamma[best_update[0], best_update[1]] - gamma[best_update[0], solution[best_update[0]]]
        # search for the best improvement

        # each vertex of the solution
        for v in range(vertex_count):

            # if at least on conflict exists with this vertex
            if gamma[v, solution[v]] > 0:

                # each color available
                for c in range(color_count):
                    # we are looking for a different color and a move which is not tabu
                    if c != solution[v] and not tabu[v, c] > i:
                        improvement = gamma[v, c] - gamma[v, solution[v]]
                        if improvement < best_improvement:
                            best_update = (v, c)
                            best_improvement = improvement
                        elif improvement == best_improvement and xoroshiro128p_uniform_float32(rng_states,
                                                                                               thread_id) > 0.5:
                            best_update = (v, c)
                            best_improvement = improvement

        # update the solution with the best move found
        solution_objective += best_improvement

        former_color = solution[best_update[0]]
        solution[best_update[0]] = best_update[1]

        for n in range(solution.shape[0]):
            if graph[best_update[0], n]:
                gamma[n, former_color] -= 1
                gamma[n, best_update[1]] += 1

        # update tabu list for the move found
        tabu[best_update[0], best_update[1]] = i + int(0.6 * solution_objective) + int(
            xoroshiro128p_uniform_float32(rng_states, thread_id))

        # update the best solution found
        if solution_objective < best_objective:
            best_objective = solution_objective
            for j in range(vertex_count):
                best_solution[j] = solution[j]

        i += 1

    # copy the best solution in the output
    for i in range(vertex_count):
        solution_output[thread_id, i] = best_solution[i]
    f_solution_output[thread_id] = best_objective


def cuda_wrapper(threads_per_block, blocks, instance, solution_from_host, color_count, max_iter):
    # vertex_count = instance.vertex_count
    # color_count = color_count

    # TODO : allocate solution output and objective output
    # solution_output = numba.cuda.device_array((threads_per_block * blocks, vertex_count), dtype=numba.int32)
    solution_output = np.empty_like(solution_from_host)
    # solution_output = numba.cuda.to_device(solution_output_local)
    # f_solution_output = numba.cuda.device_array((threads_per_block * blocks), dtype=numba.int32)
    f_solution_output = np.empty(threads_per_block * blocks)
    # f_solution_output = numba.cuda.to_device(f_solution_output_local)

    # DEBUG ARRAY
    debug = np.zeros(threads_per_block * blocks)

    # TODO : allocate in constant memory the graph
    # graph = numba.cuda.const.array_like(instance.graph)

    rng = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    print("KERNEL START")
    tabucol[blocks, threads_per_block](rng, instance.graph, solution_from_host, solution_output, f_solution_output, max_iter, debug)
    print("KERNEL STOP")

    # solution_output = solution_output.copy_to_host()
    # f_solution_output = f_solution_output.copy_to_host()

    # check for the best solution
    best_id = np.argmin(f_solution_output)
    best_solution = solution_output[best_id]

    for i in range(threads_per_block * blocks):
        print(f"{debug[i]} : {f_solution_output[i]} / {conflicts(solution_output[i], instance.graph).sum()}")

    return best_solution, f_solution_output[best_id]


if __name__ == "__main__":
    instance = DIMACS("dimacs_instances/DSJC125.5.col")

    runs = 1
    durations = np.zeros(runs)

    vertex_count = instance.vertex_count
    color_count = 25

    for r in range(runs):
        start = time()
        solution_init = np.random.randint(0, color_count, (32, instance.vertex_count))
        solution_end, conflict = cuda_wrapper(32, 1, instance, solution_init, color_count, 10_000)
        print(solution_end.shape)
        durations[r] = time() - start

    print(f"duration : {durations}")

