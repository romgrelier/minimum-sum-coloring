import numpy as np
from dimacs_loader import DIMACS
from time import time
import numba
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


def cuda_wrapper(threads_per_block, blocks, instance, solution_host, color_count, max_iter, max_iter_without_improvement):
    vertex_count = instance.vertex_count
    color_count = color_count

    # TODO : allocate solution output and objective output
    solution_output = np.empty((threads_per_block * blocks, vertex_count), dtype=np.int32)
    f_solution_output = np.empty(threads_per_block * blocks, dtype=np.int32)

    # DEBUG ARRAY
    debug = np.zeros(threads_per_block * blocks, dtype=np.int32)

    @numba.cuda.jit
    def tabusum(rng_states, graph, solution_host, solution_output, f_solution_output, max_iter=1, max_iter_without_improvement=10):
        thread_id = numba.cuda.grid(1)

        # actual solution
        solution = numba.cuda.local.array(vertex_count, dtype=numba.int32)
        for i in range(vertex_count):
            solution[i] = solution_host[thread_id, i]
        solution_cost = 0
        for c in solution:
            solution_cost += (c + 1)

        # Gamma sum init
        gamma_sum = numba.cuda.local.array(color_count, dtype=numba.int32)
        for i in range(color_count):
            gamma_sum[i] = 0
        for c in solution:
            gamma_sum[c] += (c + 1)

        # Gamma conflict and tabu init
        gamma_conflict = numba.cuda.local.array((vertex_count, color_count), dtype=numba.int32)
        tabu = numba.cuda.local.array((vertex_count, color_count), dtype=numba.int32)
        # fill 0
        for x in range(vertex_count):
            for y in range(color_count):
                gamma_conflict[x, y] = 0
                tabu[x, y] = 0
        # init gamma
        conflict_count = 0
        for x in range(vertex_count):
            for y in range(x):
                if graph[x, y]:
                    gamma_conflict[x, solution[y]] += 1
                    gamma_conflict[y, solution[x]] += 1

                    if solution[x] == solution[y]:
                        conflict_count += 1

        # best solution
        best_solution = numba.cuda.local.array(vertex_count, dtype=numba.int32)
        for i in range(vertex_count):
            best_solution[i] = solution[i]
        best_objective = solution_cost

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
                                improvement = (solution[w] + 1) - (c + 1)

                                if improvement > best_improvement:
                                    best_move = (v, c, w)
                                    best_improvement = improvement
                                    updated = True

            if updated:
                # update solution
                former_color_w = solution[best_move[2]]
                new_color_w = solution[best_move[0]]

                former_color_v = solution[best_move[0]]
                new_color_v = best_move[1]

                for n in range(solution.shape[0]):
                    if graph[best_move[2], n]:
                        gamma_conflict[n, former_color_w] -= 1
                        gamma_conflict[n, new_color_w] += 1
                    if graph[best_move[0], n]:
                        gamma_conflict[n, former_color_v] -= 1
                        gamma_conflict[n, new_color_v] += 1

                # update sum
                gamma_sum[solution[best_move[2]]] -= (solution[best_move[2]] + 1)
                solution[best_move[2]] = solution[best_move[0]]
                solution[best_move[0]] = best_move[1]
                gamma_sum[best_move[1]] += (best_move[1] + 1)

                solution_cost = 0
                for c in range(color_count):
                    solution_cost += gamma_sum[c]

                # update tabu
                tabu[best_move[2], solution[best_move[0]]] = i + int(xoroshiro128p_uniform_float32(rng_states,thread_id))
                tabu[best_move[0], best_move[1]] = i + int(xoroshiro128p_uniform_float32(rng_states, thread_id))

                # update the best solution found
                if solution_cost < best_objective:
                    best_objective = solution_cost
                    for j in range(vertex_count):
                        best_solution[j] = solution[j]
            else:
                iter_without_improvement += 1

            i += 1

        # copy the final solution in the output
        for i in range(vertex_count):
            solution_output[thread_id, i] = best_solution[i]
        f_solution_output[thread_id] = best_objective


    rng = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    tabusum[blocks, threads_per_block](
        rng, instance.graph, solution_host, solution_output, f_solution_output, max_iter, max_iter_without_improvement
    )

    # check for the best solution
    best_id = np.argmin(f_solution_output)
    best_solution = solution_output[best_id]

    for i in range(threads_per_block * blocks):
        check_f = np.sum([(c + 1) for c in solution_output[i]])
        print(f"f : {f_solution_output[i]} / {check_f}")
        print(debug[i])

    return best_solution, f_solution_output[best_id]


if __name__ == "__main__":
    instance = DIMACS("dimacs_instances/DSJC125.5.col")

    runs = 1
    durations = np.zeros(runs)

    for r in range(runs):
        start = time()
        solution = np.empty((1024, instance.vertex_count), dtype=np.int32)
        for i in range(solution.shape[0]):
            solution[i, :] = np.arange(0, instance.vertex_count)
            np.random.shuffle(solution[i])

        solution, conflict = cuda_wrapper(
            256, 4, instance, solution, instance.vertex_count, 100, 10
        )
        durations[r] = time() - start

    print(f"duration : {durations}")
