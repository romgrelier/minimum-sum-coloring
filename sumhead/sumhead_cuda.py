from dimacs_loader import DIMACS
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba
from numba import cuda
import logging
from math import ceil
from time import time

cuda.select_device(0)

vertex_count = 0

# numba.cuda.profile_stop()


def conflicts(solution, graph):
    node_conflicts = np.zeros_like(solution)

    for v_a in range(solution.shape[0]):
        for v_b in range(v_a):
            if graph[v_a, v_b] and solution[v_a] == solution[v_b]:
                node_conflicts[v_a] += 1

    return node_conflicts


def mscp_score(solution, color_count):
    """
    gives the actual solution cost for each color
    """
    color_counter = np.zeros(color_count, dtype=np.int32)

    for c in solution:
        color_counter[c] += (c + 1)

    return color_counter


def mscp_score(solution):
    return np.sum(solution + 1)


@cuda.jit
def tabucol(rng_states, solution_per_thread, graph, solution, max_iter, gamma_device, tabu_device, k):
    thread_id = cuda.grid(1)

    gamma = gamma_device[thread_id]
    tabu = tabu_device[thread_id]

    for p in range(int(thread_id * solution_per_thread), int(thread_id * solution_per_thread + solution_per_thread)):
        # Gamma creation
        for x in range(vertex_count):
            for y in range(k):
                gamma[x, y] = 0
                tabu[x, y] = 0

        # init gamma
        conflict_count = 0
        for x in range(vertex_count):
            for y in range(x):
                if graph[x, y]:
                    gamma[x, solution[p, y]] += 1
                    gamma[y, solution[p, x]] += 1

                    if solution[p, x] == solution[p, y]:
                        conflict_count += 1

        # actual solution
        solution_objective = conflict_count

        # best solution
        best_solution = cuda.local.array(vertex_count, dtype=numba.int32)
        for i in range(vertex_count):
            best_solution[i] = solution[p, i]
        best_objective = solution_objective

        i = 0
        while i < max_iter and best_objective != 0:
            best_update = (0, 0)  # 0 : vertex, 1 : color
            best_improvement = gamma[best_update[0], best_update[1]] - gamma[
                best_update[0], solution[p, best_update[0]]]
            # search for the best improvement

            # each vertex of the solution
            for v in range(vertex_count):

                # if at least on conflict exists with this vertex
                if gamma[v, solution[p, v]] > 0:

                    # each color available
                    for c in range(k):
                        # we are looking for a different color and a move which is not tabu
                        if c != solution[p, v] and not tabu[v, c] > i:
                            improvement = gamma[v, c] - gamma[v, solution[p, v]]
                            if improvement < best_improvement:
                                best_update = (v, c)
                                best_improvement = improvement
                            elif improvement == best_improvement and xoroshiro128p_uniform_float32(rng_states,
                                                                                                   thread_id) < 0.5:
                                best_update = (v, c)
                                best_improvement = improvement

            # update the solution with the best move found
            solution_objective += best_improvement

            former_color = solution[p, best_update[0]]
            solution[p, best_update[0]] = best_update[1]

            for n in range(vertex_count):
                if graph[best_update[0], n]:
                    gamma[n, former_color] -= 1
                    gamma[n, best_update[1]] += 1

            # update tabu list for the former color
            tabu[best_update[0], best_update[1]] = i + int(0.6 * solution_objective) + int(
                10 * xoroshiro128p_uniform_float32(rng_states, thread_id))

            # update the best solution found
            if solution_objective < best_objective:
                best_objective = solution_objective
                for j in range(vertex_count):
                    best_solution[j] = solution[p, j]

            i += 1

        # copy the best solution in the output
        for i in range(vertex_count):
            solution[p, i] = best_solution[i]


@cuda.jit
def remove_conflicts(solution_per_thread, graph, solution, color_count, k_final):
    """
    Search for a new solution without conflict:
        1 - for each vertex check if an already existing color can be used
        2 - if no color is available, a new color is added and used for this vertex
    """
    thread_id = cuda.grid(1)

    for p in range(int(thread_id * solution_per_thread), int(thread_id * solution_per_thread + solution_per_thread)):

        for v_a in range(vertex_count):
            for v_b in range(v_a):
                # conflict found
                if graph[v_a, v_b] and solution[p, v_a] == solution[p, v_b]:
                    valid_color_found = False
                    new_color = 0
                    # search for a new color for v_a
                    while not valid_color_found:
                        # we check all possible conflict for this new color on v_a
                        counter = 0
                        for v_c in range(vertex_count):
                            if graph[v_a, v_c] and new_color == solution[p, v_c]:
                                # a conflict is found, we try the next available color
                                new_color += 1
                                # if no more colors are available we add one
                                if new_color == color_count:
                                    color_count += 1
                                    valid_color_found = True
                            else:
                                counter += 1
                        if counter == vertex_count:
                            valid_color_found = True

                    solution[p, v_a] = new_color

        for _ in range(color_count):
            presence = 0
            c = color_count - 1
            for v_a in range(vertex_count):
                if solution[p, v_a] == c:
                    presence += 1
            if presence == 0:
                color_count -= 1
            else:
                break

        # returns the final k value found
        k_final[p] = color_count


@cuda.jit
def tabusum(rng_states, solution_per_thread, graph, solution, f_solution_output, max_iter_without_improvement,
            gamma_device, tabu_device, k):
    thread_id = cuda.grid(1)

    gamma_conflict = gamma_device[thread_id]
    tabu = tabu_device[thread_id]

    best_solution = cuda.local.array(vertex_count, dtype=numba.int64)

    for p in range(int(thread_id * solution_per_thread), int(thread_id * solution_per_thread + solution_per_thread)):
        solution_cost = 0
        for c in solution[p]:
            solution_cost += (c + 1)

        # fill 0
        for x in range(vertex_count):
            for y in range(k):
                gamma_conflict[x, y] = 0
                tabu[x, y] = 0

        # init gamma
        conflict_count = 0
        for x in range(vertex_count):
            for y in range(x):
                if graph[x, y]:
                    gamma_conflict[x, solution[p, y]] += 1
                    gamma_conflict[y, solution[p, x]] += 1

                    if solution[p, x] == solution[p, y]:
                        conflict_count += 1

        # best solution
        for i in range(vertex_count):
            best_solution[i] = solution[p, i]
        best_objective = solution_cost

        iter_without_improvement = 0
        i = 0
        while iter_without_improvement < max_iter_without_improvement:
            best_move = (0, 0, 0)
            best_improvement = 0
            updated = False
            improved = False

            # search for vertex v
            for v in range(vertex_count):

                # search for a new color c for v
                for c in range(k):

                    # check conflict in the neighborhood
                    if gamma_conflict[v, c] == 0:

                        # second-move : search for another vertex w which will use the former color of v : c'
                        for w in range(vertex_count):
                            # check conflict with the actual color of v and tabu
                            if gamma_conflict[w, solution[p, v]] == 0 \
                                    and i > tabu[best_move[2], solution[p, best_move[0]]] \
                                    and i > tabu[best_move[0], best_move[1]]:
                                improvement = (solution[p, w] + 1) - (c + 1)

                                if improvement > best_improvement:
                                    best_move = (v, c, w)
                                    best_improvement = improvement
                                    updated = True
                                    improved = True
                                elif improvement == best_improvement and xoroshiro128p_uniform_float32(rng_states,
                                                                                                       thread_id) < 0.5:
                                    best_move = (v, c, w)
                                    best_improvement = improvement
                                    updated = True

            if updated:
                # update solution
                former_color_w = solution[p, best_move[2]]
                new_color_w = solution[p, best_move[0]]

                former_color_v = solution[p, best_move[0]]
                new_color_v = best_move[1]

                for n in range(vertex_count):
                    if graph[best_move[2], n]:
                        gamma_conflict[n, former_color_w] -= 1
                        gamma_conflict[n, new_color_w] += 1
                    if graph[best_move[0], n]:
                        gamma_conflict[n, former_color_v] -= 1
                        gamma_conflict[n, new_color_v] += 1

                # update sum
                solution[p, best_move[2]] = solution[p, best_move[0]]
                solution[p, best_move[0]] = best_move[1]

                solution_cost = 0
                for v in range(vertex_count):
                    solution_cost += (solution[p, v] + 1)

                # update tabu
                tabu[best_move[2], new_color_w] = i + int(
                    10 * xoroshiro128p_uniform_float32(rng_states, thread_id))
                tabu[best_move[0], new_color_v] = i + int(
                    10 * xoroshiro128p_uniform_float32(rng_states, thread_id))

                # update the best solution found
                if solution_cost < best_objective:
                    best_objective = solution_cost
                    for j in range(vertex_count):
                        best_solution[j] = solution[p, j]
            if not improved:
                iter_without_improvement += 1
            else:
                iter_without_improvement = 0

            i += 1

        # copy the final solution in the output
        for l in range(vertex_count):
            solution[p, l] = best_solution[l]
        f_solution_output[p] = best_objective


@cuda.jit
def distance(individuals, individuals_size, vertex_count, distances, solution_per_thread):
    thread_id = cuda.grid(1)

    # for each solution
    for i in range(int(thread_id * solution_per_thread), int(thread_id * solution_per_thread + solution_per_thread)):
        # and its neighbors
        for j in range(individuals_size):
            # check differences
            differences = 0
            if i != j:
                for k in range(vertex_count):
                    if individuals[i, k] != individuals[j, k]:
                        differences += 1
            # if differences == 0:
            #     differences = vertex_count
            # update the distance matrix
            distances[i, j] = differences
            # distances[j, i] = differences


@cuda.jit
def remove_higher_color(rng, population, vertex_count, color_counter):
    thread_id = cuda.grid(1)

    for v in range(vertex_count):
        if population[thread_id, v] == color_counter - 1:
            population[thread_id, v] = int(color_counter * xoroshiro128p_uniform_float32(rng, thread_id))


@cuda.jit
def gpx(rng, population, offspring, graph, crossovered, color_count, color_counter, offspring_conflict, offspring_mscp, pop_size,
        knn, knn_size):
    # check for a parent a ...
    p_a = cuda.grid(1)

    # actual solution from the gpx crossover
    child = cuda.local.array(vertex_count, dtype=numba.int32)
    offspring_conflict[p_a] = 100_000
    offspring_mscp[p_a] = 100_000
    best_child_score = 100_000
    best_p_b = 0

    # ... with a parent b
    # for p_b in range(pop_size):
    i = 0
    while i < knn_size:
        p_b = int(knn[p_a, i])

        # only if this crossover didn't already happened
        if crossovered[p_a, p_b] == 0:

            # initialize color_counter
            for c in range(color_count):
                color_counter[p_a, c] = 0  # parent a
                color_counter[p_a + pop_size, c] = 0  # parent b

            # initialize child
            for v in range(vertex_count):
                child[v] = -1  # init the child to no color

            # how many each color are in the solution
            for v in range(vertex_count):
                color_counter[p_a, population[p_a, v]] += 1
                color_counter[p_a + pop_size, population[p_b, v]] += 1

            # applies color available according to the gpx crossover
            for c in range(color_count):
                # select a parent
                actual_parent = c % 2  # parent id in population
                actual_parent_cc = 0  # parent id in color_count
                if actual_parent == 0:
                    actual_parent = p_a
                    actual_parent_cc = p_a
                else:
                    actual_parent = p_b
                    actual_parent_cc = p_a + pop_size

                # starts with a random color
                start_color = xoroshiro128p_uniform_float32(rng, p_a)
                max_color = -1
                max_value = -1

                # search for the most used color
                for cs in range(color_count):
                    color = int((start_color + cs) % color_count)

                    # if the color is more used : update
                    if color_counter[actual_parent_cc, color] > max_value:
                        max_color = color
                        max_value = color_counter[actual_parent_cc, color]

                # update the child with the new color found
                for v in range(vertex_count):
                    if population[actual_parent, v] == max_color and child[v] == -1:
                        child[v] = c
                        # remove the color from the parents
                        color_counter[p_a, population[p_a, v]] -= 1
                        color_counter[p_a + pop_size, population[p_b, v]] -= 1

            # check for not assigned vertex and assign them with a random color
            for v in range(vertex_count):
                if child[v] == -1:
                    new_color = int(k * xoroshiro128p_uniform_float32(rng, p_a))
                    if new_color > color_count:
                        new_color = color_count - 1
                    child[v] = new_color

            # how many conflicts does this new child have
            child_conflicts = 0
            for x in range(vertex_count):
                for y in range(x):
                    if graph[x, y] and child[x] == child[y]:
                        child_conflicts += 1

            # compute the mscp score
            child_mscp = 0
            for v in range(vertex_count):
                child_mscp += (child[v] + 1)

            # check if the new child is better
            if child_conflicts + 2.0 * child_mscp < best_child_score:
                crossovered[p_a, best_p_b] = 0
                best_p_b = p_b
                crossovered[p_a, best_p_b] = 1
                offspring_conflict[p_a] = child_conflicts
                offspring_mscp[p_a] = child_mscp
                best_child_score = child_conflicts + 2.0 * child_mscp
                for v in range(vertex_count):
                    offspring[p_a, v] = child[v]

        i += 1


if __name__ == "__main__":
    instance = DIMACS("graph_coloring/dimacs_instances/DSJC125.5.col")
    
    # distance mean / min / mean / max

    stats = np.zeros((26, 4), dtype=np.float32)
    # HYPERPARAMETERS
    vertex_count = instance.vertex_count
    k = 20
    pop_size = 8192

    # CUDA
    threads_per_block = 256
    blocks = 32
    rng = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)

    # gamma matrix for tabucol and tabusum
    gamma_device = cuda.device_array((pop_size, vertex_count, k))
    # tabu matrix for tabucol and tabusum
    tabu_device = cuda.device_array((pop_size, vertex_count, k))
    # crossover already done
    crossovered_matrix = np.zeros((pop_size, pop_size), dtype=np.int32)

    # DATA
    # distances matrix from a solution to another
    distances = np.empty((pop_size, pop_size))
    distances_gpx = np.empty((pop_size, pop_size))
    # actual population of solution
    population = np.random.randint(0, k, (pop_size, vertex_count))
    # output of crossover, tabucol and tabusum to insert into the population
    offspring = np.empty_like(population)
    # color counter for the gpx crossover, 2 per kernel, 1 for each 2 parents
    color_counters = cuda.device_array((pop_size * 2, k))
    # conflict output for each solution after the crossover
    offspring_conflict = np.empty(pop_size)
    # mscp score for each solution after the crossover
    offspring_mscp = np.empty(pop_size)
    # k ouput of the conflict remover to choose a new k
    k_final = np.empty(pop_size)

    # Logging
    logging.basicConfig(level=logging.INFO, filename="sumhead_cuda_stats.log")

    # Solution
    best_score = np.inf
    best_solution = np.random.randint(0, k, vertex_count, dtype=np.int32)

    solution_per_thread = pop_size / (threads_per_block * blocks)

    population_mscp = np.array([
        np.inf for p in population
    ])

    for e in range(25):
        print(f"iteration {e}")

        start = time()
        # compute distances between solution for gpx
        print(f"DISTANCES")
        distance[blocks, threads_per_block](population, pop_size, vertex_count, distances_gpx, solution_per_thread)
        stats[e, 0] = distances_gpx.mean()
        print(f"\tmean distances : {distances_gpx.mean()}")
        print(f"\tdistance duration : {time() - start}")

        begin = 0
        end = 128

        # knn
        knn = np.argsort(distances_gpx)[:, begin:end]
        knn = np.array(knn)

        # CROSSOVER
        print(f"CROSSOVER")
        start = time()
        gpx[blocks, threads_per_block](rng, population, offspring, instance.graph, crossovered_matrix, k, color_counters,
                                       offspring_conflict, offspring_mscp, pop_size, knn, int(end - begin))
        print(f"\tcrossover duration : {time() - start}")

        # remove higher color
        # k -= 1
        # remove_higher_color[blocks, threads_per_block](rng, offspring, vertex_count, k)

        # TABUCOL
        print(f"TABUCOL")
        start = time()
        # print(np.array([conflicts(o, instance.graph).sum() for o in offspring]).mean())
        tabucol[blocks, threads_per_block](rng, solution_per_thread, instance.graph, offspring, 10_000, gamma_device,
                                           tabu_device, k)
        # print(np.array([conflicts(o, instance.graph).sum() for o in offspring]).mean())
        remaining_conliftcs = np.array([conflicts(o, instance.graph).sum() for o in offspring])
        print(f"\tmin : {offspring_conflict.min()} -> {remaining_conliftcs.min()}")
        print(f"\tmean : {offspring_conflict.mean()} -> {remaining_conliftcs.mean()}")
        print(f"\tmax : {offspring_conflict.max()} -> {remaining_conliftcs.max()}")
        print(f"\ttabucol duration : {time() - start}")
     

        # REMOVE REMAINING CONFLICTS
        print(f"CONFLICTS")
        start = time()
        remove_conflicts[blocks, threads_per_block](solution_per_thread, instance.graph, offspring, k, k_final)

        new_k = int(k_final.max())
        if new_k != k:
            k = new_k
            gamma_device = cuda.device_array((pop_size, vertex_count, k))
            tabu_device = cuda.device_array((pop_size, vertex_count, k))
            color_counters = cuda.device_array((pop_size * 2, k))
            print(f"\tk updated : {k}")
        print(f"\tremove_conflicts duration : {time() - start}")

        # TABUSUM
        print(f"TABUSUM")
        start = time()
        f_solution_output = np.empty(pop_size)
        tabusum[blocks, threads_per_block](rng, solution_per_thread, instance.graph, offspring, f_solution_output, 20,
                                           gamma_device, tabu_device, k)
        print(f"\ttabusum duration : {time() - start}")

        print(f"\tOffspring")
        print(f"\tmin : {f_solution_output.min()}")
        print(f"\tmean : {f_solution_output.mean()}")
        print(f"\tmax : {f_solution_output.max()}")

        stats[e, 1] = f_solution_output.min()
        stats[e, 2] = f_solution_output.mean()
        stats[e, 3] = f_solution_output.max()

        np.savetxt("stats_sumhead_8192_5.csv", stats, delimiter=",")   

        # update best solution
        best_id = f_solution_output.argmin()

        if f_solution_output[best_id] < best_score:
            best_score = f_solution_output[best_id]
            np.copyto(best_solution, offspring[best_id])
            logging.info(best_score)
            logging.info(best_solution)

        print(
            f"[BEST SCORE] {best_score}/{mscp_score(best_solution).sum()} | conflicts : {conflicts(best_solution, instance.graph).sum()}")

        # POPULATION UPDATE
        print("POPULATION UPDATE")
        start = time()
        not_inserted = 0
        inserted = 0
        inserted_random = 0
        for o in range(pop_size):

            # population distance
            distances = np.sum(population != offspring[o], axis=1)

            # goodness score
            distances_min = distances.min()
            beta = 0.08 * vertex_count
            goodness_score = np.exp(beta / (distances_min + 1.0)) + population_mscp

            sorted_score = np.argsort(goodness_score)
            chosen_score = -1
            while distances[-chosen_score] < 1.0 and -chosen_score < pop_size:
                chosen_score -= 1

            # use the worst goodness score to replace it with an element from the offspring
            #if not all(population[sorted_score[chosen_score]] == offspring[o]):
            np.copyto(population[sorted_score[chosen_score]], offspring[o])
            population_mscp[sorted_score[chosen_score]] = f_solution_output[o]
            inserted += 1
            crossovered_matrix[-chosen_score, :] = 0
            crossovered_matrix[:, -chosen_score] = 0
            #elif np.random.rand() < 0.5:
            #     population[sorted_score[-1]] = np.random.randint(0, k, instance.vertex_count)
            #     color = np.random.randint(0, k, 1)
            #     for v in range(instance.vertex_count):
            #        if population[sorted_score[-1], v] == color:
            #           population[sorted_score[-1], v] = np.random.randint(0, k, 1)
                #np.copyto(population[sorted_score[-2]], offspring[o])
                #population_mscp[sorted_score[-2]] = f_solution_output[o]
                #inserted_random += 1
            #else:
            #    not_inserted += 1
        #print(f"\t{not_inserted} not inserted")
        #print(f"\t{inserted} inserted")
        #print(f"\t{inserted_random} randomly inserted")
        #print(f"\tpopulation update duration : {time() - start}")

        print(f"=========================================================")
