import torch
from torch.nn.functional import normalize
import numpy as np
from tqdm import tqdm
from dimacs_loader import DIMACS
from math import cos, pi, acos
from time import time
from parallel import parallel_run_generator
import cma
import csv


def tenscol_mscp(pathResults, idx, kwarg, device="cpu", verbose=False):
    torch.manual_seed(idx)
    torch.cuda.manual_seed(idx)
    # device = torch.device(device)

    filename, k, lr, d, alpha, beta, lambda_, mu_, nu_, l2_regul = kwarg
    # k, lr, d, alpha, beta, lambda_, mu_, nu_, l2_regul = kwarg

    # 28 | 3210
    instance = DIMACS(filename)
    # 17 | 1012
    # instance = DIMACS("dimacs_instances/DSJC125.5.col")
    # instance = DIMACS("graph_coloring/instances/latin_square_10.col")
    # instance = DIMACS("graph_coloring/instances/DSJR500.5.col")

    d = int(d)
    max_iter = 10000000
    iter_without_improvement = 0
    iter_without_improvement_limit = 1000000
    n = instance.node_count
    k = int(k)

    # adjacency matrix
    A = torch.from_numpy(instance.graph.astype(np.float32)).to(device)
    notA = 1 - A

    # color multiplier
    Vi = torch.arange(1, k+1, dtype=torch.float32, device=device)

    # weighted matrix
    a = torch.normal(mean=0.0, std=0.01, size=(d, n, k),
                     dtype=torch.float32, requires_grad=True, device=device)
    # a = torch.load(f"checkpoint_{100000}")

    # weight mask
    # mask = None

    # optimizer
    # optimizer = torch.optim.SGD([a], lr=lr)
    optimizer = torch.optim.Adam([a], lr=lr)

    # pbar
    pbar = tqdm(range(max_iter), disable=not verbose)

    # best solution
    legal_solution_found = False
    best_solution_time = 0
    best_fitness = 1000000      # cost function
    best_objective = 1000000    # mscp objective
    best_constraint = 1000000   # constraint cost
    k_solution = k

    # stats : d, gradient, kappa, sum_l2_regul, best_objective
    # stats = torch.empty((max_iter, d+4), dtype=torch.float32)

    # update_stat_rate = 10000

    time_start = time()
    for i in pbar:
        # mask = torch.ones(size=(1, 1, d), dtype=torch.float32, device=device).transpose_(0, 2)
        # mask[torch.randint(0, mask.shape[0], size=(1, int(mask.shape[0] * 0.05))).squeeze_()] = 0.0

        optimizer.zero_grad()

        # apply softmax
        a_softmax = torch.nn.functional.softmax(a, dim=2)
        # creates the representation of the solution with only 0 and 1
        a_max_values, a_max_indices = a_softmax.max(2)

        # one hot encoding of the solution
        a_onehot = torch.zeros(a_softmax.shape, device=device).scatter(-1, a_max_indices.view((d, n, 1)), 1.0)
        a_onehot = a_onehot + a_softmax - a_softmax.data
        # a_onehot_w = a_onehot * Vi #* 0.001

        # association matrix
        V = a_onehot @ a_onehot.transpose(1, 2)
        # V_w = a_onehot_w @ a_onehot_w.transpose(1, 2)
        # V_w = torch.nn.functional.softmax(V_w, dim=1)

        # conflict matrix
        C = A * V

        # concentration node/color for each solution
        concentration = torch.sum(V, dim=0)
        # concentration_w = torch.sum(V_w, dim=0)

        kappa = torch.sum(A * concentration**alpha) * lambda_
        varpi = torch.sum(notA * concentration**beta) * mu_
        sum_l2_regul = torch.sum(a**2.0) * l2_regul

        # kappa = torch.sum(A * concentration_w**alpha) * lambda_
        # varpi = torch.sum(notA * concentration_w**beta) * mu_

        # objective : minimize sum(Vi * si)
        min_col_criteria_solution = torch.sum(a_onehot * Vi, dim=(1, 2))
        min_col_criteria = torch.sum(min_col_criteria_solution)

        # constraint : conflicts
        conflict_criteria_solution = torch.sum(C, dim=(1, 2)) / 2.0
        # conflict_criteria = torch.sum(conflict_criteria_solution)

        # fitness function for each solution
        solution_fitness = conflict_criteria_solution + min_col_criteria_solution * nu_
        solution_fitness_sum = torch.sum(solution_fitness)

        # global fitness function
        global_fitness_sum = solution_fitness_sum + sum_l2_regul # + kappa - varpi

        # update best solution found
        iter_best_value, iter_best_index = conflict_criteria_solution.min(0)
        if iter_best_value == 0:  # if at least a legal solution exists
            # search for the best solution defined by the mscp
            legal_mask = conflict_criteria_solution == 0
            best_solution_value_iter, best_solution_index_iter = min_col_criteria_solution[legal_mask].min(0)
            legal_solution_found = True

            # if a better solution exists keep it
            if best_solution_value_iter < best_objective:
                best_objective = best_solution_value_iter
                best_constraint = conflict_criteria_solution[legal_mask][best_solution_index_iter]
                best_fitness = solution_fitness[legal_mask][best_solution_index_iter]
                k_solution = (torch.sum(a_onehot[legal_mask][best_solution_index_iter, :, :], dim=(0)) > 0).sum()

                best_solution_time = time() - time_start
                iter_without_improvement = 0
                # if nu_ > 1.0:
                #     nu_ = 1.0

        # keep the best constraint cost solution as best solution
        elif iter_best_value < best_constraint and not legal_solution_found:
            best_objective = min_col_criteria_solution[iter_best_index]
            best_constraint = conflict_criteria_solution[iter_best_index]
            best_fitness = solution_fitness[iter_best_index]
            k_solution = (torch.sum(a_onehot[iter_best_index, :, :], dim=(0)) > 0).sum()

            best_solution_time = time() - time_start
            iter_without_improvement = 0

        global_fitness_sum.backward()
        optimizer.step()

        # update stats
        # stats[i, :d] = min_col_criteria_solution[:]
        # stats[i, d] = torch.sum(a.grad)
        # stats[i, d + 1] = kappa
        # stats[i, d + 2] = sum_l2_regul
        # stats[i, d + 3] = best_objective
        #
        # if i % update_stat_rate == 0:
        #     with open("trace.csv", "ab") as csvfile:
        #         np.savetxt(csvfile, stats[i-update_stat_rate:i].detach().cpu().numpy(), delimiter=',')

        # record checkpoint
        # if i % 100000 == 0:
        #     torch.save(a, f"checkpoint_{i}")

        # pbar
        if i % 10 == 0:
            pbar.set_postfix(
                b_c=best_constraint.item(),         # best constraint
                b_o=best_objective.item(),          # best mscp objective
                b_f=best_fitness.item(),            # best fitness/global cost function
                k=k_solution.item(),                # color count used

                # t_c=conflict_criteria.item(),       # actual constraint
                # t_o=min_col_criteria.item(),        # actual mscp objective
                # t_f=solution_fitness_sum.item(),    # actual fitness/global cost function

                kappa=kappa.item(),
                varpi=varpi.item(),
                l2=sum_l2_regul.item(),
                grad_sum=torch.sum(a.grad).item()
            )

        if iter_without_improvement > iter_without_improvement_limit:
            # nu_ += 0.2
            # iter_without_improvement = 0
            return best_objective.item(), best_constraint.item(), best_solution_time

        iter_without_improvement += 1

    return best_objective.item(), best_constraint.item(), best_solution_time


result, constraints, duration = tenscol_mscp("", 0, [
    "graph_coloring/dimacs_instances/DSJC125.5.col",
    20,     # k
    0.001,   # learning rate
    100,    # d
    1.0,   # alpha
    1.0,    # beta
    1.0,   # lambda_
    0.1,    # mu_
    1.0,    # nu_
    1.0    # l2_regul
], "cuda:0", verbose=True)

# print(f"{result} with {constraints} conflicts in {duration} seconds")

# results, constraints, durations = parallel_run_generator(tenscol_mscp, [
#     ["graph_coloring/instances/DSJR500.1.col", 20, 0.001, 200, 1.5, 1.2, 0.1, 0.01, 0.01, 1.0],
#     ["graph_coloring/instances/DSJR500.1.col", 20, 0.001, 200, 1.5, 1.2, 0.1, 0.01, 0.01, 1.0],
#     ["graph_coloring/instances/DSJR500.1.col", 20, 0.001, 200, 1.5, 1.2, 0.1, 0.01, 0.01, 1.0],
#     ["graph_coloring/instances/DSJR500.1.col", 20, 0.001, 200, 1.5, 1.2, 0.1, 0.01, 0.01, 1.0],
#  ], njobs=4, gpus=4)

print(results)
print(constraints)
print(durations)

# bounds = [
#     [97, 110],  # k
#     [0.001, 0.1],  # learning rate
#     [50, 250],  # d
#     [1.0, 10.0],  # alpha
#     #[1.0, 10.0],  # beta
#     [0.1, 100.0],  # lambda_
#     #[0.001, 10.0],  # mu_
#     [0.01, 50.0],  # nu_
#     [1.0, 100.0],  # l2_regul
# ]

# def decode(solutions, bounds):
#     """
#         y = a + (b - a) * (1 - cos(pi * x)) / 2
#     """
#     return [
#         [
#             bounds[i][0] + (bounds[i][1] - bounds[i][0]) * (1 - cos(pi * s[i])) / 2 for i in range(len(s))
#         ] for s in solutions
#     ]


# def encode(solution, bounds):
#     """
#     1 / pi * arcos(1 - 2 * (lb - hb) / (hb - lb))
#     """
#     return [
#         1.0 / pi * acos(1.0 - 2.0 * (solution[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])) for i in range(len(solution))
#     ]


# optim = cma.CMAEvolutionStrategy(
#     # encode([100, 0.005, 100, 10.0, 1.2, 1.0, 1.0, 1.0, 60.0], bounds),
#     encode([97, 0.005, 100, 10.0, 1.0, 1.0, 60.0], bounds),
#     0.5,
# )
# best_result = np.inf
# best_hyperparameters = []
# best_time = 0

# with open("results", "w") as csvfile:
#     csvfilewriter = csv.writer(csvfile)
#     csvfilewriter.writerow([
#         "best_time", 
#         "best_result", 
#         "k", 
#         "lr", 
#         "alpha", 
#         "beta",
#         "lambda", 
#         "mu_"
#         "nu", 
#         "l2_regul",
#     ])

# while not optim.stop():
#     solutions = optim.ask()

#     solutions_decoded = decode(solutions, bounds)
#     # solutions_decoded = solutions

#     results, durations = parallel_run_generator(
#         tenscol_mscp, solutions_decoded, njobs=8, gpus=4)

#     iter_best = np.argmin(results)
#     if results[iter_best] < best_result:
#         best_result = results[iter_best]
#         best_hyperparameters = solutions[iter_best]
#         best_time = durations[iter_best]

#         with open("results", "w") as csvfile:
#             csvfilewriter = csv.writer(csvfile)
#             print(f"new best : {best_result} in {best_time} sec")
#             csvfilewriter.writerow([
#                 best_time, best_result, *best_hyperparameters
#             ])

#     optim.tell(
#         solutions,
#         results
#     )

#     optim.logger.add()


