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

    data_type = torch.float32

    k, lr, d, alpha, beta, lambda_, mu_, nu_, l2_regul = kwarg
    # k, lr, d, alpha, lambda_, nu_, l2_regul = kwarg

    # 28 | 3210
    # instance = DIMACS("dimacs_instances/DSJC250.5.col")
    # 17 | 1012
    instance = DIMACS("dimacs_instances/DSJC125.5.col")

    d = int(d)
    max_iter = 1000000
    iter_without_improvement = 0
    iter_without_improvement_limit = 1000
    n = instance.node_count
    k = int(k)

    # adjacency matrix
    A = torch.from_numpy(instance.graph.astype(np.float32)).to(device)
    notA = 1 - A

    # color multiplier
    Vi = torch.arange(1, k+1, dtype=data_type, device=device)

    # weighted matrix
    a = torch.normal(mean=0.0, std=0.01, size=(d, n, k),
                     dtype=data_type, requires_grad=True, device=device)
    # a = torch.load(f"checkpoint_{100000}")

    # weight mask
    available_mask = [100]
    actual_mask = 0
    mask_begin = 0
    mask_size = available_mask[actual_mask]
    mask_duration = 1

    # optimizer
    # optimizer = torch.optim.SGD([a], lr=lr)
    optimizer = torch.optim.Adam([a], lr=lr, betas=(0.8, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

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
    # stats = torch.empty((max_iter, mask_size+4), dtype=data_type)
    # update_stat_rate = 10000

    time_start = time()
    for i in pbar:
        # nu_ = torch.cos(torch.tensor(i, dtype=data_type) / 100.0) * 100.0

        # mask = torch.zeros(size=(1, 1, d), dtype=data_type, device=device).transpose_(0, 2)
        # mask[torch.randint(0, mask.shape[0], size=(1, int(mask.shape[0] * 0.5))).squeeze_()] = 0.0

        optimizer.zero_grad()

        # apply softmax
        a_softmax = torch.nn.functional.softmax(a[mask_begin:mask_begin+mask_size], dim=2) # * mask
        # creates the representation of the solution with only 0 and 1

        a_max_values, a_max_indices = a_softmax.max(2)

        # one hot encoding of the solution
        a_onehot = torch.zeros((mask_size, a_softmax.shape[1], a_softmax.shape[2]), device=device).scatter(-1, a_max_indices.view((mask_size, n, 1)), 1.0)
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
        # concentration = torch.softmax(V, dim=0)
        # concentration_w = torch.sum(V_w, dim=0)

        kappa = torch.sum(A * concentration**alpha) * lambda_
        varpi = torch.sum(notA * concentration**beta) * mu_
        sum_l2_regul = torch.sum(a**2.0) * l2_regul

        # kappa = torch.sum(A * concentration_w**alpha) * lambda_
        # varpi = torch.sum(notA * (1 - concentration_w)**beta) * mu_

        # objective : minimize sum(Vi * si)
        min_col_criteria_solution = torch.sum(a_onehot * Vi, dim=(1, 2))
        min_col_criteria = torch.sum(min_col_criteria_solution)

        # color counter per solution
        # min_col_criteria_k_solution = torch.sum(torch.sum(a_onehot, dim=(1)) > 0, dim=(1))
        # min_col_criteria_k = torch.sum(min_col_criteria_k_solution)

        # constraint : conflicts
        conflict_criteria_solution = torch.sum(C, dim=(1, 2)) / 2.0
        # conflict_criteria = torch.sum(conflict_criteria_solution)

        # fitness function for each solution
        solution_fitness = conflict_criteria_solution * nu_ + min_col_criteria_solution
        solution_fitness_sum = torch.sum(solution_fitness)

        # global fitness function
        global_fitness_sum = solution_fitness_sum + sum_l2_regul + kappa - varpi

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
        # stats[i, :mask_size] = min_col_criteria_solution[:]
        # stats[i, mask_size] = torch.sum(a.grad).item()
        # stats[i, mask_size + 1] = kappa.item()
        # stats[i, mask_size + 2] = sum_l2_regul.item()
        # stats[i, mask_size + 3] = best_objective.item()

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

                # lr=scheduler.get_lr()[0],
                mask=mask_size,

                kappa=kappa.item(),
                varpi=varpi.item(),
                l2=sum_l2_regul.item(),
                grad_sum=torch.sum(a.grad).item()
            )

        # next batch
        mask_begin += mask_size

        # if no more improvements are made, the batch size increases
        if iter_without_improvement > iter_without_improvement_limit:
            actual_mask += 1
            if actual_mask >= len(available_mask):
                actual_mask = 0
            mask_size = available_mask[actual_mask]
            mask_begin = 0
            iter_without_improvement = 0

        iter_without_improvement += 1

        # if i % mask_duration == 0:
        #     mask_begin += mask_size

        # restart at the first batch
        if mask_begin >= a.shape[0]:
            mask_begin = 0
            # scheduler.step()

    return best_objective.item() + best_constraint.item() * n, best_solution_time


result, duration = tenscol_mscp("", 0, [
    19,     # k
    0.001,   # learning rate
    1000,    # d
    2.36,   # alpha
    1.2,    # beta
    1.0,   # lambda_
    10.0,    # mu_
    1.2,    # nu_
    100.0,    # l2_regul
], "cuda", verbose=True)
#
# print(f"{result} in {duration} seconds")
#
# bounds = [
#     [97, 110],  # k
#     [0.001, 0.1],  # learning rate
#     [100, 500],  # d
#     [1.0, 10.0],  # alpha
#     [1.0, 10.0],  # beta
#     [0.1, 100.0],  # lambda_
#     [0.001, 10.0],  # mu_
#     [0.01, 50.0],  # nu_
#     [1.0, 100.0],  # l2_regul
# ]

def decode(solutions, bounds):
    """
        y = a + (b - a) * (1 - cos(pi * x)) / 2
    """
    return [
        [
            bounds[i][0] + (bounds[i][1] - bounds[i][0]) * (1 - cos(pi * s[i])) / 2 for i in range(len(s))
        ] for s in solutions
    ]


def encode(solution, bounds):
    """
    1 / pi * arcos(1 - 2 * (lb - hb) / (hb - lb))
    """
    return [
        1.0 / pi * acos(1.0 - 2.0 * (solution[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])) for i in range(len(solution))
    ]
#
#
# optim = cma.CMAEvolutionStrategy(
#     encode([100, 0.005, 250, 2.36, 1.2, 1.0, 1.0, 1.0, 20.0], bounds),
#     0.3,
# )
# best_result = np.inf
# best_hyperparameters = []
# best_time = 0
#
# with open("results", "w") as csvfile:
#     csvfilewriter = csv.writer(csvfile)
#     csvfilewriter.writerow([
#         "best_time", "best_result", "k", "lr", "alpha", "lambda", "nu", "l2_regul",
#         {
#             "pop_size": 8
#         }
#     ])
#
#     while not optim.stop():
#         solutions = optim.ask()
#
#         solutions_decoded = decode(solutions, bounds)
#
#         results, durations = parallel_run_generator(
#             tenscol_mscp, solutions_decoded, njobs=1, gpus=1)
#
#         iter_best = np.argmin(results)
#         if results[iter_best] < best_result:
#             best_result = results[iter_best]
#             best_hyperparameters = solutions[iter_best]
#             best_time = durations[iter_best]
#
#             print(f"new best : {best_result} in {best_time} sec")
#             csvfilewriter.writerow([
#                 best_time, best_result, *best_hyperparameters
#             ])
#
#         optim.tell(
#             solutions,
#             results
#         )
#
#         optim.logger.add()

# bounds = [
#     [97, 110],  # k
#     [0.001, 0.1],  # learning rate
#     [100, 250],  # d
#     [1.0, 10.0],  # alpha
#     [1.0, 10.0],  # beta
#     [0.1, 100.0],  # lambda_
#     [0.001, 10.0],  # mu_
#     [0.01, 50.0],  # nu_
#     [1.0, 100.0],  # l2_regul
# ]
#
# print(decode([[
# 0.16702906033325052, -0.12310687039926882, 0.1623500607750789, 1.0807290165537888, -0.12202714447165344, -0.017414443681963584, -0.1072573530389534, 0.11721577441384737, 0.6373113996634463
# ]], bounds))
