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
from math import ceil


def tenscol_mscp(pathResults, idx, kwarg, device="cpu", verbose=False):
    torch.manual_seed(idx)
    torch.cuda.manual_seed(idx)
    # device = torch.device(device)

    data_type = torch.float32

    lr = 0.001
    d = 200

    # k, alpha, beta, lambda_, mu_, nu_, l2_regul = kwarg
    k, alpha, beta, gamma, mu_, nu_, pu_, l2_regul = kwarg


    # 28 | 3210
    # instance = DIMACS("dimacs_instances/DSJC250.5.col")
    17 | 1012
    instance = DIMACS("dimacs_instances/DSJC125.5.col")
    # instance = DIMACS("dimacs_instances/anna.col")

    d = int(d)
    max_iter = 100_000
    iter_without_improvement = 0
    iter_without_improvement_limit = 100000
    n = instance.node_count
    k = int(k)

    # adjacency matrix
    A = torch.from_numpy(instance.graph.astype(np.float32)).to(device)
    notA = 1 - A

    # color multiplier
    Vi = torch.arange(1, k+1, dtype=data_type, device=device)
    Vi_r = Vi.flip(0)

    eye = torch.ones(n, device=device) - torch.eye(n, device=device)

    # weighted matrix
    a = torch.normal(mean=0.0, std=0.001, size=(d, n, k),
                     dtype=data_type, requires_grad=True, device=device)
    # a = torch.load(f"checkpoint_{100000}")

    # weight mask
    mask_begin = 0
    mask_size = 200
    mask_duration = 1

    # optimizer
    # optimizer = torch.optim.SGD([a], lr=lr)
    optimizer = torch.optim.Adam([a], lr=lr)
    nb_reset = 0

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
        # a_softmax = torch.nn.functional.softmax(a[mask_begin:mask_begin+mask_size], dim=2)
        a_softmax = torch.nn.functional.softmax(a, dim=2)
        # creates the representation of the solution with only 0 and 1

        a_max_values, a_max_indices = a_softmax.max(2)

        # one hot encoding of the solution
        # a_onehot = torch.zeros((mask_size, a_softmax.shape[1], a_softmax.shape[2]), device=device).scatter(-1, a_max_indices.view((mask_size, n, 1)), 1.0)
        a_onehot = torch.zeros(a_softmax.shape, device=device).scatter(-1, a_max_indices.view((d, n, 1)), 1.0)
        a_onehot = a_onehot + a_softmax - a_softmax.data
        a_onehot_w = a_onehot * (k + 1)
        a_onehot_w_r_malus = a_onehot * Vi
        a_onehot_w_r_bonus = a_onehot * Vi_r

        # association matrix
        V = a_onehot @ a_onehot.transpose(1, 2)
        V_w = a_onehot @ a_onehot_w.transpose(1, 2)  # conflict cost
        V_w_r_malus = a_onehot @ a_onehot_w_r_malus.transpose(1, 2) # mscp malus
        V_w_r_bonus = a_onehot @ a_onehot_w_r_bonus.transpose(1, 2) # mscp bonus

        # conflict matrix
        C = A * V

        # weighted association matrix smoothing
        # V_w = torch.nn.functional.softmax(V_w, dim=1)

        # concentration node/color for each solution
        concentration_w = torch.sum(V_w, dim=0) + 1.0
        concentration_w_r_malus = torch.sum(V_w_r_malus, dim=0) + 1.0
        concentration_w_r_bonus = torch.sum(V_w_r_bonus, dim=0) + 1.0

        conflict = torch.sum(A * concentration_w**alpha) * mu_
        malus = torch.sum(notA * concentration_w_r_malus**beta) * nu_
        bonus = torch.sum(notA * concentration_w_r_bonus**gamma) * pu_
        sum_l2_regul = torch.sum(a**2.0) * l2_regul

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
        solution_fitness = min_col_criteria_solution + conflict_criteria_solution
        # solution_fitness_sum = torch.sum(solution_fitness)

        # global fitness function
        global_fitness_sum = sum_l2_regul + conflict + malus - bonus #+ solution_fitness_sum

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
                best_constraint = int(ceil(conflict_criteria_solution[legal_mask][best_solution_index_iter]))
                best_fitness = solution_fitness[legal_mask][best_solution_index_iter]
                k_solution = (torch.sum(a_onehot[legal_mask][best_solution_index_iter, :, :], dim=(0)) > 0).sum()

                best_solution_time = time() - time_start
                iter_without_improvement = 0
                # if nu_ > 1.0:
                #     nu_ = 1.0

        # keep the best constraint cost solution as best solution
        elif iter_best_value < best_constraint and not legal_solution_found:
            best_objective = min_col_criteria_solution[iter_best_index]
            best_constraint = int(ceil(conflict_criteria_solution[iter_best_index]))
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
                b_c=best_constraint,                # best constraint
                b_o=best_objective.item(),          # best mscp objective
                b_f=best_fitness.item(),            # best fitness/global cost function
                k=k_solution.item(),                # color count used

                # t_c=conflict_criteria.item(),       # actual constraint
                # t_o=min_col_criteria.item(),        # actual mscp objective
                # t_f=solution_fitness_sum.item(),    # actual fitness/global cost function

                l2=sum_l2_regul.item(),
                grad_sum=torch.sum(a.grad).item(),
            )

        # iter_without_improvement += 1

        # if iter_without_improvement >= iter_without_improvement_limit:
        #     iter_without_improvement = 0

    return best_objective.item(), best_constraint, best_solution_time


# result, constraint, duration = tenscol_mscp("", 0, [
#     15.0,      # k
#     # 0.001,    # learning rate
#     # 200,      # d
#     2.0,     # alpha
#     1.0,     # beta
#     1.0,     # gamma
#     1.0,     # mu_
#     1.0,    # nu_
#     1.0,     # pu_
#     1.0,      # l2_regul
# ], "cuda", verbose=True)


# print(f"{result} in {duration} seconds")

bounds = [
    [18, 24],  # k
    [0.1, 5.0],  # alpha
    [0.1, 5.0],  # beta
    [0.1, 2.0],  # gamma
    [0.1, 50.0],  # mu_
    [0.1, 50.0],  # nu_
    [0.1, 50.0],  # pu_
    [50.0, 170.0],  # l2_regul
]

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


optim = cma.CMAEvolutionStrategy(
    encode([22, 1.38, 1.93, 0.25, 2.0, 1.5, 1.1, 72.77], bounds),
    0.2,
    {
        "popsize": 12
    }
)

best_result = np.inf
best_hyperparameters = []
best_time = 0

with open("results_v2", "w") as csvfile:
    csvfilewriter = csv.writer(csvfile)
    csvfilewriter.writerow([
        "best_time",
        "best_result",
        "best_constraint",
        "seed",
        "k",
        "alpha",
        "beta",
        "gamma",
        "mu",
        "nu",
        "pu",
        "l2_regul",
    ])

while not optim.stop():
    solutions = optim.ask()

    solutions_decoded = decode(solutions, bounds)

    seeds, fitness, constraints, durations = parallel_run_generator(
        tenscol_mscp, solutions_decoded, njobs=12, gpus=4)

    fitness_np = np.array(fitness)
    constraints_np = np.array(constraints)
    durations_np = np.array(durations)
    solutions_decoded_np = np.array(solutions_decoded)

    best_constraint_id = np.argmin(constraints_np)
    if constraints_np[best_constraint_id] == 0:
        legal_solution = constraints_np == 0

        best_solution_id = np.argmin(fitness_np[legal_solution])

        if fitness_np[legal_solution][best_solution_id] < best_result:
            best_result = fitness_np[legal_solution][best_solution_id]
            best_hyperparameters = solutions_decoded_np[legal_solution][best_solution_id]
            best_time = durations_np[legal_solution][best_solution_id]

            print(f"new best : {best_result} in {best_time} sec")

    with open("results_v2", "a") as csvfile:
        csvfilewriter = csv.writer(csvfile)

        for i in range(len(fitness)):
            csvfilewriter.writerow([
                durations[i], fitness[i], constraints[i], seeds[i], *solutions_decoded[i]
            ])

    optim.tell(
        solutions,
        fitness_np + constraints_np * 1000
    )

    optim.logger.add()



