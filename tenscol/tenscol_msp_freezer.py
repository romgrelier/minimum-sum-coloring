from torch import normal
from torch import float32
from torch import zeros
from torch import arange
from torch import from_numpy
from torch import sum
from torch.optim import Adam
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np
from dimacs_loader import DIMACS


k, lr, d, alpha, beta, lambda_, mu_, nu_, l2_regul = [
    18,     # k
    0.008,   # learning rate
    1000,    # d
    2.18,   # alpha
    1.15,    # beta
    62.0,   # lambda_
    50.0,    # mu_
    3.0,    # nu_
    1.0    # l2_regul
]

device = "cuda:0"

def weight_to_onehot(solution):
    solution_softmax = softmax(solution, dim=2)
    solution_max_values, solution_max_indices = solution_softmax.max(2)
    solution_onehot = zeros(solution.shape, device=device).scatter(-1, solution_max_indices.view((solution.shape[0], n, 1)), 1.0)
    solution_onehot = solution_onehot + solution_softmax - solution_softmax.data

    return solution_onehot

instance = DIMACS("dimacs_instances/DSJC125.5.col")
A = from_numpy(instance.graph.astype(np.float32)).to(device)
notA = 1 - A

d = int(d)
n = instance.node_count
k = int(k)

Vi = arange(1, k+1, dtype=float32, device=device)

begin = 0
mask_size = 200

w = normal(mean=0.0, std=0.01, size=(d, n, k), dtype=float32, device=device)

# pbar
pbar = tqdm(range(100000), disable=False)

# best solution
legal_solution_found = False
best_solution_time = 0
best_fitness = 1000000  # cost function
best_objective = 1000000  # mscp objective
best_constraint = 1000000  # constraint cost
k_solution = k

for i in pbar:
    # frozen population compute
    frozen_kappa = 0
    frozen_varpi = 0
    frozen_sum_l2_regul = 0
    frozen_min_col_criteria_solution = 0
    frozen_conflict_criteria_solution = 0

    # Left
    if begin != 0:
        left = w[:begin]
        # print(f"left : {left.shape}")
        left_onehot = weight_to_onehot(left)
        left_V = left_onehot @ left_onehot.transpose(1, 2)
        left_C = A * left_V
        left_concentration = sum(left_V, dim=0)

        frozen_kappa += sum(A * left_concentration ** alpha) * lambda_
        frozen_varpi += sum(notA * left_concentration ** beta) * mu_
        frozen_sum_l2_regul += sum(left_onehot ** 2.0) * l2_regul
        frozen_min_col_criteria_solution += sum(left_onehot * Vi)
        frozen_conflict_criteria_solution += sum(left_C) / 2.0

    # Right
    if begin + mask_size != d:
        right = w[begin+mask_size:]
        # print(f"right : {right.shape}")
        right_onehot = weight_to_onehot(right)
        right_V = right_onehot @ right_onehot.transpose(1, 2)
        right_C = A * right_V
        right_concentration = sum(right_V, dim=0)

        frozen_kappa += sum(A * right_concentration ** alpha) * lambda_
        frozen_varpi += sum(notA * right_concentration ** beta) * mu_
        frozen_sum_l2_regul += sum(right_onehot ** 2.0) * l2_regul
        frozen_min_col_criteria_solution += sum(right_onehot * Vi)
        frozen_conflict_criteria_solution += sum(right_C) / 2.0

    # update population compute
    selection = w[begin:begin+mask_size]
    selection.requires_grad = True
    optimizer = Adam([selection], lr=0.001)

    for j in range(100):
        optimizer.zero_grad()

        selection_onehot = weight_to_onehot(selection)
        selection_V = selection_onehot @ selection_onehot.transpose(1, 2)
        selection_C = A * selection_V
        selection_concentration = sum(selection_V, dim=0)

        kappa = sum(A * selection_concentration ** alpha) * lambda_ + frozen_kappa
        varpi = sum(notA * selection_concentration ** beta) * mu_ + frozen_varpi
        sum_l2_regul = sum(selection ** 2.0) * l2_regul + frozen_sum_l2_regul
        selection_min_col_criteria_solution = sum(selection_onehot * Vi, dim=(1, 2))
        selection_conflict_criteria_solution = sum(selection_C, dim=(1, 2)) / 2.0

        loss = selection_min_col_criteria_solution \
               + frozen_min_col_criteria_solution \
               + selection_conflict_criteria_solution \
               + frozen_conflict_criteria_solution \
               + sum_l2_regul + kappa - varpi

        sum(loss).backward()
        optimizer.step()

        # best solution update (global but need to be local)
        iter_best_value, iter_best_index = selection_conflict_criteria_solution.min(0)
        if iter_best_value == 0:  # if at least a legal solution exists
            # search for the best solution defined by the mscp
            legal_mask = selection_conflict_criteria_solution == 0
            best_solution_value_iter, best_solution_index_iter = selection_min_col_criteria_solution[legal_mask].min(0)
            legal_solution_found = True

            # if a better solution exists keep it
            if best_solution_value_iter < best_objective:
                best_objective = best_solution_value_iter
                best_constraint = selection_conflict_criteria_solution[legal_mask][best_solution_index_iter]
                best_fitness = loss[legal_mask][best_solution_index_iter]
                k_solution = (sum(selection_onehot[legal_mask][best_solution_index_iter, :, :], dim=(0)) > 0).sum()

        # keep the best constraint cost solution as best solution
        elif iter_best_value < best_constraint and not legal_solution_found:
            best_objective = selection_min_col_criteria_solution[iter_best_index]
            best_constraint = selection_conflict_criteria_solution[iter_best_index]
            best_fitness = loss[iter_best_index]
            k_solution = (sum(selection_onehot[iter_best_index, :, :], dim=(0)) > 0).sum()

        if i % 1 == 0:
            pbar.set_postfix(
                b_c=best_constraint.item(),         # best constraint
                b_o=best_objective.item(),          # best mscp objective
                b_f=best_fitness.item(),            # best fitness/global cost function
                k=k_solution.item(),                # color count used

                # begin=begin,
                # end=begin+mask_size,

                kappa=kappa.item(),
                varpi=varpi.item(),
                l2=sum_l2_regul.item(),
                grad_sum=sum(selection.grad).item()
            )

    # next mask
    begin += mask_size
    if begin > d - mask_size:
        begin = 0