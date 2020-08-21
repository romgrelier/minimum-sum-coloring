import torch
import numpy as np
from tqdm import tqdm
from leria_internship.dimacs_loader import DIMACS
from leria_internship.chaco_loader import CHACO


def tenscol_mscp(device="cpu", verbose=False):
    instance = CHACO("chaco_instances/add20.graph")
    # instance = DIMACS("dimacs_instances/DSJC125.5.col")

    d = 4
    max_iter = 200000
    n = instance.node_count
    k = 2
    # 2  | 596
    # 4  | 1151
    # 8  | 1681
    # 16 | 2040
    # 32 | 2360
    # 64 | 2947

    c1 = int(n / k)
    c2 = int(n / k) + 1

    # adjacency matrix
    A = torch.from_numpy(instance.graph.astype(np.float32)).to(device)
    notA = 1.0 - A

    # weighted matrix
    a = torch.normal(mean=0.0, std=0.01, size=(d, n, k),
                     dtype=torch.float32, requires_grad=True, device=device)

    # optimizer
    optimizer = torch.optim.Adam([a], lr=0.001)

    # pbar
    pbar = tqdm(range(max_iter), disable=not verbose)

    # best solution
    best_gpp = 1000000
    best_equity = 100000
    best_k = k
    legal_solution_found = False

    for i in pbar:
        optimizer.zero_grad()

        # one hot solution
        a_softmax = torch.nn.functional.softmax(a, dim=2)
        _, a_max_indices = a_softmax.max(2)
        a_onehot = torch.zeros(
            a_softmax.shape, device=device).scatter(-1, a_max_indices.view((d, n, 1)), 1.0)
        a_onehot = a_onehot + a_softmax - a_softmax.data

        # vertices with same colors
        V = a_onehot @ a_onehot.transpose(1, 2)

        # vertices with different colors
        V = 1.0 - V

        # vertices with different colors and connected
        C = A * V

        # gpp objective
        gpp_objective = torch.sum(C, dim=(1, 2)) / 2.0

        # equity
        node_per_color = torch.sum(a_onehot, dim=1)
        diff_count = torch.min(torch.abs(node_per_color - c1), torch.abs(node_per_color - c2))
        e = torch.sum(diff_count, dim=1)

        # update best solution
        iter_best_value, iter_best_index = e.min(0)
        # if we found at least one proper solution
        if iter_best_value == 0:
            legal_mask = e == 0
            best_legal_value, best_legal_index = gpp_objective[legal_mask].min(0)
            legal_solution_found = True

            if best_legal_value < best_gpp:
                best_gpp = gpp_objective[legal_mask][best_legal_index].item()
                best_equity = e[legal_mask][best_legal_index].item()
                best_k = (torch.sum(a_onehot[legal_mask][best_legal_index, :, :], dim=(0)) > 0).sum().item()

        # if we didn't found a proper solution but a better one
        elif iter_best_value.item() < best_equity and not legal_solution_found:
            best_gpp = gpp_objective[iter_best_index].item()
            best_equity = e[iter_best_index].item()
            best_k = (torch.sum(a_onehot[iter_best_index, :, :], dim=(0)) > 0).sum().item()

        e = torch.sum(e)

        # concentration node/color for each solution
        concentration = torch.sum(V, dim=0)

        kappa = torch.sum(A * concentration**1.0) * 1.0
        # varpi = torch.sum(notA * concentration**1.2) * 1.0
        sum_l2_regul = torch.sum(a**2.0) * 1.0

        # loss function
        loss = torch.sum(gpp_objective) + e + sum_l2_regul + kappa

        # e * value : value higher with k value

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix(
                best_gpp=best_gpp,
                best_equity=best_equity,
                best_k=best_k,
                kappa=kappa.item(),
                l2=sum_l2_regul.item(),
                grad_sum=torch.sum(a.grad).item(),
            )


result = tenscol_mscp("cuda:0", verbose=True)
