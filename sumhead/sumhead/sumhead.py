from dimacs_loader import DIMACS
from gpx import gpx
from tabucol import tabucol
from tabusum import tabusum, create_gamma_sum
import numpy as np


class sumHEAD:
    def __init__(self, instance, cs_operator, ls_operator):
        self.instance = instance

        self.cs_operator = cs_operator
        self.ls_operator = ls_operator

        # self.k = instance.vertex_count
        self.k = 11

    def conflicts(self, solution):
        node_conflicts = np.zeros_like(solution)

        for v_a in range(solution.shape[0]):
            for v_b in range(v_a):
                if self.instance.graph[v_a, v_b] and solution[v_a] == solution[v_b]:
                    node_conflicts[v_a] += 1

        return node_conflicts

    def remove_colors(self, solution):
        """
        Colors higher than the initial number are replaced with a random valid color
        """
        for i in range(solution.shape[0]):
            if solution[i] > self.k:
                solution[i] = np.random.randint(0, self.k)

        return solution

    def color_counter(self, solution):
        """
        Returns how many colors are used for this solution
        """
        color_counter = np.zeros(self.k, dtype=np.int32)

        for c in solution:
            color_counter[c] += 1

        return np.count_nonzero(color_counter)

    def remove_conflicts(self, solution):
        """
        Search for a new solution without conflict:
            1 - for each vertex check if an already existing color can be used
            2 - if no color is available, a new color is added and used for this vertex
        """
        for v_a in range(solution.shape[0]):
            for v_b in range(v_a):
                # if we found a conflict for a node
                if self.instance.graph[v_a, v_b] and solution[v_a] == solution[v_b]:
                    valid_color_found = False
                    new_color = 0
                    # search for a new color for v_a
                    while not valid_color_found:
                        # we check all possible conflict for this new color on v_a
                        counter = 0
                        for v_c in range(solution.shape[0]):
                            if self.instance.graph[v_a, v_c] and new_color == solution[v_c]:
                                # a conflict is found, we try the next available color
                                new_color += 1
                                # if no more colors are available we add one
                                if new_color == self.k:
                                    self.k += 1
                                    valid_color_found = True
                            else:
                                counter += 1
                        if counter == solution.shape[0]:
                            valid_color_found = True

                    solution[v_a] = new_color

    def run(self, max_iter=1):
        iter_cycle = 50

        colors_cost = np.arange(1, self.k + 1)

        p_a = np.random.randint(0, self.k, size=self.instance.vertex_count)
        # p_a = np.arange(0, self.k)
        np.random.shuffle(p_a)
        c_a_cost = create_gamma_sum(p_a, self.k, colors_cost)
        p_b = np.random.randint(0, self.k, size=self.instance.vertex_count)
        # p_b = np.arange(0, self.k)
        np.random.shuffle(p_b)
        c_b_cost = create_gamma_sum(p_b, self.k, colors_cost)

        # since the beginning
        best_solution = np.empty_like(p_a)
        best_cost = np.inf

        # elite
        elite_k_coloring_1 = np.empty_like(p_a)     # elite 1 - best k-coloring
        elite_k_coloring_1_k = np.inf
        elite_k_coloring_1_conflict = np.inf
        elite_k_coloring_1_cost = np.inf

        elite_mscp_1 = np.empty_like(p_a)           # elite theta 1 - best mscp
        elite_mscp_1_k = np.inf
        elite_mscp_1_conflict = np.inf
        elite_mscp_1_cost = np.inf

        elite_k_coloring_2 = np.empty_like(p_a)     # elite 2 - best k-coloring
        elite_k_coloring_2_k = np.inf
        elite_k_coloring_2_conflict = np.inf
        elite_k_coloring_2_cost = np.inf

        elite_mscp_2 = np.empty_like(p_a)           # elite theta 2 - best mscp
        elite_mscp_2_k = np.inf
        elite_mscp_2_conflict = np.inf
        elite_mscp_2_cost = np.inf

        for i in range(max_iter):
            print(f"===[iteration {i}]===")

            p_a_conflict = self.conflicts(p_a)
            p_b_conflict = self.conflicts(p_b)

            # crossover
            print("\tgpx")
            c_a = self.cs_operator(p_a, p_b, self.k)
            print(c_a.shape)
            c_b = self.cs_operator(p_b, p_a, self.k)

            # tabucol for decreasing the number of conflicts (tabucol)
            print("\ttabucol")
            c_a, c_a_conflict = tabucol(instance, c_a, self.k, 100)
            c_b, c_b_conflict = tabucol(instance, c_b, self.k, 100)
            print(f"\tc_a : {c_a_conflict}")
            print(f"\tc_a : {self.conflicts(c_a).sum()}")
            print(f"\tc_b : {c_b_conflict}")
            print(f"\tc_b : {self.conflicts(c_b).sum()}")

            # remove the remaining conflicts by adding colors
            print("\tremove conflicts")
            self.remove_conflicts(c_a)
            self.remove_conflicts(c_b)
            print(f"\tc_a : {self.conflicts(c_a).sum()}")
            print(f"\tc_b : {self.conflicts(c_b).sum()}")

            # 2-move tabu search for improving the sum value
            print("\ttabusum")
            c_a, c_a_cost = tabusum(instance, c_a, self.k, 100, 10)
            c_b, c_b_cost = tabusum(instance, c_b, self.k, 100, 10)
            print(f"\tc_a cost : {c_a_cost}")
            print(f"\tc_a conflict : {self.conflicts(c_a).sum()}")
            print(f"\tc_b cost : {c_b_cost}")
            print(f"\tc_b conflict : {self.conflicts(c_b).sum()}")

            print(c_a.shape)
            print(c_b.shape)

            # update elite 1 with the best mscp score
            print("\telite management mscp")
            elite_mscp_1_id = np.argmin([c_a_cost, c_b_cost, elite_k_coloring_1_cost])
            if elite_mscp_1_id == 0:
                elite_mscp_1[:] = c_a[:]
                elite_mscp_1_cost = c_a_cost
            elif elite_mscp_1_id == 1:
                elite_mscp_1[:] = c_b[:]
                elite_mscp_1_cost = c_b_cost

            # update the best solution
            if elite_k_coloring_1_cost < best_cost:
                best_solution[:] = elite_mscp_1[:]
                best_cost = elite_mscp_1_cost

            # WTF ARE THESE INSTRUCTIONS SUPPOSED TO DO ? NOW I KNOW !
            print("\tremove colors")
            p_a = self.remove_colors(c_a)
            p_b = self.remove_colors(c_b)

            print("\telite management k-coloring")
            elite_k_coloring_1_id = np.argmin([self.color_counter(p_a), self.color_counter(p_a), elite_k_coloring_1_cost])
            if elite_k_coloring_1_id == 0:
                elite_k_coloring_1[:] = p_a[:]
                elite_k_coloring_1_k = self.color_counter(p_a)
            elif elite_k_coloring_1_id == 1:
                elite_k_coloring_1[:] = p_b[:]
                elite_k_coloring_1_k = self.color_counter(p_b)

            print("\telite management")
            if elite_k_coloring_1_conflict == 0:
                self.k -= 1
                print(f"\t\tcolor removed")
                # TODO: permut the smallest class (removed) with the higher number (class)

            if i % iter_cycle == 0 or np.all(p_a == p_b):
                p_a = self.remove_colors(elite_mscp_2)

                # permut with elite1 and elite2
                elite_mscp_2 = elite_mscp_1
                elite_mscp_2_cost = elite_mscp_1_cost
                # reset elite1
                elite_mscp_1 = np.random.randint(0, self.k, self.instance.vertex_count)
                elite_mscp_1_cost = create_gamma_sum(elite_mscp_1, self.k, colors_cost)

            if np.all(p_a == p_b):
                p_a[:] = elite_k_coloring_2[:]
                elite_k_coloring_2[:] = elite_k_coloring_1[:]
                elite_k_coloring_1 = np.random.randint(0, self.k, self.instance.vertex_count)

            if np.all(p_a == p_b):
                p_a = np.random.randint(0, self.k, self.instance.vertex_count)
                p_a = tabucol(instance, p_a, self.k, 100)

        return best_solution, best_cost


if __name__ == "__main__":
    graph_name = "anna"
    instance = DIMACS(f"dimacs_instances/{graph_name}.col")
    # instance = DIMACS("dimacs_instances/DSJC250.5.col")

    print(f"== {graph_name} ==")
    print(f"vertex count : {instance.vertex_count}")
    print(f"edege count : {instance.edge_count}")

    solver = sumHEAD(instance, gpx, tabucol)

    best_solution, best_cost = solver.run(1_000)

    print(f"mscp : {best_solution}")
