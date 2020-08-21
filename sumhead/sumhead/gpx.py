import numpy as np
from time import time
from numba import njit, prange


def gpx(p_a, p_b, color_count):
    p_a_color_count = np.zeros(color_count)
    p_b_color_count = np.zeros(color_count)

    child = np.zeros_like(p_a)

    # compute set size
    for k in range(color_count):
        p_a_color_count[k] = np.count_nonzero(p_a == k)
        p_b_color_count[k] = np.count_nonzero(p_b == k)

    # print("Color Count")
    # print(p_a_color_count)
    # print(p_b_color_count)

    # print("Child building")
    actual_parent = p_a
    actual_parent_color_count = p_a_color_count
    unassigned_vertex = np.ones_like(child, dtype=np.bool_)
    for c in range(color_count):
        # fond higher color count
        k = actual_parent_color_count.argmax()

        # update the child
        mask = actual_parent == k
        mask *= unassigned_vertex
        child[mask] = actual_parent[mask]
        unassigned_vertex[mask] = False

        # remove the color from the parents
        p_a_color_count[k] = 0
        p_b_color_count[k] = 0

        # change parent
        if c % 2 == 0:
            actual_parent = p_a
            actual_parent_color_count = p_a_color_count
        else:
            actual_parent = p_b
            actual_parent_color_count = p_b_color_count

        # print(p_a_color_count)
        # print(p_b_color_count)
        # print(unassigned_vertex)
        # print(child)

    # unassigned colors are assigned randomly
    child[:] += unassigned_vertex * np.random.randint(0, color_count, child.shape[0])

    # print("final state")
    # print(child)

    return child


if __name__ == "__main__":
    k = 5
    size = 20

    runs = 5
    durations = np.zeros(runs)

    for r in range(runs):
        print()
        p_a = np.random.randint(0, k, size, dtype=np.int32)
        p_b = np.random.randint(0, k, size, dtype=np.int32)
        print(p_a)
        print(p_b)
        start = time()
        child = gpx(p_a, p_b, k)
        durations[r] = time() - start
        print(child)

    print(f"duration : {durations}")
