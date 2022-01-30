import itertools

def plot_line_segments(*args, **kwargs):
    return None

# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s=0):
    V = len(graph)
    vertices = [i for i in range(V) if i!=s]
    min_cost = float('inf')
    min_cost_order = None
    for permutation in itertools.permutations(vertices):
        source = s
        path_len = 0
        for destination in permutation:
            path_len += graph[source][destination]
            source = destination
        if path_len < min_cost:
            min_cost = path_len
            min_cost_order = list(permutation)
    return min_cost_order