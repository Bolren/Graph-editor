import networkx as nx
import matplotlib.pyplot as plt

def reverse_prufer_code(code):
    for i in range(len(code)):
        code[i] -= 1

    n = len(code) + 2
    extra_code = []
    for i in range(n):
        if i not in code:
            extra_code.append(i)
    adj_matrix = [[0] * n for _ in range(n)]

    while code:
        u = code[0]
        v = extra_code[0]

        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1

        code.pop(0)
        extra_code.pop(0)

        if u not in code:
            extra_code.insert(0, u)
        
    adj_matrix[extra_code[0]][extra_code[1]] = 1
    adj_matrix[extra_code[1]][extra_code[0]] = 1


    for i in range(len(adj_matrix)):
        print(adj_matrix[i], end='\n')
    return adj_matrix

def reverse_prufer_graph(matrix):
    graph = nx.Graph()

    for i in range(len(matrix)):
        graph.add_node(i + 1)
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] > 0:
                graph.add_edge(i + 1, j + 1)

    fig, ax = plt.subplots()
    circ = nx.circular_layout(graph)
    
    nx.draw(graph, pos=circ, with_labels=True, node_color='blue', node_size=500, font_color='white')
    plt.show()


code = list(map(int, input().split(sep=' ')))

prufer_matrix = reverse_prufer_code(code)

reverse_prufer_graph(prufer_matrix)