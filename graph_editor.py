import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from tkinter import filedialog
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import sys
import copy

vertex_radius = 15

class Graph:
    def __init__(self, adjacency_matrix, vertices):
        self.adjacency_matrix = adjacency_matrix
        self.vertices = vertices
        self.num_vertices = len(vertices)
        self.adjacency_list = self.do_adjacency_list()
        self.edge_weights = copy.deepcopy(self.adjacency_matrix)
        # for i in range(len(self.edge_weights)):
        #     for j in range(len(self.edge_weights)):
        #         self.edge_weights[i][j] = int(self.edge_weights[i][j])
        # print(self.adjacency_list)

    def do_adjacency_list(self):
        adjacency_list = {i + 1: [] for i in range(len(self.adjacency_matrix))}
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix[i])):
                if self.adjacency_matrix[i][j] > 0:
                    adjacency_list[i + 1].append(j + 1)

        return adjacency_list

    def all_simple_chains(self):
        visited = set()
        chains = []
        start_vertex = 1

        def dfs(vertex, path):
            visited.add(vertex)
            path.append(vertex)

            # Рекурсивно обходим всех соседей
            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs(neighbor, path)

            chains.append(path.copy())

            path.pop()
            visited.remove(vertex)

        dfs(start_vertex, [])

        unique_chains = [chains[0]]
        for i in range(1, len(chains)):
            if chains[i] != chains[i-1][:len(chains[i])-len(chains[i-1])]:
                unique_chains.append(chains[i])

        return unique_chains


    def eulerian_cycle(self):
        for edges in self.adjacency_list.values():
            if len(edges) % 2 != 0:
                print("Not euelerian graph")
                return

        stack = []
        path = []
        curr_vertex = 1
        stack.append(curr_vertex)
        while stack:
            if self.adjacency_list[curr_vertex]:
                next_vertex = self.adjacency_list[curr_vertex].pop()
                self.adjacency_list[next_vertex].remove(curr_vertex)
                stack.append(curr_vertex)
                curr_vertex = next_vertex
            else:
                path.append(curr_vertex)
                curr_vertex = stack.pop()
        print(path[::-1])
        return path[::-1]

    def hamiltonian_cycle(self):
        chains = self.all_simple_chains()
        cycles = []
        connect = set([i + 1 for i in range(len(self.adjacency_matrix)) if self.adjacency_matrix[0][i] > 0])

        for chain in chains:
            if chain[-1] in connect and len(chain) == self.num_vertices:
                chain.append(1)
                cycles.append(chain)
        
        if cycles == []:
            print("Not hamiltonian graph")
            return False

        return cycles

    def shortest_paths(self):
        start_vertex = 1
        queue = deque([start_vertex])

        paths = {start_vertex: [start_vertex]}

        while queue:
            current_node = queue.popleft()

            for neighbor in self.adjacency_list[current_node]:
                if neighbor not in paths:
                    paths[neighbor] = paths[current_node] + [neighbor]
                    queue.append(neighbor)

        paths = dict(sorted(paths.items()))
        
        for i in range(1, len(paths) + 1):
            print(f"{i}: {paths[i]}", end='\n')
        # print(paths)
        return paths


    def set_weights(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                self.edge_weights[i][j] = matrix[i][j]


    def dijkstra(self):
        distances = [sys.maxsize] * self.num_vertices  # Инициализируем расстояния до всех вершин как бесконечность
        distances[0] = 0 
        visited = [False] * self.num_vertices

        for _ in range(self.num_vertices):
            min_distance = sys.maxsize
            min_index = -1
            for i in range(self.num_vertices):
                if not visited[i] and distances[i] < min_distance:
                    min_distance = distances[i]
                    min_index = i

            if min_index == -1:
                break  # Если все вершины посещены или недостижимы, завершаем

            visited[min_index] = True  # Помечаем вершину как посещенную

            # Обновляем расстояния до соседних вершин
            for i in range(self.num_vertices):
                if (not visited[i] and self.edge_weights[min_index][i] > 0 and
                    distances[min_index] + self.edge_weights[min_index][i] < distances[i]):
                    distances[i] = int(distances[min_index] + self.edge_weights[min_index][i])
        print(distances)
        return distances


    def kruskal(self):
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [1] * size

            def find(self, p):
                if self.parent[p] != p:
                    self.parent[p] = self.find(self.parent[p])
                return self.parent[p]

            def union(self, p, q):
                rootP = self.find(p)
                rootQ = self.find(q)
                if rootP != rootQ:
                    if self.rank[rootP] > self.rank[rootQ]:
                        self.parent[rootQ] = rootP
                    elif self.rank[rootP] < self.rank[rootQ]:
                        self.parent[rootP] = rootQ
                    else:
                        self.parent[rootQ] = rootP
                        self.rank[rootP] += 1
                    return True
                return False
            
        edges = []
        
        # Собираем все рёбра из матрицы смежности
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                if self.edge_weights[i][j] != 0:
                    edges.append((self.edge_weights[i][j], i, j))
        
        edges.sort()
        
        uf = UnionFind(self.num_vertices)
        mst = []  # Список рёбер минимального остовного дерева
        
        for weight, u, v in edges:
            if uf.union(u, v):
                mst.append((u, v, weight))
                if len(mst) == self.num_vertices - 1:
                    break
        
        mst_matrix = [[0] * self.num_vertices for _ in range(self.num_vertices)]
        for u, v, weight in mst:
            mst_matrix[u][v] = weight
            mst_matrix[v][u] = weight
        
        return mst_matrix

    def prufer_code(self):
        code = []
        tree = self.kruskal() 
        for i in range(len(tree)):
            for j in range(len(tree)):
                if tree[i][j] > 0:
                    tree[i][j] = 1
    
        # Создаем список степеней вершин
        degrees = [sum(row) for row in tree]

        # Ищем листья и добавляем их в очередь
        leaves = [i for i in range(self.num_vertices) if degrees[i] == 1]

        for _ in range(self.num_vertices - 2):
            leaf = min(leaves)
            leaves.remove(leaf)

            # Находим соседа листа
            for i in range(self.num_vertices):
                if tree[leaf][i] == 1:
                    neighbor = i
                    tree[leaf][i] = 0
                    tree[i][leaf] = 0
                    break            
            
            code.append(neighbor + 1)
            
            # Уменьшаем степень соседа
            degrees[neighbor] -= 1
            if degrees[neighbor] == 1:
                leaves.append(neighbor)
        
        print(code)
        return code
    
    def greed_tsp(self):
        cycles = self.hamiltonian_cycle()
        if cycles == False:
            print("Unable to solve TSP")
            return 0
        min_sum = float("inf")
        min_cycle = []

        for cycle in cycles:
            weights_sum = 0
            for i in range(len(cycle) - 1):
                u = cycle[i] - 1
                v = cycle[i + 1] - 1
                weights_sum += self.edge_weights[u][v]
            if weights_sum < min_sum:
                min_sum = weights_sum
                min_cycle = cycle

        print(min_sum)
        print(min_cycle)
        return min_sum
    
    def coloring1(self):
        colors = [[],[],[]]
        vertices = [i + 1 for i in range(self.num_vertices)]

        for color in colors:
            for u in vertices:
                flag = 0
                if color == []:
                    color.append(u)
                    continue
                for v in color:
                    if u in self.adjacency_list[v]:
                        flag = 1
                        break
                if flag == 1:
                    continue
                color.append(u)
            vertices = [item for item in vertices if item not in color]

        for i in range(len(colors)):
            print(f"Color {i + 1}: {colors[i]}",end='\n')
        return colors
    
    def coloring2(self):
        colors = [[],[],[]]

        for u in range(self.num_vertices):
            for color in colors:
                flag = 0
                if color == []:
                    color.append(u + 1)
                    break
                for v in color:
                    if (u + 1) in self.adjacency_list[v]:
                        flag = 1
                        break
                if flag == 0:
                    color.append(u + 1)
                    break


        for i in range(len(colors)):
            print(f"Color {i + 1}: {colors[i]}",end='\n')
        return colors
    
    def floyd_shortest_paths(self):
        n = self.num_vertices
        
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        for i in range(n):
            for j in range(n):
                if self.edge_weights[i][j] > 0:
                    dist[i][j] = self.edge_weights[i][j]
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        for i in range(n):
            print(dist[i],sep='\n')
        return dist
    

class GraphEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Editor")

        global vertex_radius
        self.graph = nx.Graph()
        self.graph_obj = None
        self.pos = {}  # Позиции вершин для отображения
        self.current_vertex = None
        self.temp_line = None  # Временная линия для визуализации создания ребра
        self.edges = {}  # Словарь для хранения рёбер и их идентификаторов на холсте
        self.v_r = vertex_radius

        self.exit_button = tk.Button(root, text="Exit", command=self.root.quit)
        self.exit_button.pack()

        self.canvas = tk.Canvas(root, width=600, height=400, bg="white")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas.bind("<Button-1>", self.add_vertex)  # Левый клик — добавление вершины
        self.canvas.bind("<Button-3>", self.start_edge)  # Правый клик — начало создания ребра
        self.canvas.bind("<B3-Motion>", self.update_temp_line)  # Движение мыши с зажатой правой кнопкой
        self.canvas.bind("<ButtonRelease-3>", self.add_edge)  # Отпускание правой кнопки — завершение создания ребра
        self.canvas.bind("<Button-2>", self.delete_element)  # Средний клик — удаление вершины или ребра

        # Кнопка для отрисовки графа
        self.draw_button = tk.Button(root, text="Draw Graph", command=self.draw_graph)
        self.draw_button.pack(side=LEFT)

        # Кнопка для очистки графа
        self.clear_button = tk.Button(root, text="Clear Graph", command=self.clear_graph)
        self.clear_button.pack(side=LEFT)

        # Кнопка для создания матрицы смежности
        self.adjacency_matrix_button = tk.Button(root, text="Show Adjacency Matrix", command=self.show_adjacency_matrix)
        self.adjacency_matrix_button.pack(side=LEFT)

        # Создаем кнопку для чтения из файла
        self.input_button = tk.Button(root, text="Select file", command=self.input_data)
        self.input_button.pack(side=BOTTOM)

        # Инициализация переменных для хранения текущих виджетов и окон
        self.current_canvas_window = None
        self.current_matrix_window = None
        self.current_weights_window = None

    def add_vertex(self, event):
        x, y = event.x, event.y
        vertex = len(self.graph.nodes) + 1
        self.graph.add_node(vertex, pos=(x, y))
        self.pos[vertex] = (x, y)
        self.canvas.create_oval(x - self.v_r, y - self.v_r, x + self.v_r, y + self.v_r, fill="blue")
        self.canvas.create_text(x, y, text=vertex, fill="white")

    def start_edge(self, event):
        x, y = event.x, event.y
        for vertex, (vx, vy) in self.pos.items():
            if (x - vx) ** 2 + (y - vy) ** 2 <= self.v_r ** 2:  # Проверка, что клик был рядом с вершиной
                self.current_vertex = vertex
                # Создаём временную линию
                self.temp_line = self.canvas.create_line(vx, vy, x, y, fill="black", dash=(4, 2))
                return

    def update_temp_line(self, event):
        if self.current_vertex and self.temp_line:
            x, y = event.x, event.y
            vx, vy = self.pos[self.current_vertex]
            self.canvas.coords(self.temp_line, vx, vy, x, y)

    def add_edge(self, event):
        if self.temp_line:
            self.canvas.delete(self.temp_line)  # Удаляем временную линию
            self.temp_line = None

        x, y = event.x, event.y
        for vertex, (vx, vy) in self.pos.items():
            if (x - vx) ** 2 + (y - vy) ** 2 <= self.v_r ** 2:
                if self.current_vertex and self.current_vertex != vertex:
                    self.graph.add_edge(self.current_vertex, vertex)
                    line_id = self.canvas.create_line(self.pos[self.current_vertex], (vx, vy), fill="black")
                    self.edges[(self.current_vertex, vertex)] = line_id
                    self.current_vertex = None
                return

    def delete_element(self, event):
        """Удаляет вершину или ребро по клику."""
        x, y = event.x, event.y

        # Проверяем, был ли клик на вершине
        for vertex, (vx, vy) in self.pos.items():
            if (x - vx) ** 2 + (y - vy) ** 2 <= self.v_r ** 2:  # Проверка, что клик был рядом с вершиной
                self.delete_vertex(vertex)
                return

        # Проверяем, был ли клик на ребре
        for edge, line_id in self.edges.items():
            v1, v2 = edge
            x1, y1 = self.pos[v1]
            x2, y2 = self.pos[v2]
            # Проверяем, находится ли точка (x, y) на линии ребра
            if self.is_point_on_line(x, y, x1, y1, x2, y2):
                self.delete_edge(edge)
                return

    def delete_vertex(self, vertex):
        """Удаляет вершину и все связанные с ней рёбра."""
        self.graph.remove_node(vertex)
        del self.pos[vertex]
        edges_to_remove = [edge for edge in self.edges if vertex in edge]
        for edge in edges_to_remove:
            self.canvas.delete(self.edges[edge])
            del self.edges[edge]

        self.renumber_vertices()
        self.rebuild_edges()

        self.canvas.delete("all")
        for v, (vx, vy) in self.pos.items():
            self.canvas.create_oval(vx - self.v_r, vy - self.v_r, vx + self.v_r, vy + self.v_r, fill="blue")
            self.canvas.create_text(vx, vy, text=v, fill="white")
        for edge in self.graph.edges:
            v1, v2 = edge
            x1, y1 = self.pos[v1]
            x2, y2 = self.pos[v2]
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="black")
            self.edges[(v1, v2)] = line_id

    def delete_edge(self, edge):
        """Удаляет ребро."""
        v1, v2 = edge
        self.graph.remove_edge(v1, v2)
        self.canvas.delete(self.edges[edge])
        del self.edges[edge]

    def is_point_on_line(self, x, y, x1, y1, x2, y2, tolerance=5):
        """Проверяет, находится ли точка (x, y) на линии между (x1, y1) и (x2, y2)."""
        # Расстояние от точки до линии
        dx = x2 - x1
        dy = y2 - y1
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return (x - x1) ** 2 + (y - y1) ** 2 <= tolerance ** 2
        t = ((x - x1) * dx + (y - y1) * dy) / length_squared
        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        distance_squared = (x - closest_x) ** 2 + (y - closest_y) ** 2
        return distance_squared <= tolerance ** 2

    def renumber_vertices(self):
        """Перенумеровывает вершины, чтобы их номера шли по порядку, начиная с 1."""
        vertices = list(self.graph.nodes)
        vertices.sort()
        new_pos = {}
        new_graph = nx.Graph()
        for i, vertex in enumerate(vertices):
            new_vertex = i + 1
            new_pos[new_vertex] = self.pos[vertex]
            new_graph.add_node(new_vertex, pos=new_pos[new_vertex])
        # Добавляем рёбра в новый граф
        for edge in self.graph.edges:
            v1, v2 = edge
            new_v1 = vertices.index(v1) + 1
            new_v2 = vertices.index(v2) + 1
            new_graph.add_edge(new_v1, new_v2)
        # Обновляем граф и позиции
        self.graph = new_graph
        self.pos = new_pos

    def rebuild_edges(self):
        """Перестраивает словарь рёбер после перенумерации вершин."""
        self.edges = {}
        for edge in self.graph.edges:
            v1, v2 = edge
            x1, y1 = self.pos[v1]
            x2, y2 = self.pos[v2]
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="black")
            self.edges[(v1, v2)] = line_id

    def draw_graph(self):
        if not self.graph.nodes:
            messagebox.showinfo("Info", "No vertices to draw!")
            return

        # Очистка предыдущего графика, если он существует
        if self.current_canvas_window:
            self.current_canvas_window.destroy()

        # Создание нового графика
        # correct_pos = {}
        # for vertex, (vx, vy) in self.pos.items():
        #     correct_pos[vertex] = (vx, -vy)
        # nx.draw(self.graph, pos=correct_pos, with_labels=True, ax=ax, node_color='blue', node_size=500, font_color='white')

        canvas_window = tk.Toplevel(self.root)
        canvas_window.title("Graph")

        fig, ax = plt.subplots()
        circ = nx.circular_layout(self.graph)
        
        nx.draw(self.graph, pos=circ, with_labels=True, node_color='blue', node_size=500, font_color='white')
        if self.graph_obj != None:
            labels = {}
            for i in range(len(self.graph_obj.edge_weights)):
                for j in range(len(self.graph_obj.edge_weights)):
                    if self.graph_obj.edge_weights[i][j] > 0:
                        labels.update({(i + 1, j + 1): self.graph_obj.edge_weights[i][j]})
            nx.draw_networkx_edge_labels(self.graph, pos=circ, edge_labels=labels, font_size=10)

        canvas = FigureCanvasTkAgg(fig, master=canvas_window)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)

        # Сохранение текущего виджета для последующей очистки
        # self.current_canvas_window = canvas_window

    def clear_graph(self):
        if self.current_matrix_window:
            self.current_matrix_window.destroy()
        
        self.graph.clear()
        self.graph_obj = None
        self.pos.clear()
        self.edges.clear()
        self.canvas.delete("all")

    def show_adjacency_matrix(self):
        if not self.graph.nodes:
            messagebox.showinfo("Info", "No vertices to create adjacency matrix!")
            return

        if self.current_matrix_window:
            self.current_matrix_window.destroy()
        if self.current_weights_window:
            self.current_weights_window.destroy()

        vertices = sorted(list(self.graph.nodes))
        adjacency_matrix = [[0] * len(vertices) for _ in range(len(vertices))]

        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Adjacency Matrix")

        for edge in self.graph.edges:
            v1, v2 = edge
            i = vertices.index(v1)
            j = vertices.index(v2)
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1

        # заголовки строк и столбцов
        for i, vertex in enumerate(vertices):
            label = tk.Label(matrix_window, text=vertex, width=4, borderwidth=1)
            label.grid(row=0, column=i + 1)
            label = tk.Label(matrix_window, text=vertex, width=4, borderwidth=1)
            label.grid(row=i + 1, column=0)

        for i, row in enumerate(adjacency_matrix):
            for j, value in enumerate(row):
                label = tk.Label(matrix_window, text=str(value), width=4, borderwidth=1)
                label.grid(row=i + 1, column=j + 1)

        # Создаём объект графа для алгоритмов
        self.graph_obj = Graph(adjacency_matrix, vertices)

        def print_chains():
            unique_chains = self.graph_obj.all_simple_chains()
            for i in range(len(unique_chains)):
                print(unique_chains[i], end='\n')

        simple_chains_button = tk.Button(matrix_window, text="All unique simple chains", command=print_chains)
        simple_chains_button.grid(row=0, column=len(adjacency_matrix[0]) + 1)

        eulerian_cycle_button = tk.Button(matrix_window, text="Eulerian cycle", command=self.graph_obj.eulerian_cycle)
        eulerian_cycle_button.grid(row=1, column=len(adjacency_matrix[0]) + 1)

        def print_hamilton():
            cycles = self.graph_obj.hamiltonian_cycle()
            for i in range(len(cycles)):
                print(cycles[i], end='\n')

        hamiltonian_cycle_button = tk.Button(matrix_window, text="Hamiltonian cycle", command=print_hamilton)
        hamiltonian_cycle_button.grid(row=2, column=len(adjacency_matrix) + 1)

        shortest_paths_button = tk.Button(matrix_window, text="Shortest paths", command=self.graph_obj.shortest_paths)
        shortest_paths_button.grid(row=3, column=len(adjacency_matrix) + 1)

        weights_button = tk.Button(matrix_window, text="Input weights", command=self.input_weights)
        weights_button.grid(row=len(adjacency_matrix) + 1, column=len(adjacency_matrix) + 1)

        coloring1_button = tk.Button(matrix_window, text="Show coloring1", command=self.graph_obj.coloring1)
        coloring1_button.grid(row=4, column=len(adjacency_matrix) + 1)

        coloring2_button = tk.Button(matrix_window, text="Show coloring2", command=self.graph_obj.coloring2)
        coloring2_button.grid(row=5, column=len(adjacency_matrix) + 1)

        self.current_matrix_window = matrix_window

    def input_data(self):
        filepath = filedialog.askopenfilename(initialdir="C:\\Users\\dshte\\Downloads\\diskra")
        file = open(filepath, "r")

        info = file.read().split(sep='\n')

        graph = info[0].split(";")
        for i in range(len(graph)):
            graph[i] = graph[i].split(",")

            for num in graph[i]:
                vertex = int(num)
                self.graph.add_node(vertex)
                self.graph.add_edge(i + 1, vertex)

        self.show_adjacency_matrix()
        matrix_size = len(self.graph_obj.adjacency_matrix)

        connections = info[2].split(";")
        for i in range(matrix_size):
            connections[0] = connections[0].split(",")
            for j in range(matrix_size):
                if self.graph_obj.adjacency_matrix[i][j] > 0:
                    if int(connections[0][0]) < 0:
                        self.graph_obj.adjacency_matrix[i][j] = -1
                    connections[0].pop(0)
            connections.pop(0)
        self.graph_obj.edge_weights = copy.deepcopy(self.graph_obj.adjacency_matrix)
        self.graph_obj.adjacency_list = self.graph_obj.do_adjacency_list()

        weights = info[1].split(";")
        for i in range(matrix_size):
            if weights:
                weights[0] = weights[0].split(",")
            else:
                break
            for j in range(i, matrix_size):
                if weights[0]:
                    weight = int(weights[0][0])
                else:
                    break

                if self.graph_obj.edge_weights[i][j] > 0:
                    self.graph_obj.edge_weights[i][j] = weight
                    weights[0].pop(0)

                if self.graph_obj.edge_weights[j][i] > 0:
                    self.graph_obj.edge_weights[j][i] = weight
                    weights[0].pop(0)

            weights.pop(0)

        return


    def input_weights(self):
        matrix_size = len(self.graph_obj.adjacency_matrix)
        entries = [[] for _ in range(matrix_size)]

        if self.current_weights_window:
            self.current_weights_window.destroy()

        self.weights_window = tk.Toplevel()
        self.weights_window.title("Input Weights")

        for i in range(matrix_size):
            label = tk.Label(self.weights_window, text=f"{i+1}", width=4, borderwidth=1)
            label.grid(row=i + 1, column=0)
            label = tk.Label(self.weights_window, text=f"{i+1}", width=4, borderwidth=1)
            label.grid(row=0, column=i + 1)
            count = 0
            for j in range(matrix_size):
                if j > i and self.graph_obj.adjacency_matrix[i][j] == 1:
                    entries[i].append((j, tk.Entry(self.weights_window, width=5)))
                    entries[i][count][1].grid(row=i + 1, column=j + 1)
                    count+=1

        input_button = tk.Button(self.weights_window, text="Input", command= lambda: self.set_weights(entries))
        input_button.grid(row=matrix_size + 1, column=0, columnspan=matrix_size + 1)

        dijkstra_button = tk.Button(self.weights_window, text="Dijkstra shortest path", command=self.graph_obj.dijkstra)
        dijkstra_button.grid(row=0, column=matrix_size + 1)

        kruskal_button = tk.Button(self.weights_window, text="Kruskal spanning tree", command=self.make_kruskal)
        kruskal_button.grid(row=1, column=matrix_size + 1)

        greed_tsp_button = tk.Button(self.weights_window, text="Greed TSP", command=self.graph_obj.greed_tsp)
        greed_tsp_button.grid(row=3, column=matrix_size + 1)

        floyd_button = tk.Button(self.weights_window, text="Floyd paths", command=self.graph_obj.floyd_shortest_paths)
        floyd_button.grid(row=4, column=matrix_size + 1)

        self.current_weights_window = self.weights_window

    def set_weights(self, entries):
        matrix_size = len(self.graph_obj.adjacency_matrix)
        weights_matrix = [[0] * matrix_size for _ in range(matrix_size)]
        for i in range(matrix_size):
            for j in range(len(entries[i])):
                if entries[i][j][1].get() in "0123456789":
                    weights_matrix[i][entries[i][j][0]] = int(entries[i][j][1].get()) if entries[i][j][1].get() != '' else 1
                    weights_matrix[entries[i][j][0]][i] = int(entries[i][j][1].get()) if entries[i][j][1].get() != '' else 1

        self.graph_obj.set_weights(weights_matrix)

        messagebox.showinfo("Info", "Weights set")       

    def make_kruskal(self):
        self.graph.clear()

        matrix_size = len(self.graph_obj.adjacency_matrix)
        kruskal_matrix = self.graph_obj.kruskal()

        for i in range(matrix_size):
            self.graph.add_node(i + 1)
        for i in range(matrix_size):
            for j in range(matrix_size):
                if kruskal_matrix[i][j] > 0:
                    self.graph.add_edge(i + 1, j + 1)

        self.prufer_button = tk.Button(self.weights_window, text="Create Prufer code", command=self.graph_obj.prufer_code)
        self.prufer_button.grid(row=2, column=matrix_size + 1)

        messagebox.showinfo("Info", "Kruskal panning tree has been made")


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphEditor(root)
    root.mainloop()
