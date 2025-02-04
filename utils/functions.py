import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
def plotar_conexoes(a):
    node_radius = 0.15 
    n = len(a)
    for servidor in range(n):
        plt.figure(figsize=(8, 3))
        for i in range(n):
            plt.plot(i, 0, 'o', markersize=30, color='lightblue') 
            plt.text(i, 0, str(i), ha='center', va='center', fontweight='bold')
        for destino in range(n):
            peso = a[servidor][destino]
            if peso == 0:
                continue
            cor = 'green' if peso > 0 else 'red' 
            if servidor == destino:
                plt.annotate(
                    "",
                    xy=(servidor, 0),
                    xytext=(servidor, 0),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=cor,
                        shrinkA=0,
                        shrinkB=0,
                        connectionstyle="arc3,rad=0.3",
                        lw=2
                    )
                )
                plt.text(servidor, 0.25, str(peso), ha='center', color=cor, 
                         bbox=dict(facecolor='white', edgecolor='none', pad=2))
            else:
                plt.annotate(
                    "",
                    xytext=(servidor + node_radius, 0),
                    xy=(destino - node_radius, 0), 
                    arrowprops=dict(
                        arrowstyle='->',
                        color=cor,
                        lw=2,
                        shrinkA=0,
                        shrinkB=0
                    )
                )
                pos_x = (servidor + destino) / 2
                if servidor == 0 and destino > 1:
                    pos_x += 0.4
                elif servidor > 1 and destino > servidor:
                    pos_x -= 0.4
                elif destino < n - 2 and servidor == n - 1:
                    pos_x -= 0.4
                elif destino == n - 1 and servidor == n - 1:
                    pos_x += 0.2
                offset = 0.15 if servidor < destino else -0.15
                plt.text(pos_x, offset, str(peso), ha='center', color=cor,
                         bbox=dict(facecolor='white', edgecolor='none', pad=2))
        
        plt.title(f'Conexões do Servidor {servidor}', fontsize=12, pad=2)
        plt.xlim(-0.5, n - 0.5)
        plt.ylim(-0.5, 0.5)
        plt.axis('off')
        plt.show()
def plot_graph(A):
    G = nx.DiGraph()
    num_servers = len(A)
    for i in range(num_servers):
        G.add_node(f"S{i+1}", carga=A[i][i])
        for j in range(num_servers):
            if j != i and A[i][j] != 0:
                G.add_edge(f"S{i+1}", f"S{j+1}")
    pos = nx.spring_layout(G)
    labels = {node: f"{node}\n{G.nodes[node]['carga']}" for node in G.nodes}
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    plt.title("Representação Gráfica dos Servidores")
    plt.show()
def eliminacao_gauss(A, b, max_iter):
    n = len(A)
    for i in range(n):
        max_index = max(range(i, n), key=lambda k: abs(A[k][i]))
        if A[max_index][i] == 0:
            raise ValueError(f"A matriz é singular, não é possível resolver o sistema para a linha {i}.")
        if max_index != i:
            A[i], A[max_index] = A[max_index], A[i]
            b[i], b[max_index] = b[max_index], b[i]
        for k in range(i + 1, n):
            fator = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= fator * A[i][j]
            b[k] -= fator * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - soma) / A[i][i]
    for _ in range(max_iter):
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        delta_x = [0] * n
        for i in range(n - 1, -1, -1):
            soma = sum(A[i][j] * delta_x[j] for j in range(i + 1, n))
            delta_x[i] = (r[i] - soma) / A[i][i]
        x = [x[i] + delta_x[i] for i in range(n)]
    
    return x
def metodo_jacobi(A, b, max_iter, tol=1e-6):
    n = len(A)
    x = [0] * n 
    x_novo = x[:]
    for _ in range(max_iter):
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_novo[i] = (b[i] - soma) / A[i][i]
        if max(abs(x_novo[i] - x[i]) for i in range(n)) < tol:
            return x_novo
        x = x_novo[:]
    return x_novo
def metodo_gauss_seidel(A, b, max_iter, tol=1e-6):
    n = len(A)
    x = [0] * n 
    for _ in range(max_iter):
        x_old = x[:]
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - soma) / A[i][i]
        if max(abs(x[i] - x_old[i]) for i in range(n)) < tol:
            return x
    return x
def analisar_variacao(A, b, max_iteracoes, func):
    solucoes = []
    iteracoes = list(range(1, max_iteracoes + 1))
    for i in iteracoes:
        solucoes.append(func(A, b, i))
    solucoes = np.array(solucoes)
    for i in range(len(A)):
        plt.figure(figsize=(8, 5))
        plt.plot(iteracoes, solucoes[:, i], marker='o', linestyle='-', label=f'x{i+1}')
        plt.xlabel('Número de Iterações')
        plt.ylabel('Valor da Solução')
        plt.title(f'Variação da Solução para x{i+1}')
        plt.legend()
        plt.grid()
        plt.show()     
def Ax(A,x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
def erroAbsoluto(A, b, x):
    return [abs(Ax(A,x)[i] - b[i]) for i in range(len(b))]
def erroRelativo(erroAbsoluto, b):
    return [erroAbsoluto[i] / abs(b[i]) if b[i] != 0 else 0 for i in range(len(b))]