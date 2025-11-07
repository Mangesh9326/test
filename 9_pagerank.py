import matplotlib.pyplot as plt
import networkx as nx


def get_graph_input():
    graph = {}
    print("Enter the graph (one node per line). Format: Node: Neighbor1 Neighbor2 ...")
    print("Type 'done' when finished.\n")
    while True:
        line = input("Enter node and neighbors: ").strip()
        if line.lower() == 'done':
            break
        if ':' not in line:
            print("Invalid format. Use format: Node: Neighbor1 Neighbor2")
            continue
        node, neighbors = line.split(':', 1)
        node = node.strip()
        neighbors = neighbors.strip().split()
        graph[node] = neighbors
    return graph


def initialize_pagerank(graph):
    num_nodes = len(graph)
    return {node: 1 / num_nodes for node in graph}


def compute_pagerank(graph, damping_factor=0.85, max_iterations=100, tol=1.0e-6):
    pagerank = initialize_pagerank(graph)
    num_nodes = len(graph)

    print("\nPageRank Iterations:")
    print("---------------------")

    for iteration in range(max_iterations):
        new_pagerank = {}
        for node in graph:
            rank_sum = 0
            for other_node in graph:
                if node in graph[other_node]:
                    rank_sum += pagerank[other_node] / len(graph[other_node])

            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Display current iteration's PageRank values
        print(f"Iteration {iteration + 1}:")
        for node in sorted(new_pagerank):
            print(f"  {node}: {new_pagerank[node]:.6f}")
        print()

        # Check convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in graph)
        if diff < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

        pagerank = new_pagerank

    return pagerank


def draw_graph(graph):
    G = nx.DiGraph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, arrows=True,
            node_color='skyblue', node_size=2000,
            font_size=12, edge_color='gray')
    plt.title("Graph Structure")
    plt.show()


def main():
    graph = get_graph_input()
    if not graph:
        print("Graph is empty. Exiting.")
        return

    draw_graph(graph)
    pagerank = compute_pagerank(graph)

    print("\nFinal PageRank values:")
    for node, rank in sorted(pagerank.items()):
        print(f"{node}: {rank:.4f}")


if __name__ == "__main__":
    main()
