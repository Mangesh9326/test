import itertools
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt

# Predefined graph with distances (A-F)
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'C': 1, 'D': 7, 'F': 10},
    'C': {'A': 4, 'B': 1, 'D': 3, 'E': 5},
    'D': {'B': 7, 'C': 3, 'E': 2, 'F': 4},
    'E': {'C': 5, 'D': 2, 'F': 6},
    'F': {'B': 10, 'D': 4, 'E': 6}
}

# Function to calculate distance of a path
def calculate_distance(path):
    distance = 0
    for i in range(len(path) - 1):
        distance += graph[path[i]][path[i+1]]
    return distance

# Generate all possible paths
def generate_paths(start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(start)
    path.append(start)

    if start == end:
        yield list(path)
    else:
        for neighbor in graph[start]:
            if neighbor not in visited:
                yield from generate_paths(neighbor, end, visited.copy(), path.copy())

# Ant Colony Optimization with iteration output
def ant_colony_optimization(start, end, iterations=20, n_ants=10, alpha=1, beta=2, evaporation=0.5):
    pheromone = { (u, v): 1.0 for u in graph for v in graph[u] }  # Initialize pheromone

    best_path = None
    best_distance = float('inf')

    print("\n=== ACO Iteration Results ===")
    for it in range(1, iterations + 1):
        paths = []
        distances = []

        for _ in range(n_ants):
            path = []
            visited = set()
            current = start
            path.append(current)
            visited.add(current)

            while current != end:
                neighbors = [n for n in graph[current] if n not in visited]
                if not neighbors:
                    break

                # Probability proportional to pheromone * (1/distance)^beta
                probabilities = []
                for n in neighbors:
                    tau = pheromone[(current, n)] ** alpha
                    eta = (1.0 / graph[current][n]) ** beta
                    probabilities.append(tau * eta)

                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]
                choice = random.choices(neighbors, weights=probabilities, k=1)[0]

                path.append(choice)
                visited.add(choice)
                current = choice

            if path[-1] == end:
                distance = calculate_distance(path)
                paths.append(path)
                distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_path = path

        # Update pheromone
        for (u, v) in pheromone:
            pheromone[(u, v)] *= (1 - evaporation)

        for path, dist in zip(paths, distances):
            for i in range(len(path) - 1):
                pheromone[(path[i], path[i+1])] += 1.0 / dist

        # Print result of this iteration
        print(f"Iteration {it}: Best Path so far = {' -> '.join(best_path)} | Distance = {best_distance}")

    return best_path, best_distance

# Plot graph with best path highlighted
def plot_graph(best_path):
    G = nx.Graph()
    for u in graph:
        for v, d in graph[u].items():
            G.add_edge(u, v, weight=d)

    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d for u, v, d in G.edges(data="weight")})

    if best_path:
        edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color="red")

    plt.title(f"Best Path: {' -> '.join(best_path)}")
    plt.show()

# Main program
if __name__ == "__main__":
    start = input("Enter start node: ").upper()
    end = input("Enter end node: ").upper()

    # Generate all possible paths
    possible_paths = list(generate_paths(start, end))
    distances = [calculate_distance(p) for p in possible_paths]

    # Show in tabular form
    df = pd.DataFrame({
        "Path": [' -> '.join(p) for p in possible_paths],
        "Distance": distances
    })
    print("\nAll Possible Paths:")
    print(df.to_string(index=False))

    # Run ACO
    best_path, best_distance = ant_colony_optimization(start, end)

    print("\nFinal Best Path found by ACO:", " -> ".join(best_path))
    print("Final Best Distance:", best_distance)

    # Plot the graph
    plot_graph(best_path)
