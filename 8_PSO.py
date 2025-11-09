import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)


class Particle:
    def __init__(self, n_clusters, n_features, data_min, data_max):
        self.position = np.random.uniform(data_min, data_max, (n_clusters, n_features))
        self.velocity = np.random.uniform(-0.1, 0.1, (n_clusters, n_features))
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        """Update the particle's velocity based on inertia, cognitive, and social components"""
        inertia = w * self.velocity

        r1 = np.random.random(self.position.shape)
        cognitive = c1 * r1 * (self.best_position - self.position)

        r2 = np.random.random(self.position.shape)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = inertia + cognitive + social

    def update_position(self, data_min, data_max):
        """Update particle position and clamp to the data range"""
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, data_min, data_max)


class PSOClustering:
    def __init__(self, n_clusters=3, n_particles=10, max_iter=100):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.particles = []
        self.cluster_centers = None

    def fitness(self, particle, data):
        """Calculate clustering fitness as total within-cluster distance"""
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((data - particle.position[i]) ** 2, axis=1))

        closest_cluster = np.argmin(distances, axis=1)

        fitness_value = 0
        for i in range(self.n_clusters):
            cluster_points = data[closest_cluster == i]
            if len(cluster_points) > 0:
                # Sum of Euclidean distances
                fitness_value += np.sum(np.sqrt(np.sum((cluster_points - particle.position[i]) ** 2, axis=1)))

        return fitness_value

    def fit(self, data):
        """Run PSO algorithm to optimize cluster centers"""
        n_features = data.shape[1]
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)

        # Initialize particles
        self.particles = [
            Particle(self.n_clusters, n_features, data_min, data_max)
            for _ in range(self.n_particles)
        ]

        self.global_best_fitness = float('inf')

        for iteration in range(self.max_iter):
            for particle in self.particles:
                current_fitness = self.fitness(particle, data)

                if current_fitness < particle.best_fitness:
                    particle.best_fitness = current_fitness
                    particle.best_position = particle.position.copy()

                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = particle.position.copy()

            # Update velocity and position of particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(data_min, data_max)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}")

        self.cluster_centers = self.global_best_position
        return self

    def predict(self, data):
        """Assign data points to nearest PSO-optimized cluster centers"""
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((data - self.cluster_centers[i]) ** 2, axis=1))
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    # Generate synthetic dataset with 3 clusters
    X, true_labels = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

    pso = PSOClustering(n_clusters=3, n_particles=20, max_iter=100)
    pso.fit(X)

    predicted_labels = pso.predict(X)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', alpha=0.7)
    plt.scatter(
        pso.cluster_centers[:, 0],
        pso.cluster_centers[:, 1],
        c='red',
        marker='X',
        s=200,
        label='Cluster Centers'
    )
    plt.title('PSO Clustering Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pso_clustering_results.png')
    plt.show()
