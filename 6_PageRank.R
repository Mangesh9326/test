library(igraph)

# Adjacency matrix
adj <- matrix(c(
  0, 1, 1, 0,
  0, 0, 1, 0,
  1, 0, 0, 1,
  0, 0, 1, 0
), nrow = 4, byrow = TRUE)

# Node names
node_names <- c("A", "B", "C", "D")

# Create graph
g <- graph_from_adjacency_matrix(adj, mode = "directed")
V(g)$name <- node_names

# PageRank (Google-style, d = 0.85 by default)
V(g)$pagerank <- round(page_rank(g)$vector, 3)

# Plot
plot(
  g,
  layout = layout_with_kk(g),
  vertex.size = 35 + 200 * V(g)$pagerank,
  vertex.label = paste0(V(g)$name, "\nPR: ", V(g)$pagerank),
  vertex.color = "lightblue",
  vertex.frame.color = "darkblue",
  edge.arrow.size = 0.5,
  main = "PageRank (A, B, C, D)"
)
