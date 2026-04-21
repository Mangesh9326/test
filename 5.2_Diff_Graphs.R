library(igraph)

kite <- make_graph("Krackhardt_Kite")
atri <- count_triangles(kite)
plot(kite, vertex.label = atri)
transitivity(kite, type = "local")
count_triangles(kite) / (degree(kite) * (degree(kite) - 1) / 2)

g1 <- sample_pa(50, power = 1, directed = FALSE)
g2 <- sample_smallworld(dim = 1, size = 100, nei = 5, p = 0.05)

g <- union(g1, g2)
g <- simplify(g)

ebc <- cluster_edge_betweenness(g, directed = FALSE)
plot(ebc, g)

mods <- sapply(0:ecount(g), function(i) {
  g_temp <- delete_edges(g, ebc$removed.edges[seq_len(i)])
  membership <- components(g_temp)$membership
  modularity(g_temp, membership)
})
mods
