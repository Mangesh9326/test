library(igraph)

optimal_edges <- which.max(mods) - 1
g2 <- delete_edges(g, ebc$removed.edges[seq_len(optimal_edges)])
V(g2)$color <- components(g2)$membership

g_layout <- layout_with_fr(g)
g2_layout <- layout_with_fr(g2)

plot(g, layout = g_layout, vertex.label = NA)
plot(g2, layout = g2_layout, vertex.label = NA)
plot(kite)

fg <- cluster_fast_greedy(g2)

V(g2)$color <- membership(fg)

plot(g2, layout = g2_layout, vertex.label = NA)

as_adj_edge_list(kite, mode = "all")

centr_degree(g, mode = "in", normalized = TRUE)

closeness(g, mode = "all", weights = NA)
centr_clo(g, mode = "all", normalized = TRUE)

betweenness(g, directed = TRUE, weights = NA)
edge_betweenness(g, directed = TRUE, weights = NA)
centr_betw(g, directed = TRUE, normalized = TRUE)

eigen_centrality(g, directed = TRUE, scale = TRUE)$vector
centr_eigen(g, directed = TRUE, normalized = TRUE)
