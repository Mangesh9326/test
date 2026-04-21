optimal_edges <- which.max(mods) - 1
g2 <- delete_edges(g, ebc$removed.edges[seq(length = optimal_edges)])
V(g2)$color <- components(g2)$membership
g$layout <- layout_with_fr(g)

plot(g, vertex.label = NA)
plot(g2, vertex.label = NA)
plot(kite)

as_adj_edge_list(kite, mode = c("all", "out", "in", "total"))
fc <- cluster_fast_greedy(g2)
max_step <- min(which.max(fc$modularity), length(fc$steps))
com <- cut_at(fc, steps = max_step)
V(g2)$color <- fc$membership + 1
g2$layout <- layout_with_fr(g)
plot(g2, vertex.label = NA)

library(igraph)
centr_degree(g, mode="in", normalized=T)

closeness(g, mode = "all", weights = NA)
centr_clo(g, mode = "all", normalized = "T")

betweenness(g, directed = T, weights = NA)
edge_betweenness(g, directed = T, weights = NA)
centr_betw(g, directed = T, normalized = T)

centr_eigen(g, directed = T, normalized = T)
