library(igraph)

g <- erdos.renyi.game(10, 0.3, directed = TRUE)
gl <- sample_gnp(10, 0.3, directed = TRUE)

# Updated function
den <- edge_density(g)
den

deg <- degree(g)
deg

rec <- reciprocity(g)
rec

tra <- transitivity(g, type = "global")
tra

plot(g)