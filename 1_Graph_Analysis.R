install.packages("igraph")

library(igraph)

g <- graph.formula(1-2, 1-3, 2-3, 2-4, 3-5, 4-5, 4-6, 4-7, 5-6, 6-7)

V(g)
E(g)

plot(g)

dg <- graph.formula(1-+2, 1-+3, 2++3)
plot(dg)

dg1 <- graph.formula(Sam-+Mary, Sam-+Tom, Mary++Tom)
plot(dg1)

# Number of vertices
vcount(g)

# Number of edges
ecount(g)

# Degree (FIXED)
igraph::degree(g)

# In-degree
igraph::degree(dg, mode = "in")

# Out-degree
igraph::degree(dg, mode = "out")

# Node with lowest degree
V(dg)$name[igraph::degree(dg) == min(igraph::degree(dg))]

# Node with highest degree
V(dg)$name[igraph::degree(dg) == max(igraph::degree(dg))]

# Neighbors
neighbors(g, 5)
neighbors(g, 2)

# Adjacency list
as_adj_list(dg)
as_adj_list(g)
