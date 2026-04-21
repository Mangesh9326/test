library(igraph)

# Network as a Sociogram (Network Graph)

ng <- graph.formula(
  Andy++Garth, Garth-+Bill, Bill-+Elena, Elena++Frank, 
  Carol-+Andy, Carol-+Elena, Carol++Dan, Carol++Bill, 
  Dan++Andy, Dan++Bill
)

plot(ng)

# Network as a Matrix (Adjacency Matrix)

net_matrix <- graph.formula(
  Ava -+ Bob, Ava -+ Carl, Bob -+ Carl, Carl -+ Dave, Dave -+ Ava
)

adj_matrix <- as_adjacency_matrix(net_matrix, sparse = FALSE)

adj_matrix

# Network as an Edge List

net_edges <- graph.formula(
  Hope ++ Tom, Tom ++ Gary, Gary ++ Sam, Sam ++ Hope, Hope ++ Gary
)

edge_list <- as_edgelist(net_edges)

edge_list
