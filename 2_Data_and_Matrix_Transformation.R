library(igraph)

nodes <- read.csv("Dataset1-Media-Example-NODES.csv", header=T, , as.is=T)
head(nodes)

links <- read.csv("Dataset1-Media-Example-EDGES.csv", header=T, as.is=T)
head(links)

net <- graph_from_data_frame(d=links, vertices=nodes, directed=T)

m=as.matrix(net)
as_adjacency_matrix(net)

plot(net)
