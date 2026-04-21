library(igraph)

#9.1

davis <- read.csv("davis.csv", header = FALSE)

g <- graph.data.frame(davis, directed = FALSE)

plot(g)


#9.2

# Check for bipartite mapping
bipartite.mapping(g)

V(g)$type <- bipartite_mapping(g)$type
plot(g)


#9.3

plot(g, vertex.label.cex = 0.8, vertex.label.color = "black")


#9.4

V(g)$color <- ifelse(V(g)$type, "lightblue", "salmon")
V(g)$shape <- ifelse(V(g)$type, "circle", "square")
E(g)$color <- "lightgray"

plot(g, vertex.label.cex = 0.8, vertex.label.color = "black")


#9.5

V(g)$label.color <- "black"

V(g)$label.cex <- 1
V(g)$frame.color <- "gray"
V(g)$size <- 18

plot(g, layout = layout_with_graphopt)
plot(g, layout = layout.bipartite, vertex.size = 7, vertex.label.cex = 0.6)


