# Load the library
if(!require(igraph)) install.packages("igraph")
library(igraph)

# --- 1. Create the Two-Mode Data ---
# Let's define memberships: 5 Directors, 3 Committees
relations <- data.frame(
  Director = c("Alice", "Alice", "Bob", "Charlie", "Charlie", "David", "Eve", "Eve"),
  Committee = c("Audit", "Risk", "Audit", "Risk", "Comp", "Comp", "Audit", "Comp")
)

# Create the bipartite graph
g_2mode <- graph_from_data_frame(relations, directed = FALSE)

# We must tell igraph it is bipartite
V(g_2mode)$type <- V(g_2mode)$name %in% relations$Committee

# Plot Two-Mode
plot(g_2mode, 
     layout = layout_as_bipartite, 
     vertex.color = ifelse(V(g_2mode)$type, "salmon", "lightblue"),
     main = "Two-Mode Network: Directors & Committees")

# --- 2. Project into One-Mode Networks ---
# bipartite_projection returns a list of two graphs
projections <- bipartite_projection(g_2mode)

g_persons <- projections$proj1
g_committees <- projections$proj2

# --- 3. Visualize One-Mode Networks ---
par(mfrow=c(1,2)) # Set plot window to side-by-side

# Person-by-Person Sociogram
plot(g_persons, 
     vertex.color = "lightblue", 
     main = "Person-by-Person\n(Shared Committees)")

# Committee-by-Committee Sociogram
plot(g_committees, 
     vertex.color = "salmon", 
     main = "Committee-by-Committee\n(Shared Members)")
