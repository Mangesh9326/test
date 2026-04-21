# Calculate Hamming distance
library(e1071)

x <- c(0, 0, 0, 0)
y <- c(0, 0, 1, 0)
z <- c(1, 0, 1, 1)
w <- c(0, 1, 1, 1)

hamming.distance(x, y)
hamming.distance(y, z)
hamming.distance(x, w)
hamming.distance(z, w)
hamming.distance(x, z)
