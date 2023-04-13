#load libraries
library(ggplot2)

#import .csv file 
scatter <- read.csv('C:/Users/SaderPC/My Drive/Career/Projects/data_science_challenge/scatter.csv')

#extract the x values
x <- scatter$x

# calculate bin width
q1 <- quantile(x, 0.25)
q3 <- quantile(x, 0.75)
iqr <- q3 - q1
n <- length(x)
bin_width <- 2 * iqr / (n^(1/3))

print(bin_width)

# plot the graph 
histogram_plot <- ggplot(scatter, aes(x=x)) +
                         geom_histogram(binwidth = 6.55, 
                                        color = "black",
                                        fill = "dodgerblue") +
                         labs (titel = "Histogram of x values", x = "X values", y = "Frequency")

print(histogram_plot)
