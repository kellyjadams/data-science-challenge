#load libraries
library(ggplot2)

#import .csv file 
scatter <- read_csv('C:/Users/SaderPC/My Drive/Career/Projects/data_science_challenge/scatter.csv')

# get the important values
linear_model <- lm(y ~ x, scatter)
summary(linear_model)

slope <- coef(linear_model)[["x"]]
y_intercept <- coef(linear_model)[["(Intercept)"]]

cat("Slope (m):", slope, "\n")
cat("Y-intercept (b):", y_intercept, "\n")

# create dataframe 
scatter_df <- data.frame(scatter)

# plot the graph 
scatter_plot <- ggplot(scatter_df, aes(x=x, y=y)) + 
  geom_point(color = "red") +
  labs (title = "Linear Regression Model",
        x = "x",
        y = "y") +
  geom_smooth(method = lm, se=FALSE, col="blue", lty="dashed")

print(scatter_plot)
