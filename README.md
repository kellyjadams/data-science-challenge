# Data Ccience Challenge

## Question 1

There is a laundry bin with 20% blue socks.
You randomly draw 10.
What is the probability that 2 are blue?

### Answer

The probability that 2 are blue is: **0.3020**

### Explanation

We use a binomial distribution because the probability for $x$ successes of an experiment (2 blue socks) in $n$ trials (10 socks are drawn), given a success probability of $p$ for each trial (0.2) at the experiment. 

Using binomial distribution formula (Bernoulli trial):
$P(X=k)= \binom{n}{k} p^{x}(1-p)^{n-x}$
Where:

- $n$ is the number of trials (10 socks are drawn), $n=10$
- $k$ is the number of successes (2 blue socks), $k=2$
- $p$ is the probability of success on any given trial (0.2), $p = 0.2$

Using the formula to calculate the probability:
$P(X=2) = \binom{10}{2} 0.2^{2}(1-0.2)^{10-2}$
$P(X=2) = (45) 0.2^{2}(0.8)^{8}$
$P(X=2) = 0.3020$

## Question 2

Out of a set of 1000 emails, 700 are spam.
400 of the emails have the word "free"; 300 of those are spam.
100 of the emails have the word "credit"; 90 of those are spam.
You get an email that contains both "free" and "credit".
What is the probability it is spam?

### Answer

The probability it is spam is **96.8%**.

### Explanation

We are going to use Bayes' Theorem because we need to calculate the conditional probability of an event, based on the occurrence of other events. 

Bayes' Theorem is: 
$$P(A|B) = \frac{P(A) \times P(B |A)}{P(B)}$$

where:

- $P(A)$ = The probability of A occurring
- $P(B)$ = The probability of B occurring
- $P(A|B)$ = The probability of A given B
- $P(B|A)$ = The probability of B given A    

For this given problem let's define the following events:

- A: Email is spam
- B: Email contains the word "free"
- C: Email contains the word "credit"

We are given:

- $P(A) = 700/1000 = 0.7$ (probability that an email is spam)
- $P(B) = 400/1000 = 0.4$ (probability that an email contains the word "free")
- $P(C) = 100/1000 = 0.1$  (probability that an email contains the word "credit")
- $P(B|A) = 300/700 = 0.429$  (probability that an email contains the word "free" given that it is spam)
- $P(C|A) = 90/700 = 0.129$  (probability that an email contains the word "credit" given that it is spam)

We want to find $P(A|B∩C)$, the probability of the email is spam given that it contains both "free" and "credit". The equation, using **Bayes' Theorem** can be written as: 
$$P(A|B∩C) = \frac{P(B∩C|A) \times P(A)}{P(B∩C)}$$

 We need to first find the probability of $P(B∩C|A)$ which is the probability of the email containing both "free" and "credit" given that it is spam.

$P(B∩C|A) = P(B|A) \times P(C|A)$ (assuming independence between B and C given A
$P(B∩C|A) = (300/700) \times (90/700)$
$P(B∩C|A) = 0.429 \times 0.129$

Next we find $P(B∩C)$, which is the probability that an email contains both "free" and "credit". 

$P(B∩C) = P(B) \times P(C)$
$P(B∩C) = (400/1000) \times (100/1000)$
$P(B∩C) = (0.4) \times (0.1)$

Now we can plug all of these values into Baye's theorem with the equation:
$P(A|B∩C) = \frac{(0.429 \times 0.129) \times (0.7)}{(0.4 \times 0.1)}$
$P(A|B∩C) = \frac{0.0387}{0.4}$
$P(A|B∩C) = 0.9675$

## Question 3

Using `scatter.csv`: how do you interpret the linear regression?

### Answer

I found the following values for the linear regression based on `scatter.csv`]: 

- Slope: The slope is -0.006432585. This implies a weak negative relationship between x and y. 
- R squared: The R-squared value is 0.005559. This implies that the linear model is a poor fit for the data because the majority of the variability in y isn't captured by the relationship with x. 
- P-value: The p-value is 0.461 which is generally considered not statistically significant. We can't confidently conclude that there's a relationship between x and y. 

In conclusion, the linear regression model suggests a very weak negative relationship between x and y. The low R-squared value and the high p-value indicate that the model doesn't explain the variance in the data and the relationship isn't statistically significant. 


### Explanation 

I used R to create a linear regression. Below is the code: 

```R
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
```

View the code [q3_scatter.R](https://github.com/kellyjadams/data-science-challenge/blob/main/q3_scatter.R).

We get the following values using the `lm` function in R. 

- Intercept: 0.105960
- R squared: 0.005559
- Slope: -0.006432585
- P value: 0.461

## Question 4

Using `scatter.csv`: Plot a histogram of the x values.

### Answer

View the histogram in the image: `q4_histogram.png` attached. 

### Explanation

I used R to create my histogram. Specifically the ggplot 2 package. Below is the R code. 

```
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
```

The bin width out of this calculation is: 6.55.

View the [q4_histogram.R](https://github.com/kellyjadams/data-science-challenge/blob/main/q4_histogram.R). 

## Question 5

SQL: There are two tables.
One table contains a list of tests and the other contains the positive results of the tests.
Positives and tests are matched by their ID.
There is at most one positive matching to a test.

Tables:

```
test:
	column: name = 'id', type = int
	column: name = timestamp, type = timestamp
	
positive:
	column: name = 'id', type = int
```

Write an SQL query that shows the daily positive rate.

Example output of your query would look something like:

```
| date       | rate |
----------------------
| 2022-01-01 | 0.23 |
| 2022-01-02 | 0.33 |
... etc.
```

### Answer 

```sql
WITH daily_tests AS (
    SELECT DATE(timestamp) AS test_date, COUNT(id) AS total_tests
    FROM test
    GROUP BY test_date
),
daily_positives AS (
    SELECT DATE(t.timestamp) AS positive_date, COUNT(p.id) AS total_positives
    FROM test t
    JOIN positive p ON t.id = p.id
    GROUP BY positive_date
)
SELECT dt.test_date,
       (dp.total_positives * 100.0) / dt.total_tests AS positive_rate
FROM daily_tests dt
LEFT JOIN daily_positives dp ON dt.test_date = dp.positive_date
ORDER BY dt.test_date;
```

### Explanation

- The `daily_tests` CTE calculates the number of tests conducted each day (by using `DATE` ) and using `COUNT` to count the number of ids. It is grouped by the test date.
- The `daily_positives` CTE calculates the number of positive tests for each day (by using `DATE`) and using `COUNT` to count the number of ids. It is also grouped by the positive date. 
- Then the main query joins the two CTEs using a `LEFT JOIN`. Which returns all of the rows from the `daily_tests` and the matching rows from `positive_tests`. The join matches on the `id` of both tables. 
- Finally the positive rate is calculated by dividing the total positives by the total tests, and multiplying by 100 to get the percentage. 

## Challenge 6

Train a prediction model with `train.csv` using the final column as the labels.
Use this model to create labels for the entries in `test.csv`.
Though the labels in `train.csv` are 0/1, the labels created for `test.csv` can be continuous [0 to 1].
Output should be a file called `labels.txt`.
Each line of `labels.txt` is a predicted label for the corresponding line in `test.csv`.
For example, `labels.txt` line 6 will be the predicted label for line 6 in `test.csv`.
The contents of `labels.txt` will look something like:

```bash
0.5355877
0.128361
0.2841359
0.4216329
...etc
```

The quality of the predictions will be judged by RMSE.
Please include the source code used to create your model as well as a write-up of your technique.

### Answer

I used Python to train the prediction model. I used a **random forest model** Below is the updated code: 

```python 
# Random forest model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train = train_data.iloc[:, 2:5]
y_train = train_data.iloc[:, 5]
X_test = test_data.iloc[:, 2:5]

# Instantiate the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities for the test dataset
y_pred = model.predict_proba(X_test)[:, 1]

# Save the predicted probabilities to a text file
np.savetxt("labels_random_forest.txt", y_pred, fmt="%.7f")
```

### Explanation

#### Original Code

I was not able to create a `labels.txt` file in my original code because it required too much memory. I realized the problem was I made a few assumptions. 

Assumed two things:

1. That it was necessary to include the timestamp
2. It was necessary to include the categorical column (the second column)

Below is my initial code: 

```python
# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Read the data
train_data = pd.read_csv("train.csv", header=None, sep=',')
test_data = pd.read_csv("test.csv", header=None, sep=',')

# Convert the timestamp to a Unix timestamp
train_data[0] = pd.to_datetime(train_data[0]).view('int64') // 10**9
test_data[0] = pd.to_datetime(test_data[0]).view('int64') // 10**9

# One-hot encode the categorical column (the city names in this case)
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore') #helps with memory
encoded_train_data = encoder.fit_transform(train_data[[1]])
encoded_test_data = encoder.transform(test_data[[1]])

# Remove the original categorical column and concatenate the encoded data
train_data = train_data.drop([1], axis=1)
train_data = pd.concat([train_data, pd.DataFrame(encoded_train_data)], axis=1)

test_data = test_data.drop([1], axis=1)
test_data = pd.concat([test_data, pd.DataFrame(encoded_test_data)], axis=1)

# Split the data into features and labels
X_train = train_data.drop([5], axis=1)
y_train = train_data.iloc[:, -2]

# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict the labels for the test data
y_test_pred = lr.predict_proba(test_data)

# Output the predicted probabilities to a file called labels.txt
with open("labels.txt", "w") as f:
    for prob in y_test_pred[:, 1]:
        f.write(f"{prob}\n")
```

Originally I did the following for both:

For (1) **the time stamp** I converted it to a Unix timestamp. This was necessary because the column has a non-metric value, which causes an error . Since the machine learning model expects numeric values. 

For (2) **the categorical column** I used one-hot encode because the values were strings. Machine learning models it expects numeric values. So I had to convert this data (categorical) to numerical values using one-hot encoding using the `fit_transform` method. I also added `sparse=True` parameter to help with memory efficiency. 

The output would look something like:

```
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [1. 0. 0. 0.]]
```

Each row represents a value from the original data. The columns of the output represent the possible values of the categorical data. A value of 1 in a column indicates that the original value belongs to that category. Meaning for the first row which has the value of 1 in the first column, which indicates that the original string was "lakewood". The second row of the output has a value of 1 in the second column, which indicates that the original string was "lake forest". And so on. 

Then I removed the original string column (second column) because it is no longer needed. Then concatenated the encoded data so it can be used in the machine learning model. I did this for both the training data and the test data. 

Then I split the data into features and labels. Since it needs to be prepared in order to use Logistic Regression model. I separate the features and target variables so it can be used as input to the machine learning model separately. This helps evaluate the performance of the machine learning model. The training data (features and labels) are used to train the model, and the test data (features and labels) are used to evaluate the performance of the model. I separated the training data into two variables. The first 5 columns will be assigned the the `x` variable while the last column will be assigned to the `y` variable. 

Next it's time to train the logistic regression model by using the training data.  First, I create a logistic regression object `lr`.  Then I train the logistic regression model using the training data (`X_train` and `y_train`). 

Then I predict the labels for the test data using the trained logistic regression model. The `predict_proba()` method gives the predicted probabilities for each class. Specifically for the positive class. This is the probability that a particular instance belongs to the positive class (what we want to predict). 

Finally, the output is to create a new file called `labels.txt` with the predicted probabilities for the positive class (second column of `y_test_pred`). This file will contain one predicted probability per line, with each line representing the probability of the corresponding instance belonging to the positive class. 

The main problem with this was because the one-hot encoding made the training data much larger. 

#### New Code

First I went back to the drawing board and tested out three main models:

1. **Logistic Regression**: Which was initial model, because the training data only had an output of 0 or 1 (binary). But I realized the output would also give a binary answer. But this is not correct, since the prompt mentioned the labels for `test.csv` can be continuous. View that output file [here](https://github.com/kellyjadams/data-science-challenge/blob/main/labels_logistic.txt).
2. **Gradient Boosting**: This can be used to capture complex relationships. But it required more fine tuning. it is also highly accurate for non-linear data. View that output file [here](https://github.com/kellyjadams/data-science-challenge/blob/main/labels_gradient.txt). 
3. **Random Forest**: This can also capture more complex relationships but it is more accurate for linear data. View that output file [here](https://github.com/kellyjadams/data-science-challenge/blob/main/c6_random_forest_model.py). 

For all of these tests the only thing I changed in the code was the model I was going to use. Everything else stayed the same. 

This is the base code:

```python
import pandas as pd
# import which every libarary was needed for the specific model 
import numpy as np

train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)

X_train = train_data.iloc[:, 2:5]
y_train = train_data.iloc[:, 5]
X_test = test_data.iloc[:, 2:5]

#Model Code
#This is where I would include the line to train the specific model 

model.fit(X_train, y_train)

# Predict probabilities for the test dataset
y_pred = model.predict_proba(X_test)[:, 1]

# Save the predicted probabilities to a text file
np.savetxt("labels_support_vector.txt", y_pred, fmt="%.7f")
```

An explanation of the above code: 

* First I imported the libraries needed
* Then I load the .csv files into the Pandas dataframe
* Next I preprocess the data by removing the timestamp and location columns, which aren't necessary to my model and is what slowed down my previous model. I created the input features (X) from columns and output labels (y) for training. For `X_train` I selected columns 2 to 4. For `y_train` I selected column 5 (the last column). For `X_test` I selected columns 2 to 4 from the `test.csv` file. 
* The `#Model Code` line is where I could include the line for using the specific model
* Then I would fit the model to the training data using the `X_train` and `y_train` .
* After I predict the probabilities for the test dataset. Which obtain probability estimates for each class. 
* Finally, I used `numpy.savetxt()` function to save the predicted probabilities to a text file. I specified the format `%.7f` to save the numbers with 7 decimal places. 

The final model ended up being a **Random Forest Model**. Because it was the quickest to run and highly accurate. The only code changed from the above is to replace the line: 

`#This is where I would include the line to train the specific model ` with 

`model = RandomForestClassifier(n_estimators=100, random_state=42)` 

Otherwise it all stays the same. 
