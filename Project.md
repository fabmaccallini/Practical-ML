# Practical Machine Learning - Course Project
Fabrizio Maccallini  
Tuesday, November 18, 2014  
### Summary
The goal of the project is to predict the type of exercise (*"classe"* variable in the training set) people carrying the device, such as Jawbone Up, Nike FuelBand, Fitbit, performed. The project is divided in three parts:  
1. Loading, cleaning and processing the data.  
2. Partitioning, training and cross-validating the train set.    
3. Predicting the test set.  

### 1. Data processing

```r
# Load the data
train <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c("#DIV/0!", "NA", ""))
test <- read.csv("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings = c("#DIV/0!", "NA", ""))
```

After loading the two data sets we observed very large numbers of NAs concentrated in specific variables of the *train* data set.  

```r
# Clean the train set
na <- apply(train, 2, function(x) sum(is.na(x)))
summary(na)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##       0       0   19220   12030   19220   19620
```

```r
# Closer look between Q1 and Median (Q2)
quantile(na, probs = seq(0.25, 0.5, 0.05))
```

```
##   25%   30%   35%   40%   45%   50% 
##     0     0     0 19216 19216 19216
```

```r
# Clean the NA variables
train.clean <- train[, na == 0]
# Clean the descriptive variables
train.clean <- train.clean[, -c(1: 7)]
# Clean the test set
test.clean <- test[, names(train.clean[, -53])] # "classe" not present in test
```

We decided to remove from the train set all the 107 variables having 97.93% or more of the observations equal to NAs, and to retain only the 53 ones without NAs.  
The first seven variables were also removed as they do not carry any information relevant for our purpose.
Then we went through the same cleaning process for the test set, to be used for our predictions.  

### 2. Training the model  

```r
# partition train.clean
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y = train.clean$classe, p = 3/4)[[1]]
training = train.clean[inTrain,]
testing = train.clean[-inTrain,]
```

The cleaned train set was paritioned into a training subset (75%) and a validation subset (25%). The model used for classifing the  type of exercise (variable *"classes"*) is the *random forest* (rf), being one of the most accurate learning algorithms. The number of folds used for cross validation is 10, generally accepted as a good compromise between bias and variance. In order to reduce the calculation time we are going to enable parallel computing on 4 cores.  

```r
# use multi-core support
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoParallel(core = 4)
# train the model with cross-validation
model <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(method = "cv", number = 10))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# validate the model
pred <-  predict(model, newdata = testing)
result <- confusionMatrix(pred, testing$classe)
print(result)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    4    0    0    0
##          B    1  943    6    0    0
##          C    0    1  845   10    1
##          D    0    1    4  793    3
##          E    0    0    0    1  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9908, 0.9955)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9917          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9937   0.9883   0.9863   0.9956
## Specificity            0.9989   0.9982   0.9970   0.9980   0.9998
## Pos Pred Value         0.9971   0.9926   0.9860   0.9900   0.9989
## Neg Pred Value         0.9997   0.9985   0.9975   0.9973   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1923   0.1723   0.1617   0.1829
## Detection Prevalence   0.2851   0.1937   0.1748   0.1633   0.1831
## Balanced Accuracy      0.9991   0.9960   0.9927   0.9922   0.9977
```

The model was expecting an accuracy rate of 99.29%, with an estimated out of sample error of 0.71%. Predicting the testing subset the model achieved an accuracy rate of 99.35%, the out of sample error is 0.65% with 0.45, 0.92% confidence interval. The estimate of the out of sample error was fairly in line with the actual value from the prediction.

### 3. Model prediction

```r
# predict the test set
test.pred <-  predict(model, newdata = test.clean)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Running the classification model on the test set, produced the following predictions: B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B.  

```r
# function provided by the instructor
pml_write_files = function(x){
  n = length(x)
  for(i in 1: n){
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
pml_write_files(test.pred)
closeAllConnections()
```
