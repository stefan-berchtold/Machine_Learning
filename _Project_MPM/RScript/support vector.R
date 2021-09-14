df <- readRDS("C:/home/stefan/__Machine Learning/_Project_MPM/data/Pollution_Bejing", refhook = FALSE)
str(df)
View(df)


library(tidyverse) 
library(e1071)     
library(ROCR)
library(ggplot2)
library(data.table)

data_class <- copy(df)

data_class$classification <- rep(0, times = length(df$pm2.5))
data_class$classification[exp(df$pm2.5) < 51] <- 1
data_class$classification[exp(df$pm2.5) >= 51 & exp(df$pm2.5) < 101] <- 2
data_class$classification[exp(df$pm2.5) >= 101 & exp(df$pm2.5) < 151] <- 3
data_class$classification[exp(df$pm2.5) >= 151 & exp(df$pm2.5) < 201] <- 4
data_class$classification[exp(df$pm2.5) >= 300] <- 6

## Since we could not hack into the ETH Supercomputer -Euler and Leonard-, we had to find a different approach to let our 
## Support Vector Machine algorithm run without waiting untill Christmas for results. That is why we chose to draw a sample of
## 10% of our originial data and test is with another 10% sample

sample <- data_class%>% sample_frac(0.1)
View(sample)

## To find the best model with respect to cost, we define a wide cost range and let the tune function find the best performing 
## model with a linear kernel 
cost_range <-
  c(1e-10,
    1e-7,
    1e-5,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.1,
    1,
    5,
    10,
    100)
    
 

tune.out <- tune(
  svm,
  as.factor(classification) ~ .,
  data = sample,
  kernel = "linear",
  ranges = list(cost = cost_range)
)

summary(tune.out
        )

bestsvm <- tune.out$best.model
summary(bestsvm)

sample_test <- data_class%>% sample_frac(0.1)

predicition_svm <- table(predict = predict(bestsvm, sample_test),
                        truth = sample$classification)

print(predicition_svm)


      

      
      
      
corrects=sum(diag(predicition_svm))
errors=sum(predicition_svm)-corrects
sum(predicition_svm)
performance_test=corrects/(corrects+errors)
print(performance_test)


## We repeat the same steps above with a radial kernel to see if our prediction accuracy improves
tune.out.radial <- tune(
  svm,
  as.factor(classification) ~ .,
  data = sample,
  kernel = "radial",
  ranges = list(cost = cost_range)
)

summary(tune.out.radial)

bestsvm.radial <- tune.out.radial$best.model
summary(bestsvm.radial)

predicition_svm.radial <- table(predict = predict(bestsvm.radial, sample_test),
                        truth = sample$classification)

print(predicition_svm.radial)

corrects1=sum(diag(predicition_svm.radial))
errors1=sum(predicition_svm.radial)-corrects
performance_test1=corrects/(corrects+errors)
print(performance_test1)



## The same for a polynomial kernel



tune.out.pol <- tune(
  svm,
  as.factor(classification) ~ .,
  data = sample,
  kernel = "polynomial",
  ranges = list(cost = cost_range)
)

summary(tune.out.pol)

bestsvm.pol <- tune.out.pol$best.model
summary(bestsvm.pol)

predicition_svm.pol <- table(predict = predict(bestsvm.pol, sample_test),
                                truth = sample$classification)

print(predicition_svm.pol)

corrects2=sum(diag(predicition_svm.pol))
errors2=sum(predicition_svm.pol)-corrects
performance_test2=corrects/(corrects+errors)
print(performance_test)


## And a simoid kernel


tune.out.sig <- tune(
  svm,
  as.factor(classification) ~ .,
  data = sample,
  kernel = "sigmoid",
  ranges = list(cost = cost_range)
)

summary(tune.out.sig)

bestsvm.sig <- tune.out.sig$best.model
summary(bestsvm.sig)

predicition_svm.sig <- table(predict = predict(bestsvm.sig, sample_test),
                             truth = sample$classification)

print(predicition_svm.sig)

corrects3=sum(diag(predicition_svm.sig))
errors3=sum(predicition_svm.sig)-corrects
performance_test3=corrects/(corrects+errors)
print(performance_test)


## Conclusion all models are shit

