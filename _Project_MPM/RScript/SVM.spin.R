
df <- readRDS("C:/home/stefan/__Machine Learning/_Project_MPM/data/Pollution_Bejing", refhook = FALSE)
str(df)
View(df)


library(tidyverse) 
library(e1071)     
library(ROCR)
library(ggplot2)



svmfit <- svm(
    pm2.5 ~ .,
    data = df,
    kernel = "linear",
    cost = 10,
    scale = FALSE
  )

options(scipen = 99) # penalty for displaying scientific notation
options(digits = 4) 



df_class <-
    data.frame(x = df, t = as.factor(
        as.numeric(exp(df$pm2.5))< 4
    ))


View(df_class)