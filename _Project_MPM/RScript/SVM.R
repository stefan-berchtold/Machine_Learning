
df <- readRDS("C:/home/stefan/__Machine Learning/_Project_MPM/data/Pollution_Bejing", refhook = FALSE)
str(df)
View(df)


library(tidyverse) 
library(e1071)     
library(ROCR)
library(ggplot2)

df$classification <- rep(0, times = length(df$pm2.5))
df$classification[exp(df$pm2.5) < 51] <- 1
df$classification[exp(df$pm2.5) >= 51 & exp(df$pm2.5) < 101] <- 2
df$classification[exp(df$pm2.5) >= 101 & exp(df$pm2.5) < 151] <- 3
df$classification[exp(df$pm2.5) >= 151 & exp(df$pm2.5) < 201] <- 4
df$classification[exp(df$pm2.5) >= 201 & exp(df$pm2.5) < 300] <- 5
df$classification[exp(df$pm2.5) >= 300] <- 6



svmfit <- svm(
    classification ~ TEMP,
    data = df,
    kernel = "linear",
    cost = 10,
    scale = FALSE
  )


