############################################
# Data Cleaning
# Datum: 22.Juni 2020
############################################

setwd("C:\home\stefan\__Machine Learning\_Project_MPM/data")
getwd()


### Daten laden
df <- read.csv("C:/home/stefan/__Machine Learning/_Project_MPM/data/PRSA_data.csv", sep = ",", header = TRUE)


### Daten Übersicht
head(df)

str(df)

View(df)

############################################
# Prüfen von NA
############################################
sum(is.na(df) == TRUE)

df <- na.omit(df)

### Preparation

# Change numeric month to a to a abbreviated month name of the month
df$month <- month.abb[df$month]

# change year and month as factor 
df$year <-  as.factor(df$year)
df$month <-  as.factor(df$month)

str(df)

############################################
# Plotting
############################################
plot(df$pm2.5)
hist(df$pm2.5)

# Transform the response variable because it is right-skwed
df$pm2.5 <-  log(df$pm2.5)

plot(df$pm2.5)
hist(df$pm2.5)

# remove INF 
df <-  df[is.finite(df$pm2.5), ]

############################################
# Speichern des Datensatzes 
############################################
saveRDS(df, "C:/home/stefan/__Machine Learning/_Project_MPM/data/Pollution_Bejing")


############################################
# Seession Information
############################################
sessionInfo()

