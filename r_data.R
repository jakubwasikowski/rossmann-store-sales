library(readr)
library(xgboost)


cat("reading the train and test data\n")
train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")
store <- read_csv("data/store.csv")

train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integeverr columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
# seperating out the elements of the date column for the train set
train$Month <- as.integer(format(as.Date(train$Date), "%m"))
train$Year <- as.integer(format(as.Date(train$Date), "%y"))
train$Day <- as.integer(format(as.Date(train$Date), "%d"))

# seperating out the elements of the date column for the test set
test$Month <- as.integer(format(as.Date(test$Date), "%m"))
test$Year <- as.integer(format(as.Date(test$Date), "%y"))
test$Day <- as.integer(format(as.Date(test$Date), "%d"))

feature.names <- names(train)[c(1,2,6:21)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  print(f)
  print(class(train[[f]]))
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    print(levels)
    train[[paste(f, "numeric", sep="_")]] <- as.integer(factor(train[[f]], levels=levels))
    test[[paste(f, "numeric", sep="_")]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)

write.csv(train, "data/R_train_transformed.csv", row.names=FALSE)
write.csv(test, "data/R_test_transformed.csv", row.names=FALSE)

