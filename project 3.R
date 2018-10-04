"Data Set Input and intial analysis
The dataset we started working on was obtained from the UCI repository. It initially had 1567 
observations (rows) which were outputs of 590 sensor measurements (columns/variables) and a 
label of Yield Pass/Fail."

#Loading data to workspace

library(data.table)
feature=fread("C:/Users/Randhir/Desktop/ISEN613 project/SECOM UCI/secom.data.txt", data.table = F)
label = fread("C:/Users/Randhir/Desktop/ISEN613 project/SECOM UCI/secom_labels.data.txt", data.table = F)
data = cbind(label,feature)
colnames(data) = c("Class", "Time", paste0(rep("Feature", ncol(feature)), seq(1,ncol(feature))))
data$Class = factor(data$Class, labels = c("pass", "fail"))
data$Time =  as.POSIXct(data$Time, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
dim(data)

"On close observation we find that there are many missing values and equal values which need to 
be tackled before any further operation on the Dataset."

sum(is.na(data))

"Here there are 41951 missing values in the table.

Step I: Removing Redundant data and Missing Values"

#data cleaning

# Here the Time variable is redundent for our study so we need to remove it.
index_vr1 = which(colnames(data) == "Time")
# Remove coulmns with equal value as those features wont be useful for us.
equal_v = apply(data, 2, function(x) max(na.omit(x)) == min(na.omit(x)))
index_vr2 = which(equal_v == T)
#we remove the features that have more than 40% of values missing
row_NA = apply(data, 1, function(x) sum(is.na(x))/ncol(data))
col_NA = apply(data, 2, function(x) sum(is.na(x))/nrow(data))
index_mr = which(col_NA > 0.4)

"index_vr1, index_vr2, index_mr might have common coulmn numbers.unique function is used to
remove the repeating column numbers"
data_cleaned = data[,-unique(c(index_vr1, index_vr2, index_mr))]

#We have reduced the number of features to 443 after above steps.
dim(data_cleaned)
#Initially the missing values in the data set was 41951 which we have reduced to 8008 now. 
sum(is.na(data_cleaned))

"Now we cannot delete all the rows or columns with missing values. That would lead to loss of 
data and our prediction would be biased. So our next step is to fill the missing values.

Step II: Imputation using different techniques"

#Imputation

#Knn Imputation
library(DMwR)
data_Imputed  = knnImputation(data_cleaned)
sum(is.na(data_Imputed))   #no missing values
dim(data_Imputed)
fix(data_Imputed)

#Splitting into training & testing data
set.seed(2)
index = sample(1:nrow(data_Imputed), nrow(data_Imputed)/10)
train = data_Imputed[-index,]
test = data_Imputed[index,]

#Case Boosting
library('ROSE')
#oversampling
over_sampled_data = ovun.sample(Class~ ., data =train, method = "over")$data
table(over_sampled_data$Class)
#ROSE
train_rose = ROSE(Class ~ ., data = train, seed = 1)$data
table(train_rose$Class)


#Lasso
library(glmnet)
fit_LS = glmnet(as.matrix(train_rose[,-1]), train_rose[,1], family="binomial", alpha=1)
plot_glmnet(fit_LS, "lambda", label=5)
library(plotmo)
fit_LS_cv = cv.glmnet(as.matrix(train_rose[,-1]), as.matrix(as.numeric(train_rose[,1])-1), type.measure="class", family="binomial", alpha=1)
plot(fit_LS_cv)
coef = coef(fit_LS_cv, s = "lambda.min")
coef_df = as.data.frame(as.matrix(coef))
index_LS = rownames(coef_df)[which(coef_df[,1] != 0)][-1] 

#SVM
library(e1071)
svm.fit=svm(Class~., data=train_rose[,c("Class",index_LS)], kernel="polynomial", cost=10,scale=FALSE)
svm.pred=predict(svm.fit,test)
summary(svm.fit)
table(svm.pred,test$Class)
mean(svm.pred==test$Class)







