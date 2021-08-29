## Recommendation using matrix factorization with ALS
## Data from "Amazon fine food reviews"

pacman::p_load(dplyr, readxl, recosystem)

rating = read_xlsx("Reviews_2012.xlsx")

# Check duplications: numbers of customers who have rated the same products
nrow(rating) - nrow(distinct(rating[c("UserId", "ProductId", "Score")], ProductId, UserId))

# Dedup with mean function
rating <-
    aggregate(data = rating, Score ~ UserId + ProductId, FUN = mean)

# Sort rating by UserId
rating <- rating[with(rating, order(UserId)), ]

user_id <- unique(rating$UserId)
product_id <- unique(rating$ProductId)

# Construct user table
user_table <- as.data.frame(user_id)
user_table$ID <- seq.int(nrow(user_table))

# Construct product table
product_table <- as.data.frame(product_id)
product_table$ID <- seq.int(nrow(product_table))

# Replace product string and user string with product index and user index
tmp <- rating
tmp[]  <- user_table$ID[match(unlist(rating), user_table$user_id)]
rating$UserId <- tmp$UserId

tmp <- rating
tmp[]  <-
    product_table$ID[match(unlist(rating), product_table$product_id)]
rating$ProductId <- tmp$ProductId

rm(tmp)
rm(user_id)
rm(product_id)
head(rating)

# Partition into train and test data set
train_indexes <-
    sample(1:nrow(rating), size = floor(0.9 * nrow(rating)))
train <- rating[train_indexes, ]
test <- rating[-train_indexes, ]

# Load into recosystem format
trainset = data_memory(train$UserId, train$ProductId, train$Score, index1 = TRUE)
testset = data_memory(test$UserId, test$ProductId, test$Score, index1 = TRUE)

r = Reco()

# Factorisation using the optimised parameters
# Optimised factorisation using r$tune
opts = r$tune(
    trainset,
    opts = list(
        dim = c(10, 20),
        lrate = c(0.1, 0.2),
        costp_l1 = c(0, 0.1),
        costp_l2 = c(0.01, 0.1),
        costq_l1 = c(0, 0.1),
        costq_l2 = c(0.01, 0.1),
        nthread = 8, # change if needed
        niter = 40
    )
)
opts$min

r$train(trainset, opts = opts$min)
show(r)

# Exports the two matrix to the current directory (as mat_P.txt, mat_Q.txt)
r$output() 
P = as.matrix(read.table("mat_P.txt")) # the user factors matrix
Q = as.matrix(read.table("mat_Q.txt")) # the item factors matrix
rownames(P) = paste0("u", 1:nrow(P))
rownames(Q) = paste0("i", 1:nrow(Q))

# Get predictions:
# this multiplies the user vectors in testset, with the item vectors in Q
test$prediction <- r$predict(testset, out_memory())
head(test)

# Compute prediction MAE
test$MAE = abs(test$Score - test$prediction)
mean(test$MAE, na.rm = TRUE) # show the MAE

# Helper functions
avgMAE = function(preds) {
    plist = unlist(preds)
    errors = sapply(1:(length(plist)/2),function(i) abs(plist[i*2-1]-plist[i*2]))
    errors = errors[errors != Inf]
    mean(errors,na.rm=TRUE)
}

showCM = function(preds, like) {
    plist = unlist(preds)
    cnts = sapply(1:(length(plist)/2), function(i) {
        pred = plist[i*2-1] ; actual = plist[i*2]
        if (!is.na(pred) & !is.nan(actual)) {
            if (pred>=like) {if(actual>=like) c(1,0,0,0) else c(0,1,0,0)}
            else if(actual<like) c(0,0,1,0) else c(0,0,0,1) 
        } else c(0,0,0,0)
    })
    s = rowSums(cnts)   #returns cnts for: TP, FP, TN, FN
    
    cat(sprintf("TN=%5d FP=%5d\n",s[3],s[2]))
    cat(sprintf("FN=%5d TP=%5d  (total=%d)\n",s[4],s[1], sum(s)))
    cat(sprintf("accuracy  = %0.1f%%\n",(s[1]+s[3])*100/sum(s)))
    cat(sprintf("precision = %3.1f%%\n",s[1]*100/(s[1]+s[2])))
    cat(sprintf("recall    = %3.1f%%\n",s[1]*100/(s[1]+s[4])))
}

# Compute prediction MAE with threshold
preds = t(test[, c("prediction", "Score")])
preds = unlist(preds)
cat("avg MAE =", avgMAE(preds))
showCM(preds, like = 3)


# Compute user vector from entire dataset
rating_set = data_memory(rating$UserId, rating$ProductId, rating$Score, index1 = TRUE)
r$train(rating_set, opts = opts$min)
r$output() 
P = as.matrix(read.table("mat_P.txt")) # the user factors matrix

target = 6432# can select any user
T = as.matrix(P[target,]); T# get the latent features for the target
prats = Q %*% T# multiply T by the item features matrix Q
prats = prats[order(prats,decreasing = TRUE),]# contains predicted ratings for all items
head(prats) 
