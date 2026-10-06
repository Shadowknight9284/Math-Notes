# Stats 596 -- Homework 2 -- Pranav Tikkawar
# Save this file as homework2.R beside hw2.tex.
# Set the R working directory to that folder before sourcing this script.
# Outputs are written to hw2_results/; all names match hw2.tex.
# Requires MASS and glmnet. No numerical results are hard-coded.

options(stringsAsFactors = FALSE, width = 120)
needed <- c("MASS", "glmnet")
missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install required packages first: install.packages(c(",
       paste(sprintf('"%s"', missing), collapse = ", "), "))")
}
resdir <- "hw2_results"
dir.create(resdir, recursive = TRUE, showWarnings = FALSE)

# LaTeX table exporter. Input headings contain intentional math markup.
write.tex.table <- function(d, filename, digits = 6) {
  text <- lapply(d, function(v) {
    if (is.numeric(v)) formatC(v, digits = digits, format = "f") else as.character(v)
  })
  text <- as.data.frame(text, check.names = FALSE)
  rows <- apply(text, 1, paste, collapse = " & ")
  bs <- intToUtf8(92)
  cmd <- function(s) paste0(bs, s)
  ending <- paste0(" ", bs, bs)
  lines <- c(cmd("begin{center}"), cmd("small"),
    cmd("setlength{\\tabcolsep}{4pt}"),
    cmd(paste0("begin{tabular}{l", paste(rep("r", ncol(d)-1), collapse=""), "}")),
    cmd("toprule"), paste0(paste(names(d), collapse=" & "), ending),
    cmd("midrule"), paste0(rows, ending), cmd("bottomrule"),
    cmd("end{tabular}"), cmd("end{center}"))
  writeLines(lines, filename, useBytes = TRUE)
}

# Exercise 1.11: exact normal-mean risks and independent integration checks.
ridge.mse <- function(mu, lambda) (1 + lambda^2*mu^2)/(1 + lambda)^2
lasso.mse <- function(mu, lambda) {
  a <- -lambda-mu
  b <- lambda-mu
  (1+lambda^2)*(pnorm(a)+pnorm(b, lower.tail=FALSE)) -
    (a+2*lambda)*dnorm(a) + (b-2*lambda)*dnorm(b) +
    mu^2*(pnorm(b)-pnorm(a))
}
checks <- expand.grid(mu=c(-3,0,2), lambda=c(.1,1,5,10))
checks$absolute.error <- mapply(function(mu, lambda) {
  a <- -lambda-mu; b <- lambda-mu
  value <- integrate(function(u) (u+lambda)^2*dnorm(u), -Inf, a)$value +
    mu^2*(pnorm(b)-pnorm(a)) +
    integrate(function(u) (u-lambda)^2*dnorm(u), b, Inf)$value
  abs(value-lasso.mse(mu,lambda))
}, checks$mu, checks$lambda)
stopifnot(max(checks$absolute.error)<1e-6)
write.csv(checks, file.path(resdir,"ex111_risk_checks.csv"), row.names=FALSE)
mu.grid <- seq(-15,15,length.out=1201)
lambdas <- c(.1,1,5,10)
colors <- c("black","blue","red","darkgreen")
pdf(file.path(resdir,"ex111_mse.pdf"), width=11, height=5)
par(mfrow=c(1,2), mar=c(4,4,3,1))
for (method in c("Ridge","Lasso")) {
  risk <- if (method=="Ridge") ridge.mse else lasso.mse
  values <- sapply(lambdas, function(lam) risk(mu.grid,lam))
  matplot(mu.grid,values,type="l",lty=1,col=colors,lwd=2,
          xlab=expression(mu),ylab="Mean squared error",main=method)
  abline(h=1,lty=2,col="gray50")
  legend("top",paste("lambda =",lambdas),col=colors,lty=1,lwd=2,bty="n")
  tab <- data.frame(mu=mu.grid,values)
  names(tab) <- c("mu",paste0("lambda_",lambdas))
  write.csv(tab,file.path(resdir,paste0("ex111_",tolower(method),"_mse.csv")),row.names=FALSE)
}
dev.off()

# Textbook simulation. Generate all data before drawing any CV folds.
make.data <- function(p) {
  set.seed(0)
  n <- 200
  mu <- rep(0,p)
  msig <- toeplitz(.8^(1:p-1))
  sig <- .5
  x <- MASS::mvrnorm(n,mu,msig)
  y <- x[,1] + pmax(0,apply(x[,2:4],1,sum))^2/3 + rnorm(n,0,sig)
  ntest <- 100
  xtest <- MASS::mvrnorm(ntest,mu,msig)
  ytest <- xtest[,1] + pmax(0,apply(xtest[,2:4],1,sum))^2/3 + rnorm(ntest,0,sig)
  list(x=x,y=y,xtest=xtest,ytest=ytest)
}

# Ridge minimizes RSS/(2*n) + lambda*sum(slopes^2)/2.
# Only centering is performed, to profile out an unpenalized intercept.
ridge.path <- function(x,y,lambda) {
  n <- nrow(x)
  xb <- colMeans(x); yb <- mean(y)
  xc <- sweep(x,2,xb,"-"); yc <- y-yb
  s <- svd(xc)
  keep <- s$d > max(s$d)*1e-10
  d <- s$d[keep]
  if (!length(d)) stop("Centered design has zero rank.")
  U <- s$u[,keep,drop=FALSE]; V <- s$v[,keep,drop=FALSE]
  uy <- drop(crossprod(U,yc))
  W <- vapply(lambda,function(lam) d/(d^2+n*lam),numeric(length(d)))
  W <- matrix(W,nrow=length(d),ncol=length(lambda))
  B <- V %*% sweep(W,1,uy,"*")
  shrink <- vapply(lambda,function(lam) d^2/(d^2+n*lam),numeric(length(d)))
  shrink <- matrix(shrink,nrow=length(d),ncol=length(lambda))
  list(beta=B,intercept=yb-drop(xb %*% B),df=1+colSums(shrink))
}

# Gaussian Lasso on original scales with an unpenalized intercept.
# alpha=1 preserves the stated Lasso objective when glmnet undoes its
# internal response transformation. No manual x or y rescaling is used.
lasso.path <- function(x, y, lambda) {
  fit <- glmnet::glmnet(
    x = x,
    y = y,
    family = "gaussian",
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    intercept = TRUE,
    thresh = 1e-10,
    maxit = 3000000
  )

  if (length(fit$lambda) != length(lambda)) {
    stop(
      "glmnet returned ", length(fit$lambda),
      " of ", length(lambda), " requested penalties. ",
      "The path did not finish; do not use incomplete results."
    )
  }

  if (max(abs(fit$lambda - lambda)) >
      1e-10 * max(1, max(lambda))) {
    stop("Returned penalties do not match the requested grid.")
  }

  list(
    beta = as.matrix(fit$beta),
    intercept = as.numeric(fit$a0),
    df = NULL
  )
}
fit.path <- function(x,y,lambda,method) {
  if (method=="ridge") ridge.path(x,y,lambda) else lasso.path(x,y,lambda)
}
predict.path <- function(fit,x) sweep(x %*% fit$beta,2,fit$intercept,"+")
path.mse <- function(y,pred) colMeans(sweep(pred,1,y,"-")^2)

# Equal-size fold errors, fold-based SE, and the largest acceptable lambda.
cv.path <- function(x,y,lambda,method,folds) {
  K <- max(folds)
  error <- matrix(NA_real_,K,length(lambda))
  for (k in seq_len(K)) {
    validation <- which(folds==k)
    training <- which(folds!=k)
    fit <- fit.path(x[training,,drop=FALSE],y[training],lambda,method)
    error[k,] <- path.mse(y[validation],predict.path(fit,x[validation,,drop=FALSE]))
  }
  average <- colMeans(error)
  se <- apply(error,2,sd)/sqrt(K)
  imin <- which.min(average)
  eligible <- which(average<=average[imin]+se[imin])
  i1se <- eligible[which.max(lambda[eligible])]
  list(mean=average,se=se,imin=imin,i1se=i1se,fold.errors=error)
}

# Original-scale Lasso KKT checks, including the intercept equation.
lasso.kkt <- function(x,y,fit,lambda) {
  residual <- sweep(predict.path(fit,x),1,y,"-")
  gradient <- crossprod(x,residual)/nrow(x)
  vapply(seq_along(lambda),function(k) {
    b <- fit$beta[,k]
    active <- abs(b)>1e-8
    violations <- numeric(length(b))
    violations[active] <- abs(gradient[active,k]+lambda[k]*sign(b[active]))
    violations[!active] <- pmax(abs(gradient[!active,k])-lambda[k],0)
    max(c(violations,abs(mean(residual[,k]))))
  },numeric(1))
}

run.method <- function(dat,method,folds5,folds10,p) {
  x <- dat$x; y <- dat$y
  xc <- sweep(x,2,colMeans(x),"-"); yc <- y-mean(y)
  if (method=="ridge") {
    scale <- max(svd(xc,nu=0,nv=0)$d)^2/nrow(x)
    lambda <- scale*10^seq(4,-8,length.out=240)
  } else {
    lambda.max <- max(abs(drop(crossprod(xc,yc))))/nrow(x)
    lambda <- lambda.max * 10^seq(
  from = 0,
  to = -4,
  length.out = 240
)
  }
  fit <- fit.path(x,y,lambda,method)
  train <- path.mse(y,predict.path(fit,x))
  test <- path.mse(dat$ytest,predict.path(fit,dat$xtest))
  cv5 <- cv.path(x,y,lambda,method,folds5)
  cv10 <- cv.path(x,y,lambda,method,folds10)
  stem <- paste0("p",p,"_",method)
  if (method=="ridge") {
    axis <- fit$df
    xlabel <- "Degrees of freedom (including intercept)"
  } else {
    reference <- ridge.path(x,y,0)
    axis <- colSums(abs(fit$beta))/sum(abs(reference$beta[,1]))
    xlabel <- "L1 shrinkage fraction (minimum-norm LS reference)"
    checks <- lasso.kkt(x,y,fit,lambda)
    write.csv(data.frame(lambda=lambda,kkt.violation=checks),
              file.path(resdir,paste0(stem,"_kkt.csv")),row.names=FALSE)
    if (max(checks)>1e-4) {
      stop(stem,": KKT violation exceeds 1e-4; inspect convergence.")
    }
  }
  best <- which.min(test)
  selected <- c(best,cv5$imin,cv5$i1se,cv10$imin,cv10$i1se)
  selection <- c("Test-grid oracle","5-fold CV minimum","5-fold CV 1-SE",
                 "10-fold CV minimum","10-fold CV 1-SE")
  tab <- data.frame(Selection=selection,lambda=lambda[selected],
                    `Test MSE`=test[selected],check.names=FALSE)
  write.csv(tab,file.path(resdir,paste0(stem,"_selection.csv")),row.names=FALSE)
  latex.tab <- tab
  names(latex.tab)[2] <- "$\\lambda$"
  write.tex.table(latex.tab,file.path(resdir,paste0(stem,"_selection.tex")))
  detail <- data.frame(lambda=lambda,axis=axis,train=train,test=test,
                       cv5=cv5$mean,se5=cv5$se,cv10=cv10$mean,se10=cv10$se)
  write.csv(detail,file.path(resdir,paste0(stem,"_path.csv")),row.names=FALSE)
  coef <- rbind(intercept=fit$intercept,fit$beta)
  rownames(coef) <- c("Intercept",paste0("x",seq_len(ncol(x))))
  write.csv(coef,file.path(resdir,paste0(stem,"_coefficients.csv")))
  for (K in c(5,10)) {
    cv <- if(K==5) cv5 else cv10
    write.csv(cv$fold.errors,file.path(resdir,paste0(stem,"_cv",K,"_fold_errors.csv")),row.names=FALSE)
  }
  at.endpoint <- c(best,cv5$imin,cv10$imin) %in% c(1,length(lambda))
  if(any(at.endpoint)) {
    warning(stem,": a test/CV minimum is at a grid endpoint. Inspect and extend the grid if needed.")
  }
  list(method=method,lambda=lambda,fit=fit,axis=axis,xlabel=xlabel,
       train=train,test=test,cv5=cv5,cv10=cv10,table=tab,best=best,
       endpoint=at.endpoint)
}

plot.result <- function(z,p) {
  stem <- paste0("p",p,"_",z$method)
  # Keep penalty-path order; the L1 reference fraction need not be monotone.
  pdf(file.path(resdir,paste0(stem,"_path.pdf")),width=8,height=5.2)
  par(mar=c(4.5,4,3,1))
  matplot(z$axis,t(z$fit$beta),type="l",lty=1,
          xlab=z$xlabel,ylab="Slope coefficient",
          main=paste(z$method,"coefficient path: p =",p))
  abline(h=0,col="gray80")
  dev.off()

  pdf(file.path(resdir,paste0(stem,"_errors.pdf")),width=11,height=5)
  par(mfrow=c(1,2),mar=c(4.5,4,3,1))
  E <- cbind(z$train,z$test,z$cv5$mean,z$cv10$mean)
  for (zoom in c(FALSE,TRUE)) {
    limits <- range(E)
    if (zoom) {
      minimum <- min(c(z$test,z$cv5$mean,z$cv10$mean))
      limits <- c(0,max(1,2.5*minimum))
    }
    matplot(z$axis,E,type="l",lty=c(1,1,2,3),lwd=2,
            col=c("black","red","blue","darkgreen"),ylim=limits,
            xlab=z$xlabel,ylab="Mean squared error",
            main=if(zoom) "Prediction-error detail" else "Full error range")
    abline(v=z$axis[c(z$cv5$imin,z$cv5$i1se)],col="blue",lty=c(2,3))
    abline(v=z$axis[c(z$cv10$imin,z$cv10$i1se)],col="darkgreen",lty=c(2,3))
    legend("topleft",c("Training","Test","5-fold CV","10-fold CV"),
           col=c("black","red","blue","darkgreen"),lty=c(1,1,2,3),
           lwd=2,bty="n",cex=.8)
  }
  dev.off()

  pdf(file.path(resdir,paste0(stem,"_cv.pdf")),width=11,height=5)
  par(mfrow=c(1,2),mar=c(4.5,4,3,1))
  for (K in c(5,10)) {
    cv <- if(K==5) z$cv5 else z$cv10
    plot(z$axis,cv$mean,type="l",xlab=z$xlabel,ylab="CV MSE",
         main=paste(K,"fold CV with one-SE bands"),
         ylim=range(c(cv$mean-cv$se,cv$mean+cv$se)))
    lines(z$axis,cv$mean+cv$se,col="gray50",lty=2)
    lines(z$axis,cv$mean-cv$se,col="gray50",lty=2)
    abline(v=z$axis[c(cv$imin,cv$i1se)],col=c("blue","red"),lty=c(2,3))
    legend("topleft",c("CV mean","CV mean +/- SE","CV minimum","1-SE choice"),
           col=c("black","gray50","blue","red"),lty=c(1,2,2,3),bty="n",cex=.8)
  }
  dev.off()
}

results <- list()
for (p in c(20,100,200)) {
  cat("\n========== p =",p,"==========\n")
  dat <- make.data(p)
  for (kind in c("train","test")) {
    raw <- if(kind=="train") cbind(dat$y,dat$x) else cbind(dat$ytest,dat$xtest)
    d <- data.frame(Row=as.character(1:3),raw[1:3,1:6],check.names=FALSE)
    names(d) <- c("Row","$y$",paste0("$x_",1:5,"$"))
    cat("\n",kind,"data: first rows\n")
    print(d)
    write.tex.table(d,file.path(resdir,paste0("p",p,"_",kind,"_rows.tex")))
    write.csv(d,file.path(resdir,paste0("p",p,"_",kind,"_rows.csv")),row.names=FALSE)
  }
  set.seed(123)
  permutation <- sample.int(nrow(dat$x))
  folds5 <- folds10 <- integer(nrow(dat$x))
  folds5[permutation] <- rep(1:5,length.out=nrow(dat$x))
  folds10[permutation] <- rep(1:10,length.out=nrow(dat$x))
  write.csv(data.frame(row=seq_len(nrow(dat$x)),fold5=folds5,fold10=folds10),
            file.path(resdir,paste0("p",p,"_folds.csv")),row.names=FALSE)
  results[[as.character(p)]] <- list()
  for (method in c("ridge","lasso")) {
    cat("\nFitting",method,"and both CV paths...\n")
    z <- run.method(dat,method,folds5,folds10,p)
    plot.result(z,p)
    print(z$table,row.names=FALSE)
    results[[as.character(p)]][[method]] <- z
  }
}

# Numerical summaries are generated from the actual computed tables.
fmt <- function(x) sprintf("%.4f",x)
method.sentence <- function(z) {
  t <- z$table
  paste0(if(z$method=="ridge") "Ridge" else "Lasso",
    " achieved a minimum test-grid MSE of ",fmt(t[[3]][1]),
    "; the 5-fold minimum-CV and one-SE choices gave test MSEs ",
    fmt(t[[3]][2])," and ",fmt(t[[3]][3]),
    ", while the corresponding 10-fold choices gave ",
    fmt(t[[3]][4])," and ",fmt(t[[3]][5]),". ")
}
r20 <- results[["20"]]
summary17 <- paste0(method.sentence(r20$ridge),method.sentence(r20$lasso),
  "The test-grid oracle is an infeasible benchmark, whereas the CV choices use only training data. ",
  "The one-SE rule selects a penalty at least as large as the minimum-CV choice, favoring ",
  "a more regularized fit; its prediction cost or benefit is shown by the reported test errors. ",
  "Ridge continuously shrinks coefficients, while Lasso permits exact zeros. Both are linear ",
  "approximations to a nonlinear conditional mean, so regularization reduces estimation ",
  "variability without removing the underlying mean-model misspecification.")
writeLines(summary17,file.path(resdir,"ex117_summary.tex"))
summary18 <- paste0("For $p=100$, ",method.sentence(results[["100"]]$ridge),
  method.sentence(results[["100"]]$lasso),"For $p=200$, ",
  method.sentence(results[["200"]]$ridge),method.sentence(results[["200"]]$lasso),
  "Increasing the dimension adds correlated predictors without changing the four variables ",
  "in the generating mean. The comparisons reflect both increased estimation complexity and ",
  "the realized samples. At $p=200$, the centered design has rank at most 199, making the ",
  "unpenalized slope solution nonunique, while positive ridge penalties remain well defined. ",
  "The Lasso fraction uses the minimum-norm least-squares reference. Changing the dimension ",
  "also changes the random-number sequence, so these are not paired comparisons on identical observations.")
writeLines(summary18,file.path(resdir,"ex118_summary.tex"))
combined <- do.call(rbind,lapply(names(results),function(p) {
  do.call(rbind,lapply(c("ridge","lasso"),function(method) {
    data.frame(p=as.integer(p),Method=method,results[[p]][[method]]$table,check.names=FALSE)
  }))
}))
write.csv(combined,file.path(resdir,"all_selection_results.csv"),row.names=FALSE)
saveRDS(results,file.path(resdir,"all_results.rds"))
capture.output(sessionInfo(),file=file.path(resdir,"sessionInfo.txt"))

# Verify every file imported by the supplied LaTeX document exists.
expected <- c("ex111_mse.pdf","ex117_summary.tex","ex118_summary.tex")
for (p in c(20,100,200)) {
  expected <- c(expected,paste0("p",p,"_",c("train","test"),"_rows.tex"))
  for (method in c("ridge","lasso")) {
    stem <- paste0("p",p,"_",method)
    expected <- c(expected,paste0(stem,c("_path.pdf","_errors.pdf","_cv.pdf","_selection.tex")))
  }
}
stopifnot(all(file.exists(file.path(resdir,expected))))
cat("\nFinished. All",length(expected),"LaTeX input files were generated.\n")
cat("Inspect any grid-endpoint warnings, then compile hw2.tex.\n")
