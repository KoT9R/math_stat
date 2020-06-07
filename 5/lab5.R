library(mixtools)  
library(MASS)
library(sfsmisc)
library(sp)
#library(NBPSeq)
GetEllipsePoints <- function(m.x, m.y, sigma, q = 0.75, n = 200)
{
  k <- qchisq(q, 2) 
  sigma <- k * sigma 
  e <- eigen(sigma) 
  angles <- seq(0, 2*pi, length.out=n) 
  cir1.points <- rbind(cos(angles), sin(angles))
  ellipse.centered <- (e$vectors %*% diag(sqrt(abs(e$values)))) %*% cir1.points 
  ellipse.biased <- ellipse.centered + c(m.x, m.y) 
  return(ellipse.biased)
}

ellipse_bvn <- function(bvn, N, rho) 
{
  plot(bvn, pch = 1,lwd = 1, xlim=c(-3,3), ylim=c(-3,3),
       main = paste("Двумерное нормальное распределение, ","N = ", N, ", ρ = ", rho))

  m.x <- mean(bvn[, 1])
  m.y <- mean(bvn[, 2])
  p <- GetEllipsePoints(m.x, m.y, sigma, 0.9)
  points(p[1, ], p[2, ], type="l", col="red")
  
}

quad <- function(bvn, N) {
  v <- replicate(4, 0)
  for (i in 1:N) {
    x <- bvn[i,1]
    y <- bvn[i,2]
    if (x > 0  && y > 0) {
      v[1] <- v[1] + 1
    } 
    else if (x < 0  && y > 0) {
      v[2] <- v[2] + 1
    }
    else if (x < 0  && y < 0) {
      v[3] <- v[3] + 1
    }
    else if (x > 0  && y < 0) {
      v[4] <- v[4] + 1
    }
  }
  for (i in 1:4) {
    v[i] <- v[i] / N
  }
  res <- v[1]+v[3]-v[2]-v[4]
  return(res)
}

research_correlation <- function(N, rho)
{
  
  mu1 <- 0; s1 <- 1
  mu2 <- 0; s2 <- 1
  mu <- c(mu1,mu2)
  sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2), 2)
  
  x <- 1:4
  rho_pearson <- 0
  rho_spearman <- 0
  rho_quad <- 0
  for (i in 1:1000) {
    bvn <- mvrnorm(N, mu = mu, Sigma = sigma)
    rho_pearson[i] <- cor(bvn[, 1], bvn[, 2], method = "pearson")
    rho_spearman[i] <- cor(bvn[, 1], bvn[, 2], method = "spearman")
    rho_quad[i] <- quad(bvn, N)
  }
  print(c(mean(rho_pearson), mean(rho_spearman), mean(rho_quad)))
  print(c(mean(rho_pearson^2), mean(rho_spearman^2), mean(rho_quad^2)))
  print(c(var(rho_pearson), var(rho_spearman), var(rho_quad)))
}

#------------------------------------------------------------------------------
N <- 100
rho <- 0.9
mu1 <- 0; s1 <- 1
mu2 <- 0; s2 <- 1
mu <- c(mu1,mu2)

sigma <- matrix(c(s1^2, s1*s2*rho, s1*s2*rho, s2^2), 2)
bvn <- mvrnorm(N, mu = mu, Sigma = sigma )
colnames(bvn) <- c("x","y")

ellipse_bvn(bvn, N, rho) 
#------------------------------------------------------------------------------
N <- 60
rho <- 0
research_correlation(N, rho)
#------------------------------------------------------------------------------