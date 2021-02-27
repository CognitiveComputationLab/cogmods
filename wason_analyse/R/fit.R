################################################################################
#### R script used to fit the data from Ragni, Kola, & Johnson-Laird (2018) ####
################################################################################





## required packages
library(MPTinR)
library(gtools)


dat4 <- read.table("/Users/matze/Nextcloud/Bachelorthesis/Code/r_test/4pattern_ragni.txt", header=TRUE)
dat16 <- read.table("/Users/matze/Nextcloud/Bachelorthesis/Code/r_test/16pattern_ragni.csv", header=TRUE, sep=";")


head(dat4)
head(dat16)



mod_inf_g <- "
(1-a) * (1-p) * (1-notP) * (1-q) * (1-notQ)
a * c * (1-d) * (1-s) * i + (1-a) * (1-p) * (1-notP) * (1-q) * notQ
a * c * (1 - d) * s * i + (1-a) * (1-p) * (1-notP) * q * (1-notQ)
a * (1-c ) * (1-x) * (1-d) * i  + (1-a) * (1-p) * (1-notP) * q * notQ
a * c * d * (1-s) * i + (1-a) * (1-p) * notP * (1-q) * (1-notQ)
a * (1-c) * x * (1-s) * i + (1-a) * (1-p) * notP * (1-q) * notQ
a * c * d * (1-s) * (1-i) + a * c * (1 - d) * s * (1-i) + (1-a) * (1-p) * notP * q * (1-notQ)
(1-a) * (1-p) * notP * q * notQ
a * c * d * s * i + (1-a) * p * (1-notP) * (1-q) * (1-notQ)
a * c * d * s * (1-i) + a * c * (1-d) * (1-s) * (1-i) + (1-a) * p * (1-notP) * (1-q) * notQ
a * (1-c) * x * s * i + (1-a) * p * (1-notP) * q * (1-notQ)
(1-a) * p * (1-notP) * q * notQ
a * (1-c) * (1-x) * d * i  + (1-a) * p * notP * (1-q) * (1-notQ)
(1-a) * p * notP * (1-q) * notQ
(1-a) * p * notP * q * (1-notQ)
a * (1-c) * (1-x) * (1-d) * (1-i) + a * (1-c) * (1-x) * d * (1-i) + a * (1-c) * x * (1-s) * (1-i) + a * (1-c) * x * s * (1-i) + (1-a) * p * notP * q * notQ
"


mod_inf_g_expand <- "
(1-a) * (1-p) * (1-notP) * (1-q) * (1-notQ)
a * c * (1-d) * (1-sb) * i + (1-a) * (1-p) * (1-notP) * (1-q) * notQ
a * c * (1 - d) * sb * i + (1-a) * (1-p) * (1-notP) * q * (1-notQ)
a * (1-c ) * (1-x) * (1-d) * i  + (1-a) * (1-p) * (1-notP) * q * notQ
a * c * d * (1-sf) * i + (1-a) * (1-p) * notP * (1-q) * (1-notQ)
a * (1-c) * x * (1-sfb) * i + (1-a) * (1-p) * notP * (1-q) * notQ
a * c * d * (1-sf) * (1-i) + a * c * (1 - d) * sb * (1-i) + (1-a) * (1-p) * notP * q * (1-notQ)
(1-a) * (1-p) * notP * q * notQ
a * c * d * sf * i + (1-a) * p * (1-notP) * (1-q) * (1-notQ)
a * c * d * sf * (1-i) + a * c * (1-d) * (1-sb) * (1-i) + (1-a) * p * (1-notP) * (1-q) * notQ
a * (1-c) * x * sfb * i + (1-a) * p * (1-notP) * q * (1-notQ)
(1-a) * p * (1-notP) * q * notQ
a * (1-c) * (1-x) * d * i  + (1-a) * p * notP * (1-q) * (1-notQ)
(1-a) * p * notP * (1-q) * notQ
(1-a) * p * notP * q * (1-notQ)
a * (1-c) * (1-x) * (1-d) * (1-i) + a * (1-c) * (1-x) * d * (1-i) + a * (1-c) * x * (1-sfb) * (1-i) + a * (1-c) * x * sfb * (1-i) + (1-a) * p * notP * q * notQ
"



psycop <- "
pc*pi
(1-pc)*pi
pc*(1-pi)
(1-pc)*(1-pi)
"


check.mpt(textConnection(mod_inf_g))
check.mpt(textConnection(mod_inf_g_expand))
check.mpt(textConnection(psycop))




#abstract
a_mod_inf_g  <- fit.mpt(dat16[dat16$content =="abstract",-1],textConnection(mod_inf_g))
a_mod_inf_g_expand  <- fit.mpt(dat16[dat16$content =="abstract",-1],textConnection(mod_inf_g_expand))
a_psycop <- fit.mpt(dat4[dat4$content =="abstract",-1],textConnection(psycop))

# everyday
e_mod_inf_g  <- fit.mpt(dat16[dat16$content =="generalization",-1],textConnection(mod_inf_g))
e_mod_inf_g_expand  <- fit.mpt(dat16[dat16$content =="generalization",-1],textConnection(mod_inf_g_expand))
e_psycop <- fit.mpt(dat4[dat4$content =="generalization",-1],textConnection(psycop))


#deontic
d_mod_inf_g  <- fit.mpt(dat16[dat16$content =="deontic",-1],textConnection(mod_inf_g))
d_mod_inf_g_expand  <- fit.mpt(dat16[dat16$content =="deontic",-1],textConnection(mod_inf_g_expand))
d_psycop <- fit.mpt(dat4[dat4$content =="deontic",-1],textConnection(psycop))


#overall
fit_mod_inf_g  <- fit.mpt(dat16[,-1],textConnection(mod_inf_g))
fit_mod_inf_g_expand  <- fit.mpt(dat16[,-1],textConnection(mod_inf_g_expand))
fit_psycop <- fit.mpt(dat4[,-1],textConnection(psycop))


#advantage of expanded inf_g
# abstract
round(a_mod_inf_g$information.criteria$aggregated[2]-a_mod_inf_g_expand$information.criteria$aggregated[2],1)

# everyday
round(e_mod_inf_g$information.criteria$aggregated[2]-e_mod_inf_g_expand$information.criteria$aggregated[2],1)

# deontic
round(d_mod_inf_g$information.criteria$aggregated[2]-d_mod_inf_g_expand$information.criteria$aggregated[2],1)

# overall
round(fit_mod_inf_g$information.criteria$aggregated[2]-fit_mod_inf_g_expand$information.criteria$aggregated[2],1)


#Gsquare for every Dataset individual and aggregated
round(fit_mod_inf_g$goodness.of.fit$individual$G.Squared,8)
round(fit_mod_inf_g$goodness.of.fit$aggregated$G.Squared,8)

round(fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared,8)
round(fit_mod_inf_g_expand$goodness.of.fit$aggregated$G.Squared,8)

round(fit_psycop$goodness.of.fit$individual$G.Squared,8)
round(fit_psycop$goodness.of.fit$aggregated$G.Squared,8)


sel_mod_inf_g <- which(round(fit_mod_inf_g$goodness.of.fit$individual$G.Squared,5) >0)

seldat_mod_inf_g <- dat16[sel_mod_inf_g,]

sel_mod_inf_g_expand <- which(round(fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared,5) >0)

seldat_mod_inf_g_expand <- dat16[sel_mod_inf_g_expand,]

sel_psycop <- which(round(fit_psycop$goodness.of.fit$individual$G.Squared,5) >0)

seldat_psycop <- dat4[sel_psycop,]


# misfits with gsquare > 0
selfit_mod_inf_g <- fit_mod_inf_g$goodness.of.fit$individual$G.Squared[which(round(fit_mod_inf_g$goodness.of.fit$individual$G.Squared,5) >0)]
selfit_mod_inf_g_expand <- fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared[which(round(fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared,5) >0)]
selfit_psycop <- fit_psycop$goodness.of.fit$individual$G.Squared[which(round(fit_psycop$goodness.of.fit$individual$G.Squared,5) >0)]



#error of each prediction
miss_mod_inf_g <- (fit_mod_inf_g$data$predicted$individual-fit_mod_inf_g$data$observed$individual)/matrix(rep(rowSums(dat16[,-1]),each=16),ncol=16,byrow=TRUE)
miss_mod_inf_g_expand <- (fit_mod_inf_g_expand$data$predicted$individual-fit_mod_inf_g_expand$data$observed$individual)/matrix(rep(rowSums(dat16[,-1]),each=16),ncol=16,byrow=TRUE)
miss_psycop <- (fit_psycop$data$predicted$individual-fit_psycop$data$observed$individual)/matrix(rep(rowSums(dat4[,-1]),each=4),ncol=4,byrow=TRUE)


mis_sel_mod_inf_g <- miss_mod_inf_g[which(round(fit_mod_inf_g$goodness.of.fit$individual$G.Squared,5) > 0),]
mis_sel_mod_inf_g_expand <- miss_mod_inf_g_expand[which(round(fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared,5) > 0),]
mis_sel_psycop <- miss_psycop[which(round(fit_psycop$goodness.of.fit$individual$G.Squared,5) > 0),]



mis_dat_mod_inf_g <- dat16[which(round(fit_mod_inf_g$goodness.of.fit$individual$G.Squared,5) > 0),]
mis_dat_mod_inf_g_expand <- dat16[which(round(fit_mod_inf_g_expand$goodness.of.fit$individual$G.Squared,5) > 0),]
mis_dat_psycop <- dat4[which(round(fit_psycop$goodness.of.fit$individual$G.Squared,5) > 0),]


mean(mis_sel_mod_inf_g[,1] < 0) # underestimate
mean(mis_sel_mod_inf_g[,2] > 0) # overestimate
mean(mis_sel_mod_inf_g[,4] < 0) # underestimate
mean(mis_sel_mod_inf_g_expand[,1] < 0) # underestimate
mean(mis_sel_mod_inf_g_expand[,2] > 0) # overestimate
mean(mis_sel_mod_inf_g_expand[,4] < 0) # underestimate
mean(mis_sel_psycop[,1] < 0) # underestimate
mean(mis_sel_psycop[,2] > 0) # overestimate
mean(mis_sel_psycop[,4] < 0) # underestimate



### table 2 ###
nrow(dat16[dat$content=="abstract",])
nrow(dat16[dat$content=="generalization",])
nrow(dat16[dat$content=="deontic",])
nrow(dat16)

nrow(dat4[dat$content=="abstract",])
nrow(dat4[dat$content=="generalization",])
nrow(dat4[dat$content=="deontic",])
nrow(dat4)

nrow(mis_dat_mod_inf_g[mis_dat_mod_inf_g $content=="abstract",])/nrow(dat16[dat16$content=="abstract",])
nrow(mis_dat_mod_inf_g[mis_dat_mod_inf_g $content=="generalization",])/nrow(dat16[dat16$content=="generalization",])
nrow(mis_dat_mod_inf_g[mis_dat_mod_inf_g $content=="deontic",])/nrow(dat16[dat16$content=="deontic",])
nrow(mis_sel_mod_inf_g)/nrow(dat16)

nrow(mis_dat_mod_inf_g_expand[mis_dat_mod_inf_g_expand$content=="abstract",])/nrow(dat16[dat16$content=="abstract",])
nrow(mis_dat_mod_inf_g_expand[mis_dat_mod_inf_g_expand$content=="generalization",])/nrow(dat16[dat16$content=="generalization",])
nrow(mis_dat_mod_inf_g_expand[mis_dat_mod_inf_g_expand$content=="deontic",])/nrow(dat16[dat16$content=="deontic",])
nrow(mis_sel_mod_inf_g_expand)/nrow(dat16)

nrow(mis_dat_psycop[mis_dat_psycop$content=="abstract",])/nrow(dat4[dat4$content=="abstract",])
nrow(mis_dat_mod_inf_g[mis_dat_psycop $content=="generalization",])/nrow(dat4[dat4$content=="generalization",])
nrow(mis_dat_mod_inf_g[mis_dat_psycop $content=="deontic",])/nrow(dat4[dat4$content=="deontic",])
nrow(mis_dat_psycop)/nrow(dat4)

