
## Stats 490 – Homework 3 R code
## Questions: 4.8, 4.13, 4.19, 4.28, 4.45, 4.47

library(dplyr)
library(ggplot2)
library(agricolae)

########################
## 4.8 – Cloth strength
########################

bolt <- factor(rep(1:5, each = 4))
chemical <- factor(rep(1:4, times = 5))

tensile <- c(
  73, 68, 74, 71,
  67, 73, 67, 75,
  72, 70, 75, 68,
  78, 73, 68, 73,
  71, 75, 75, 69
)

dat48 <- data.frame(
  tensile = tensile,
  chemical = chemical,
  bolt = bolt
)

fit48 <- aov(tensile ~ chemical + bolt, data = dat48)
summary(fit48)

chem_summary48 <- dat48 |>
  group_by(chemical) |>
  summarise(
    n = n(),
    mean = mean(tensile),
    sd = sd(tensile),
    .groups = "drop"
  )
chem_summary48

bolt_summary48 <- dat48 |>
  group_by(bolt) |>
  summarise(
    n = n(),
    mean = mean(tensile),
    sd = sd(tensile),
    .groups = "drop"
  )
bolt_summary48

TukeyHSD(fit48, "chemical")
pairwise.t.test(dat48$tensile, dat48$chemical,
                p.adjust.method = "none")

##############################
## 4.13 – Brochure responses
##############################

region <- factor(rep(c("NE", "NW", "SE", "SW"), each = 3))
design <- factor(rep(1:3, times = 4))

responses <- c(
  250, 400, 275,  # NE
  350, 525, 340,  # NW
  219, 390, 200,  # SE
  375, 580, 310   # SW
)

dat413 <- data.frame(
  responses = responses,
  design = design,
  region = region
)

fit413 <- aov(responses ~ design + region, data = dat413)
summary(fit413)

design_summary413 <- dat413 |>
  group_by(design) |>
  summarise(
    n = n(),
    mean = mean(responses),
    sd = sd(responses),
    .groups = "drop"
  )
design_summary413

region_summary413 <- dat413 |>
  group_by(region) |>
  summarise(
    n = n(),
    mean = mean(responses),
    sd = sd(responses),
    .groups = "drop"
  )
region_summary413

pairwise_LSD_413 <- pairwise.t.test(dat413$responses, dat413$design,
                                    p.adjust.method = "none",
                                    pool.sd = TRUE)
pairwise_LSD_413

TukeyHSD(fit413, "design")

###############################
## 4.19 – Grain size (RCBD)
###############################

stir <- factor(rep(c(5, 10, 15, 20), each = 4))
furnace <- factor(rep(1:4, times = 4))

grain <- c(
  8,  4,  5,  6,  # 5 rpm
  14, 5,  6,  9,  # 10 rpm
  14, 6,  9,  2,  # 15 rpm
  17, 9,  3,  6   # 20 rpm
)

dat419 <- data.frame(
  grain = grain,
  stir = stir,
  furnace = furnace
)

fit419 <- aov(grain ~ stir + furnace, data = dat419)
summary(fit419)

stir_summary419 <- dat419 |>
  group_by(stir) |>
  summarise(
    n = n(),
    mean = mean(grain),
    sd = sd(grain),
    .groups = "drop"
  )
stir_summary419

furnace_summary419 <- dat419 |>
  group_by(furnace) |>
  summarise(
    n = n(),
    mean = mean(grain),
    sd = sd(grain),
    .groups = "drop"
  )
furnace_summary419

#############################
## 4.28 – Latin square ANOVA
#############################

order_ls <- factor(rep(1:4, each = 4))
operator_ls <- factor(rep(1:4, times = 4))

method <- c(
  "C", "D", "A", "B",
  "B", "C", "D", "A",
  "A", "B", "C", "D",
  "D", "A", "B", "C"
)

time <- c(
  10, 14, 7, 8,
  7, 18, 11, 8,
  5, 10, 11, 9,
  10, 10, 12, 14
)

method <- factor(method, levels = c("A", "B", "C", "D"))

dat428 <- data.frame(
  time = time,
  method = method,
  operator = operator_ls,
  order = order_ls
)

fit428 <- aov(time ~ method + operator + order, data = dat428)
summary(fit428)

method_summary428 <- dat428 |>
  group_by(method) |>
  summarise(
    n = n(),
    mean = mean(time),
    sd = sd(time),
    .groups = "drop"
  )
method_summary428

operator_summary428 <- dat428 |>
  group_by(operator) |>
  summarise(
    n = n(),
    mean = mean(time),
    sd = sd(time),
    .groups = "drop"
  )
operator_summary428

order_summary428 <- dat428 |>
  group_by(order) |>
  summarise(
    n = n(),
    mean = mean(time),
    sd = sd(time),
    .groups = "drop"
  )
order_summary428

pairwise_LSD_428 <- pairwise.t.test(dat428$time, dat428$method,
                                    p.adjust.method = "none",
                                    pool.sd = TRUE)
pairwise_LSD_428

TukeyHSD(fit428, "method")

#############################################
## 4.45 – BIBD for gasoline additives
#############################################

additive <- c(
  1, 1, 1, 1,  # additive 1 on cars 2,3,4,5
  2, 2, 2, 2,  # additive 2 on cars 1,2,3,5
  3, 3, 3, 3,  # additive 3 on cars 1,3,4,5
  4, 4, 4, 4,  # additive 4 on cars 1,2,3,4
  5, 5, 5, 5   # additive 5 on cars 1,2,3,5
)

car <- c(
  2, 3, 4, 5,  # additive 1
  1, 2, 3, 5,  # additive 2
  1, 3, 4, 5,  # additive 3
  1, 2, 3, 4,  # additive 4
  1, 2, 3, 5   # additive 5
)

mileage <- c(
  17, 14, 13, 12,  # additive 1
  14, 14, 13, 10,  # additive 2
  12, 13, 12,  9,  # additive 3
  13, 11, 11, 12,  # additive 4
  11, 12, 10,  8   # additive 5
)

additive <- factor(additive)
car <- factor(car)

dat445 <- data.frame(
  mileage = mileage,
  additive = additive,
  car = car
)

fit445 <- aov(mileage ~ additive + car, data = dat445)
summary(fit445)

add_summary445 <- dat445 |>
  group_by(additive) |>
  summarise(
    n = n(),
    mean = mean(mileage),
    sd = sd(mileage),
    .groups = "drop"
  )
add_summary445

car_summary445 <- dat445 |>
  group_by(car) |>
  summarise(
    n = n(),
    mean = mean(mileage),
    sd = sd(mileage),
    .groups = "drop"
  )
car_summary445

LSD_add_445 <- LSD.test(fit445, "additive", p.adj = "none")
LSD_add_445

pairwise_LSD_445 <- pairwise.t.test(dat445$mileage, dat445$additive,
                                    p.adjust.method = "none",
                                    pool.sd = TRUE)
pairwise_LSD_445

#############################################
## 4.47 – BIBD for hardwood concentration
#############################################

conc <- factor(c(
  2, 4, 8,
  4, 6, 10,
  6, 8, 12,
  8, 10, 14,
  2, 10, 12,
  2, 12, 14,
  4, 6, 14
))

day <- factor(rep(1:7, each = 3))

strength <- c(
  114, 126, 141,
  120, 137, 145,
  117, 129, 120,
  149, 150, 136,
  120, 143, 118,
  117, 123, 130,
  119, 134, 127
)

dat447 <- data.frame(
  strength = strength,
  conc = conc,
  day = day
)

fit447 <- aov(strength ~ conc + day, data = dat447)
summary(fit447)

conc_summary447 <- dat447 |>
  group_by(conc) |>
  summarise(
    n = n(),
    mean = mean(strength),
    sd = sd(strength),
    .groups = "drop"
  )
conc_summary447

day_summary447 <- dat447 |>
  group_by(day) |>
  summarise(
    n = n(),
    mean = mean(strength),
    sd = sd(strength),
    .groups = "drop"
  )
day_summary447

LSD_conc_447 <- LSD.test(fit447, "conc", p.adj = "none")
LSD_conc_447