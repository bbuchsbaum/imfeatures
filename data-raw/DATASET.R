## code to prepare `DATASET` dataset goes here

places_categories <- read.table("data-raw/places365.txt", header=TRUE)
usethis::use_data(places_categories, overwrite = TRUE)
