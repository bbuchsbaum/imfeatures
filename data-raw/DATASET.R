## code to prepare `DATASET` dataset goes here

places_categories <- read.table("data-raw/places365.txt")
path <- places_categories$V1
category <- sapply(strsplit(categories, "/"), function(x) x[length(x)])
places_cat365 <- data.frame(index=1:length(path), path=path, category=category)
usethis::use_data(places_cat365, overwrite = TRUE)
