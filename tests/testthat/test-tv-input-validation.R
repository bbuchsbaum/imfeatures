context("thingsvision input validation")

test_that("im_features_tv checks non-empty arguments", {
  tmp <- tempfile(fileext = ".png")
  file.create(tmp)
  expect_error(
    im_features_tv(impaths = tmp, model_name = "", source = "torchvision", module_name = "avgpool"),
    "model_name'")
  unlink(tmp)
})

test_that("im_feature_sim_tv checks non-empty arguments", {
  tmp1 <- tempfile(fileext = ".png")
  tmp2 <- tempfile(fileext = ".png")
  file.create(c(tmp1, tmp2))
  expect_error(
    im_feature_sim_tv(impaths = c(tmp1, tmp2), model_name = "", source = "torchvision", module_names = "avgpool"),
    "model_name'")
  unlink(c(tmp1, tmp2))
})
