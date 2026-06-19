# predict the class of an image using Keras model

predict the class of an image using Keras model

## Usage

``` r
im_predict(impath, model = NULL, target_size = c(224, 224), topn = 12)
```

## Arguments

- impath:

  path to image file

- model:

  the Keras model

- target_size:

  the target image dimensions for approproate for model

- topn:

  number of top predictions to return (default: 12)

## Examples

``` r
if (FALSE) { # \dontrun{
# Predict class of a single image
img_path <- system.file("extdata", "dog.jpg", package = "imfeatures")

# Use default VGG16 model trained on ImageNet
predictions <- im_predict(img_path, topn = 5)
print(predictions)  # Top 5 predicted classes with scores

# Use a custom pre-loaded model
library(keras3)
resnet_model <- application_resnet50(weights = 'imagenet')
predictions_resnet <- im_predict(
  impath = img_path,
  model = resnet_model,
  target_size = c(224, 224),
  topn = 10
)

# Predict using VGG16-Places365 for scene recognition
places_model <- load_vgg16_places()
scene_predictions <- im_predict(
  impath = "path/to/landscape.jpg",
  model = places_model,
  topn = 3
)
# Will return scene categories like "mountain", "forest", etc.
} # }
```
