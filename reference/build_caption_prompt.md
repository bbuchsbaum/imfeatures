# Build a caption prompt for image description

Build a caption prompt for image description

## Usage

``` r
build_caption_prompt(
  template = c("factual", "dense", "alt_text", "product", "art"),
  focus = c("objects", "text", "layout", "colors"),
  min_words = 40,
  max_words = 120,
  negatives = .caption_negatives_default,
  extra_instructions = NULL
)
```

## Arguments

- template:

  Character string specifying the caption template to use. Options are
  "factual", "dense", "alt_text", "product", or "art".

- focus:

  Character vector of focus areas. Valid options include "objects",
  "text", "layout", "colors", "materials", "actions", "people".

- min_words:

  Minimum number of words for the caption.

- max_words:

  Maximum number of words for the caption.

- negatives:

  Character vector of negative instructions to avoid certain types of
  descriptions.

- extra_instructions:

  Optional additional instructions to append.

## Value

A character string containing the complete prompt for image captioning.

## Examples

``` r
if (FALSE) { # \dontrun{
# Build a dense caption prompt focusing on objects and colors
prompt <- build_caption_prompt(
  template = "dense",
  focus = c("objects", "colors"),
  min_words = 60,
  max_words = 120
)
} # }
```
