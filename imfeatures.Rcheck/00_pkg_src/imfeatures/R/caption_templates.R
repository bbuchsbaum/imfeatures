#' Caption Prompt Templates and Management
#'
#' @description
#' Functions for building and managing image caption prompts for LLM-based
#' image description generation.
#'
#' @keywords internal
#' @name caption-templates
NULL

#' Focus areas for image captioning
#' @rdname caption-focus-map
#' @keywords internal
.caption_focus_map <- list(
  objects   = "Name concrete objects and their attributes (color, shape, material, count).",
  text      = "Transcribe any legible text exactly as it appears.",
  layout    = "Describe spatial arrangement (foreground/background, left/right, center, perspective).",
  colors    = "Mention prominent colors and relationships (dominant hues, accents).",
  materials = "Note textures and materials when visually evident.",
  actions   = "Describe actions or interactions, if any.",
  people    = "If people appear, describe non-identifying attributes (approximate age group, clothing); never guess identity."
)

#' Default negative instructions for captions
#' @rdname caption-negatives-default
#' @keywords internal
.caption_negatives_default <- c(
  "Avoid speculation that is not visible in the image.",
  "Avoid brand/identity guesses unless text is plainly visible.",
  "Avoid subjective marketing language (premium, refined, elegant)."
)

#' Caption template functions
#' @rdname caption-templates-list
#' @keywords internal
.caption_templates <- list(
  factual = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Provide a precise, objective visual description of the image in ",
      min_words, "-", max_words, " words.\n",
      if (length(focus_lines)) paste0("Focus on:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- ", paste(c(negatives, "Write in complete sentences."), collapse = "\n- ")
    )
  },
  dense = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Provide a thorough, information-dense description suited for retrieval in ",
      min_words, "-", max_words, " words.\n",
      "Enumerate salient entities, attributes, colors, text, and composition.\n",
      if (length(focus_lines)) paste0("Emphasize:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Be concrete and specific; avoid opinions.\n- ",
      paste(c(negatives, "Prefer nouns/adjectives over metaphors."), collapse = "\n- ")
    )
  },
  alt_text = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Write screen-reader-friendly ALT TEXT in ",
      min_words, "-", max_words, " words, capturing the essential content.\n",
      if (length(focus_lines)) paste0("Include:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Prioritize what a blind user needs to know.\n- ",
      paste(c(negatives, "Keep tone neutral; no redundant 'Image of...' preface."), collapse = "\n- ")
    )
  },
  product = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Describe the product(s) in the image in ", min_words, "-", max_words, " words.\n",
      "Include form, materials, colorway, condition, key features, and any legible labeling.\n",
      if (length(focus_lines)) paste0("Emphasize:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- No pricing or marketing hype.\n- ",
      paste(c(negatives, "Do not invent specs you cannot see."), collapse = "\n- ")
    )
  },
  art = function(min_words, max_words, focus_lines, negatives) {
    paste0(
      "Describe the artwork photographically and formally in ", min_words, "-", max_words, " words.\n",
      "Note medium, palette, composition, motifs, and visible inscriptions.\n",
      if (length(focus_lines)) paste0("Consider:\n- ", paste(focus_lines, collapse = "\n- "), "\n") else "",
      "Constraints:\n- Avoid speculative provenance or biography.\n- ",
      paste(c(negatives, "If style is clear (e.g., cubist), state cautiously."), collapse = "\n- ")
    )
  }
)

#' Build a caption prompt for image description
#'
#' @param template Character string specifying the caption template to use.
#'   Options are "factual", "dense", "alt_text", "product", or "art".
#' @param focus Character vector of focus areas. Valid options include
#'   "objects", "text", "layout", "colors", "materials", "actions", "people".
#' @param min_words Minimum number of words for the caption.
#' @param max_words Maximum number of words for the caption.
#' @param negatives Character vector of negative instructions to avoid certain
#'   types of descriptions.
#' @param extra_instructions Optional additional instructions to append.
#'
#' @return A character string containing the complete prompt for image captioning.
#'
#' @examples
#' \dontrun{
#' # Build a dense caption prompt focusing on objects and colors
#' prompt <- build_caption_prompt(
#'   template = "dense",
#'   focus = c("objects", "colors"),
#'   min_words = 60,
#'   max_words = 120
#' )
#' }
#'
#' @export
build_caption_prompt <- function(
  template = c("factual", "dense", "alt_text", "product", "art"),
  focus = c("objects", "text", "layout", "colors"),
  min_words = 40,
  max_words = 120,
  negatives = .caption_negatives_default,
  extra_instructions = NULL
) {
  template <- match.arg(template)
  
  # Validate focus keywords
  unknown <- setdiff(focus, names(.caption_focus_map))
  if (length(unknown)) {
    cli::cli_warn("Ignoring unknown focus keyword(s): {.val {unknown}}")
    focus <- intersect(focus, names(.caption_focus_map))
  }
  
  focus_lines <- unname(.caption_focus_map[focus])
  prompt <- .caption_templates[[template]](min_words, max_words, focus_lines, negatives)
  
  if (!is.null(extra_instructions) && nzchar(extra_instructions)) {
    prompt <- paste(prompt, "\nAdditional instructions:\n", extra_instructions)
  }
  
  prompt
}

#' Get available caption templates
#'
#' @return A character vector of available template names.
#' @export
get_caption_templates <- function() {
  names(.caption_templates)
}

#' Get available focus areas
#'
#' @return A character vector of available focus area names.
#' @export
get_caption_focus_areas <- function() {
  names(.caption_focus_map)
}