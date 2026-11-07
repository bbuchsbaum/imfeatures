# Package index

## Image and saliency entropy

- [`compute_edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md)
  [`edge_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_edge_entropy.md)
  : Calculate Edge Entropy Features from Images
- [`compute_saliency_features()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_saliency_features.md)
  : Saliency-weighted edge entropy with spatial controls
- [`saliency_location_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/saliency_location_entropy.md)
  : Saliency location entropy and effective attended fraction
- [`saliency_multiscale_entropy()`](https://bbuchsbaum.github.io/imfeatures/reference/saliency_multiscale_entropy.md)
  : Multiscale location entropy of a saliency map
- [`saliency_donor_assignments()`](https://bbuchsbaum.github.io/imfeatures/reference/saliency_donor_assignments.md)
  : Reproducible donor-map permutations for saliency controls
- [`image_mse()`](https://bbuchsbaum.github.io/imfeatures/reference/image_mse.md)
  : Multiscale entropy for images

## Feature hierarchies

- [`as_feature_hierarchy()`](https://bbuchsbaum.github.io/imfeatures/reference/as_feature_hierarchy.md)
  : Build a named feature-hierarchy object
- [`subset_feature_blocks()`](https://bbuchsbaum.github.io/imfeatures/reference/subset_feature_blocks.md)
  : Subset the observation rows of feature blocks
- [`hierarchy_recipe()`](https://bbuchsbaum.github.io/imfeatures/reference/hierarchy_recipe.md)
  : Compile a hierarchy transformation recipe
- [`block_pca()`](https://bbuchsbaum.github.io/imfeatures/reference/block_pca.md)
  : Specify per-stage PCA compression
- [`ordered_innovation()`](https://bbuchsbaum.github.io/imfeatures/reference/ordered_innovation.md)
  : Specify ordered innovation residualization
- [`fit_hierarchy_transform()`](https://bbuchsbaum.github.io/imfeatures/reference/fit_hierarchy_transform.md)
  : Fit an inductive hierarchy transformation
- [`apply_hierarchy_transform()`](https://bbuchsbaum.github.io/imfeatures/reference/apply_hierarchy_transform.md)
  [`predict(`*`<hierarchy_transform>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/apply_hierarchy_transform.md)
  : Apply a fitted hierarchy transformation
- [`residualize_tiers()`](https://bbuchsbaum.github.io/imfeatures/reference/residualize_tiers.md)
  : Hierarchical residualization of tiered feature matrices with PCA
- [`predict(`*`<residualized_tiers>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/predict.residualized_tiers.md)
  : Predict method for residualized_tiers
- [`vgg_hierarchy_spec()`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_hierarchy_spec.md)
  : Specify a VGG feature hierarchy
- [`vgg16_block_end_spec()`](https://bbuchsbaum.github.io/imfeatures/reference/vgg16_block_end_spec.md)
  : VGG-16 block-end hierarchy with depth-adaptive DCT pooling
- [`vgg19_block_end_spec()`](https://bbuchsbaum.github.io/imfeatures/reference/vgg19_block_end_spec.md)
  : VGG-19 block-end hierarchy with depth-adaptive DCT pooling
- [`vgg_block_end_taps()`](https://bbuchsbaum.github.io/imfeatures/reference/vgg_block_end_taps.md)
  : Block-end tap names for a VGG architecture
- [`extract_feature_hierarchy()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_feature_hierarchy.md)
  : Extract a named feature hierarchy from images
- [`global_average_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/global_average_pool.md)
  : Global average pooling over spatial axes
- [`global_max_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/global_max_pool.md)
  : Global max pooling over spatial axes
- [`adaptive_grid_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/adaptive_grid_pool.md)
  : Adaptive rectangular grid pooling
- [`dct_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/dct_pool.md)
  : Low-frequency 2-D DCT pooling
- [`radial_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/radial_pool.md)
  : Radial centre/periphery pooling
- [`identity_pool()`](https://bbuchsbaum.github.io/imfeatures/reference/identity_pool.md)
  : Leave activations unchanged, flattening 4-D maps
- [`pool_activations()`](https://bbuchsbaum.github.io/imfeatures/reference/pool_activations.md)
  : Apply a pooler to an activation tensor
- [`pooler_feature_names()`](https://bbuchsbaum.github.io/imfeatures/reference/pooler_feature_names.md)
  : Feature names for a pooled stage
- [`as_nhwc()`](https://bbuchsbaum.github.io/imfeatures/reference/as_nhwc.md)
  : Convert a 4-D activation tensor to NHWC layout
- [`linear_cka()`](https://bbuchsbaum.github.io/imfeatures/reference/linear_cka.md)
  : Linear-kernel CKA between two observation-by-feature matrices
- [`effective_rank()`](https://bbuchsbaum.github.io/imfeatures/reference/effective_rank.md)
  : Spectral participation-ratio effective rank
- [`hierarchy_diagnostics()`](https://bbuchsbaum.github.io/imfeatures/reference/hierarchy_diagnostics.md)
  : Stimulus-only diagnostics for a fitted hierarchy transform
- [`check_transform_invariance()`](https://bbuchsbaum.github.io/imfeatures/reference/check_transform_invariance.md)
  : Check batch invariance and row-permutation equivariance

## Deep network features

- [`extract_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features.md)
  [`im_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features.md)
  : extract features from intermediate layers
- [`extract_vgg_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_vgg_features.md)
  : Extract VGG-16 features by tier
- [`print(`*`<vgg_feature_set>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/print.vgg_feature_set.md)
  : Print method for vgg_feature_set objects
- [`compute_feature_similarity()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_feature_similarity.md)
  [`im_feature_sim()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_feature_similarity.md)
  : Compute similarity matrix for a set of image using feature vectors
  from a Keras 3 model
- [`im_predict()`](https://bbuchsbaum.github.io/imfeatures/reference/im_predict.md)
  : predict the class of an image using Keras model
- [`load_vgg16_places()`](https://bbuchsbaum.github.io/imfeatures/reference/load_vgg16_places.md)
  [`vgg16_places()`](https://bbuchsbaum.github.io/imfeatures/reference/load_vgg16_places.md)
  : Load VGG16 Places365 model
- [`places_cat365`](https://bbuchsbaum.github.io/imfeatures/reference/places_cat365.md)
  : Places 365 categories
- [`plot_im_scatter()`](https://bbuchsbaum.github.io/imfeatures/reference/plot_im_scatter.md)
  [`im_scatter()`](https://bbuchsbaum.github.io/imfeatures/reference/plot_im_scatter.md)
  [`plot_im_scatter3d()`](https://bbuchsbaum.github.io/imfeatures/reference/plot_im_scatter.md)
  [`im_scatter3d()`](https://bbuchsbaum.github.io/imfeatures/reference/plot_im_scatter.md)
  : a 2D scatterplot with images displayed at each location

## thingsvision backend

- [`tv_get_extractor()`](https://bbuchsbaum.github.io/imfeatures/reference/tv_get_extractor.md)
  : Get a thingsvision extractor object
- [`print(`*`<thingsvision_extractor>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/print.thingsvision_extractor.md)
  : Print method for thingsvision_extractor objects
- [`show_model()`](https://bbuchsbaum.github.io/imfeatures/reference/show_model.md)
  : Show Model Architecture for thingsvision_extractor
- [`tv_extract()`](https://bbuchsbaum.github.io/imfeatures/reference/tv_extract.md)
  : Generic feature extraction
- [`tv_extract(`*`<thingsvision_extractor>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/tv_extract.thingsvision_extractor.md)
  : Extract features using a thingsvision_extractor object
- [`tv_align(`*`<thingsvision_extractor>`*`)`](https://bbuchsbaum.github.io/imfeatures/reference/tv_align.thingsvision_extractor.md)
  : Align features using a thingsvision_extractor object
- [`tv_create_dataset()`](https://bbuchsbaum.github.io/imfeatures/reference/tv_create_dataset.md)
  : Create a thingsvision ImageDataset
- [`tv_create_dataloader()`](https://bbuchsbaum.github.io/imfeatures/reference/tv_create_dataloader.md)
  : Create a thingsvision DataLoader
- [`extract_features_tv()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features_tv.md)
  [`im_features_tv()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_features_tv.md)
  : Extract Deep Learning Features using thingsvision
- [`compute_feature_similarity_tv()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_feature_similarity_tv.md)
  [`im_feature_sim_tv()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_feature_similarity_tv.md)
  : Compute Similarity Matrix using thingsvision Features
- [`get_backend()`](https://bbuchsbaum.github.io/imfeatures/reference/get_backend.md)
  : Get Backend Name
- [`get_transformations()`](https://bbuchsbaum.github.io/imfeatures/reference/get_transformations.md)
  : Get Preprocessing Transformations
- [`get_default_module_name()`](https://bbuchsbaum.github.io/imfeatures/reference/get_default_module_name.md)
  : Get default module name for a model
- [`list_module_names()`](https://bbuchsbaum.github.io/imfeatures/reference/list_module_names.md)
  : List available module names for a model
- [`get_recommended_layers()`](https://bbuchsbaum.github.io/imfeatures/reference/get_recommended_layers.md)
  : Get recommended layers for feature extraction

## CLIP, captions and text

- [`clip_features()`](https://bbuchsbaum.github.io/imfeatures/reference/clip_features.md)
  : Extract CLIP embeddings (final or intermediate layers)
- [`clip_text_features()`](https://bbuchsbaum.github.io/imfeatures/reference/clip_text_features.md)
  : Extract CLIP text features from text strings
- [`caption_features()`](https://bbuchsbaum.github.io/imfeatures/reference/caption_features.md)
  : Extract Caption Features from Images
- [`caption_features_many()`](https://bbuchsbaum.github.io/imfeatures/reference/caption_features_many.md)
  : Batch caption feature extraction with progress
- [`build_caption_prompt()`](https://bbuchsbaum.github.io/imfeatures/reference/build_caption_prompt.md)
  : Build a caption prompt for image description
- [`get_caption_templates()`](https://bbuchsbaum.github.io/imfeatures/reference/get_caption_templates.md)
  : Get available caption templates
- [`get_caption_focus_areas()`](https://bbuchsbaum.github.io/imfeatures/reference/get_caption_focus_areas.md)
  : Get available focus areas
- [`list_caption_providers()`](https://bbuchsbaum.github.io/imfeatures/reference/list_caption_providers.md)
  : List available caption providers
- [`get_default_caption_model()`](https://bbuchsbaum.github.io/imfeatures/reference/get_default_caption_model.md)
  : Get default model for provider
- [`check_caption_provider_auth()`](https://bbuchsbaum.github.io/imfeatures/reference/check_caption_provider_auth.md)
  : Check if provider API key is configured
- [`embed_text()`](https://bbuchsbaum.github.io/imfeatures/reference/embed_text.md)
  : Compute text embeddings
- [`cosine_similarity()`](https://bbuchsbaum.github.io/imfeatures/reference/cosine_similarity.md)
  : Compute cosine similarity between text embeddings
- [`compute_caption_similarity()`](https://bbuchsbaum.github.io/imfeatures/reference/compute_caption_similarity.md)
  : Compute similarity between caption embeddings
- [`extract_multimodal_features()`](https://bbuchsbaum.github.io/imfeatures/reference/extract_multimodal_features.md)
  : Extract multimodal features (visual + caption)

## Python setup

- [`imfeatures_config()`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md)
  [`install_imfeatures_python()`](https://bbuchsbaum.github.io/imfeatures/reference/imfeatures_config.md)
  : Configure the Python environment for imfeatures
- [`install_python_deps()`](https://bbuchsbaum.github.io/imfeatures/reference/install_python_deps.md)
  : Install Python dependencies for imfeatures
- [`use_existing_python()`](https://bbuchsbaum.github.io/imfeatures/reference/use_existing_python.md)
  : Use existing Python installation for imfeatures
- [`configure_hpc_python()`](https://bbuchsbaum.github.io/imfeatures/reference/configure_hpc_python.md)
  : Configure Python for HPC systems
- [`hpc_setup`](https://bbuchsbaum.github.io/imfeatures/reference/hpc_setup.md)
  : HPC Setup for imfeatures
