// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// To enable OpenMP:
// 1. Uncomment the next line
// // [[Rcpp::plugins(openmp)]]
// 2. Ensure your C++ compiler supports OpenMP and R is configured to use it
//    (e.g., by setting flags in ~/.R/Makevars, such as PKG_CXXFLAGS = -fopenmp)
// 3. Uncomment the #include <omp.h> line below
// 4. Uncomment the relevant '#pragma omp ...' lines within the functions

#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>

// Uncomment the following line if you enable OpenMP
// #include <omp.h>

using namespace Rcpp;
using namespace arma;

// -----------------------------------------------------------------------------
// 1) ENTROPY FUNCTION
// -----------------------------------------------------------------------------
inline double entropyCpp(const NumericVector &a) {
  double s = 0.0;
  for (double v : a) { s += v; }
  if (s <= 0.0) return 0.0;
  double inv_s = 1.0 / s;
  double out = 0.0;
  for (double v : a) {
    if (v > 0)
      out -= (v * inv_s) * std::log2(v * inv_s);
  }
  return out;
}

// -----------------------------------------------------------------------------
// 2) FIRST-ORDER ENTROPY (using response matrices)
// -----------------------------------------------------------------------------
double first_order_entropyCpp(const IntegerMatrix &resp_bin,
                              const NumericMatrix &resp_val,
                              int num_filters) {
  NumericVector first_order_bin(num_filters, 0.0);
  int nrow = resp_bin.nrow();
  int ncol = resp_bin.ncol();
  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c < ncol; c++) {
      int b = resp_bin(r, c); // Expecting 1-based index
      if (b >= 1 && b <= num_filters)
        first_order_bin[b - 1] += resp_val(r, c);
    }
  }
  return entropyCpp(first_order_bin);
}

// -----------------------------------------------------------------------------
// 3) CREATE GABOR FILTER
// -----------------------------------------------------------------------------
NumericMatrix create_gabor_cpp(int size, double theta, int octave) {
  double amplitude = 1.0;
  double phase = M_PI / 2.0;
  double frequency = std::pow(0.5, octave);
  int hrsf = 4;
  double sigma = 1.0 / (M_PI * frequency) *
    std::sqrt(std::log(2.0)/2.0) *
    ((std::pow(2.0, hrsf) + 1.0) / (std::pow(2.0, hrsf) - 1.0));
  double omega = 2.0 * M_PI * frequency;
  NumericMatrix gabor(size, size);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);
  double omega_sin = omega * sin_theta;
  double omega_cos = omega * cos_theta;
  double inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
  for (int r = 0; r < size; r++) {
    double y = (double)r - (double)size/2.0 + 1.0;
    for (int c = 0; c < size; c++) {
      double x = (double)c - (double)size/2.0 + 1.0;
      double gauss = std::exp(-(x*x + y*y) * inv_2sig2);
      double slant = x * omega_sin + y * omega_cos;
      gabor(r, c) = gauss * amplitude * std::cos(slant + phase);
    }
  }
  return gabor;
}

// -----------------------------------------------------------------------------
// 4) CREATE A FILTER BANK
// -----------------------------------------------------------------------------
List filter_bank_cpp(int num_filters, int flt_size, int octave = 3) {
  NumericVector bins_vec(num_filters);
  double step = 2.0 * M_PI / (double)num_filters;
  for (int i = 0; i < num_filters; i++) {
    bins_vec[i] = step * i;
  }

  // The filter bank is stored as a 3D array with dimensions:
  // [num_filters x flt_size x flt_size]. We use R's column-major convention.
  NumericVector flt_raw( Dimension(num_filters, flt_size, flt_size) );

  // --- Optional OpenMP Parallelization ---
  // #pragma omp parallel for
  for (int i = 0; i < num_filters; i++) {
    NumericMatrix gab = create_gabor_cpp(flt_size, bins_vec[i], octave);
    // --- Debug: Print range/sum of the first filter --- 
    if (i == 0) {
      Rcpp::Rcout << "[C++] filter_bank: First Gabor filter range: [" 
                  << Rcpp::min(gab) << ", " << Rcpp::max(gab) 
                  << "], Sum = " << Rcpp::sum(gab) << "\n";
    }
    // -----------------------------------------------
    for (int r = 0; r < flt_size; r++) {
      for (int c = 0; c < flt_size; c++) {
        flt_raw[i + num_filters * (r + flt_size * c)] = gab(r, c);
      }
    }
  }

  return List::create(
    _["bins_vec"]    = bins_vec,
    _["flt_weights"] = flt_raw,
    _["octave"]      = octave,
    _["flt_size"]    = flt_size
  );
}

// -----------------------------------------------------------------------------
// 5) 2D CONVOLUTION USING RcppArmadillo (calls arma::conv2 with "same" output)
// -----------------------------------------------------------------------------
arma::mat convolve2d_arma(const arma::mat& image, const arma::mat& kernel) {
  return arma::conv2(image, kernel, "same");
}

// -----------------------------------------------------------------------------
// 6) APPLY FILTER BANK TO AN IMAGE
// -----------------------------------------------------------------------------
List run_filterbank_cpp(const NumericMatrix &image_r, const List &fbank) {
  NumericVector bins_vec = fbank["bins_vec"];
  NumericVector flt_raw  = fbank["flt_weights"];
  int flt_size           = fbank["flt_size"];
  int num_filters        = bins_vec.size();
  
  // Convert R matrix to arma::mat.
  arma::mat image = Rcpp::as<arma::mat>(image_r); 
  int h = image.n_rows;
  int w = image.n_cols;

  // --- Debug: Print input image range --- 
  Rcpp::Rcout << "[C++] run_filterbank: Input image range: [" 
              << Rcpp::min(image_r) << ", " << Rcpp::max(image_r) 
              << "], Mean = " << Rcpp::mean(image_r) << "\n";
  // ------------------------------------
  
  // Prepare a 3D array for filtered responses: dimensions [num_filters, h, w].
  NumericVector img_filt( Dimension(num_filters, h, w) ); 

  // --- Get imager functions using Rcpp::Function --- 
  /* // REMOVED: Reverting to arma::conv2
  Environment imager_env = Environment::namespace_env("imager");
  Function convolve_func = imager_env["convolve"];
  Function as_cimg_func = imager_env["as.cimg"];
  */
  // -------------------------------------------------
  
  // --- Convert input image to cimg once ---
  /* // REMOVED
  SEXP image_cimg = as_cimg_func(image_r);
  */
  // ---------------------------------------
  
  for (int i = 0; i < num_filters; i++) {
    // Extract filter i into a NumericMatrix (required by as.cimg)
    /* // REMOVED
    NumericMatrix kernel_mat(flt_size, flt_size);
    */
    // Extract filter i into an arma::mat
    arma::mat kernel(flt_size, flt_size);
    for (int r = 0; r < flt_size; r++) {
      for (int c = 0; c < flt_size; c++) {
        // kernel_mat(r, c) = flt_raw[i + num_filters * (r + flt_size * c)];
        kernel(r, c) = flt_raw[i + num_filters * (r + flt_size * c)]; 
      }
    }

    // --- Convert kernel to cimg --- 
    /* // REMOVED
    SEXP kernel_cimg = as_cimg_func(kernel_mat);
    */
    // ----------------------------
    
    // --- Call imager::convolve --- 
    /* // REMOVED
    SEXP conv_result_sexp = convolve_func(image_cimg, kernel_cimg);
    */
    // -----------------------------
    
    // --- Convert result back to NumericMatrix --- 
    /* // REMOVED
    // The result of convolve is a cimg, convert it back for storage
    // (Alternatively, could work with SEXP directly, but matrix is easier)
    NumericMatrix conv_mat = as<NumericMatrix>(conv_result_sexp); 
    */
    // --- Call arma::conv2 --- 
    arma::mat conv = convolve2d_arma(image, kernel); // Use arma version again
    // ------------------------
    
    // --- Debug: Print range/sum of the first convolution result --- 
    if (i == 0) {
        Rcpp::Rcout << "[C++] run_filterbank: First convolution result (conv) range: [" 
                    << conv.min() << ", " << conv.max() // Use arma min/max
                    << "], Sum = " << arma::accu(conv) << "\n"; // Use arma accu
    }
    // -----------------------------------------------------------

    // Store result into the 3D array img_filt
    for (int r = 0; r < h; r++) {
      for (int c = 0; c < w; c++) {
        // Check bounds before assignment, although conv_mat should match h, w
        // if (r < conv_mat.nrow() && c < conv_mat.ncol()) {
        //      img_filt[i + num_filters * (r + h * c)] = conv_mat(r, c);
        // } else {
        //      Rcpp::warning("Convolution result dimension mismatch at filter %d, pixel (%d, %d)", i, r, c);
        //      img_filt[i + num_filters * (r + h * c)] = 0.0; // Assign zero if out of bounds
        // }
        // Direct assignment from arma::mat conv
        img_filt[i + num_filters * (r + h * c)] = conv(r, c);
      }
    }
  }
  
  // Determine per-pixel maximum responses and winning filter (R uses 1-based indices)
  IntegerMatrix resp_bin(h, w);
  NumericMatrix resp_val(h, w);
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      double max_val = -1.0e99;
      int max_idx = 1;
      for (int f = 0; f < num_filters; f++) {
        double val = img_filt[f + num_filters * (r + h * c)];
        if (val > max_val) {
          max_val = val;
          max_idx = f + 1;
        }
      }
      resp_bin(r, c) = max_idx;
      resp_val(r, c) = max_val;  // Storing raw maximum response
    }
  }
  
  Rcpp::Rcout << "[C++] run_filterbank: Range resp_val (raw max): [" 
              << Rcpp::min(resp_val) << ", " << Rcpp::max(resp_val) << "]\n";

  // Optionally, zero out border regions (nlines can be adjusted)
  int nlines = 2; // Enable border zeroing (matching R logic)
  if (nlines > 0) {
    if (h > 2*nlines && w > 2*nlines) {
        for (int r = 0; r < nlines; r++) {
            for (int c = 0; c < w; c++) {
                resp_val(r, c) = 0.0;
                resp_val(h - 1 - r, c) = 0.0;
            }
        }
        for (int c = 0; c < nlines; c++) {
            for (int r = nlines; r < h - nlines; r++) {
                resp_val(r, c) = 0.0;
                resp_val(r, w - 1 - c) = 0.0;
            }
        }
    } else {
         for (int r = 0; r < h; r++) {
             for (int c = 0; c < w; c++) {
          resp_val(r, c) = 0.0;
             }
         }
    }
  }
  
  Rcpp::Rcout << "[C++] run_filterbank: Range resp_val (after border zero): [" 
              << Rcpp::min(resp_val) << ", " << Rcpp::max(resp_val) << "]\n";

  return List::create(
    _["num_filters"] = num_filters,
    _["resp_bin"]    = resp_bin,
    _["resp_val"]    = resp_val
  );
}

// -----------------------------------------------------------------------------
// 7) DO COUNTING: Compute the pairwise histogram of filter responses.
// -----------------------------------------------------------------------------
List do_counting_cpp(const IntegerMatrix &resp_bin,
                     NumericMatrix &resp_val, // modified in place (zeroing out values below cutoff)
                     int maxdiag,
                     int circ_bins,
                     const NumericVector &bins_vec) {
  int h = resp_val.nrow();
  int w = resp_val.ncol();
  int size = h * w;

  // 1) Compute the average response before thresholding.
  double sum_resp = 0.0;
  for (int i = 0; i < size; i++) {
      sum_resp += resp_val[i]; 
  }
  double complex_before = (size > 0) ? (sum_resp / (double)size) : 0.0;
  Rcpp::Rcout << "[C++] do_counting: complex_before = " << complex_before << "\n";
  Rcpp::Rcout << "[C++] do_counting: Range resp_val (input): [" 
              << Rcpp::min(resp_val) << ", " << Rcpp::max(resp_val) << "]\n";

  // 2) Determine cutoff from the k-th highest response.
  std::vector<double> allresp;
  allresp.reserve(size);
  for (int i = 0; i < size; i++)
      allresp.push_back(resp_val[i]);
      int k_element = 10000;
      int actual_k = std::min(k_element, (int)allresp.size()); 
  double cutoff = 0.0;
  if (actual_k > 0) {
      std::partial_sort(allresp.begin(), allresp.begin() + actual_k, allresp.end(), std::greater<double>());
    cutoff = allresp[actual_k - 1];
  }
  Rcpp::Rcout << "[C++] do_counting: Calculated cutoff (k-th=" << k_element << "): " << cutoff << "\n";

  // 3) Zero out values below the cutoff.
  int non_zero_count_after_cutoff = 0;
  for (int i = 0; i < size; i++) {
    if (resp_val[i] < cutoff) {
      resp_val[i] = 0.0;
    } else if (resp_val[i] != 0.0) {
        non_zero_count_after_cutoff++;
    }
  }
  Rcpp::Rcout << "[C++] do_counting: Range resp_val (after cutoff): [" 
              << Rcpp::min(resp_val) << ", " << Rcpp::max(resp_val) << "]\n";
  Rcpp::Rcout << "[C++] do_counting: Non-zero pixels after cutoff: " 
              << non_zero_count_after_cutoff << "\n";

  // 4) Collect coordinates of nonzero responses.
  std::vector<int> ex, ey;
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      if (resp_val(r, c) != 0.0) {
        ex.push_back(r);
        ey.push_back(c);
      }
    }
  }
  int n = ex.size();
  int gabor_bins = bins_vec.size();
  
  // Initialize the 3D counts array: dimensions [maxdiag x circ_bins x gabor_bins]
  NumericVector counts( Dimension(maxdiag, circ_bins, gabor_bins), 0.0 );

  double angle_scale = (circ_bins > 0) ? (double)circ_bins / (2.0 * M_PI) : 0.0;
  double shift_scale = (gabor_bins > 0) ? (double)circ_bins / (double)gabor_bins : 0.0;

  // --- Optional OpenMP Parallelization can be applied here --- 
  for (int cp = 0; cp < n; cp++) {
    int xcp = ex[cp];
    int ycp = ey[cp];
    int orcp = resp_bin(xcp, ycp); // 1-based
    double valcp = resp_val(xcp, ycp);
    double shift = (double)orcp * shift_scale;
    for (int i = 0; i < n; i++) {
      int x = ex[i];
      int y = ey[i];
      int orp = resp_bin(x, y);
      double valp = resp_val(x, y);
      int orel = (orp - orcp + gabor_bins) % gabor_bins;
      int dx = x - xcp;
      int dy = y - ycp;
      double dd = std::sqrt((double)(dx*dx + dy*dy));
      int disti = static_cast<int>(std::round(dd));
      if (disti < 0) disti = 0;
      if (disti >= maxdiag) disti = maxdiag - 1;
      double angle = std::atan2((double)dy, (double)dx);
      double ddir  = angle * angle_scale + shift;
      int dir = static_cast<int>(std::floor(ddir + 0.5));
      dir = (dir % circ_bins + circ_bins) % circ_bins;
      counts[disti + maxdiag * (dir + circ_bins * orel)] += (valcp * valp);
    }
  }

  Rcpp::Rcout << "[C++] do_counting: Range counts cube: [" 
              << Rcpp::min(counts) << ", " << Rcpp::max(counts) 
              << "], Sum = " << Rcpp::sum(counts) << "\n";

  return List::create(
    _["counts"]         = counts,
    _["complex_before"] = complex_before
  );
}

// -----------------------------------------------------------------------------
// 8) DO STATISTICS: Normalize counts and compute circular statistics and Shannon entropy.
// -----------------------------------------------------------------------------
List do_statistics_cpp(const NumericVector &counts,
                       const NumericVector &bins_vec) {
  IntegerVector dims = counts.attr("dim");
  if (dims.length() != 3)
    stop("Input 'counts' must be a 3D array.");
  int maxdiag    = dims[0];
  int circ_bins  = dims[1];
  int gabor_bins = dims[2];
  
  if (bins_vec.length() != gabor_bins)
    stop("Length of 'bins_vec' must match 3rd dimension of 'counts'.");
  
  Rcpp::Rcout << "[C++] do_statistics: Input counts range: [" 
              << Rcpp::min(counts) << ", " << Rcpp::max(counts) 
              << "], Sum = " << Rcpp::sum(counts) << "\n";

  NumericVector counts_sum( Dimension(maxdiag, circ_bins), 0.0 );

  for (int d = 0; d < maxdiag; d++) {
    for (int c = 0; c < circ_bins; c++) {
      double s = 0.0;
      for (int g = 0; g < gabor_bins; g++) {
        s += counts[d + maxdiag * (c + circ_bins * g)];
      }
      counts_sum[d + maxdiag * c] = s + 1.0e-9;
    }
  }
  
  Rcpp::Rcout << "[C++] do_statistics: counts_sum range: [" 
              << Rcpp::min(counts_sum) << ", " << Rcpp::max(counts_sum) 
              << "], Sum = " << Rcpp::sum(counts_sum) << "\n";

  NumericVector normalized_counts( Dimension(maxdiag, circ_bins, gabor_bins), 0.0 );
  for (int d = 0; d < maxdiag; d++) {
    for (int c = 0; c < circ_bins; c++) {
      double sum_for_norm = counts_sum[d + maxdiag * c];
      if (sum_for_norm > 1.0e-10) {
          for (int g = 0; g < gabor_bins; g++) {
            int idx = d + maxdiag * (c + circ_bins * g);
          normalized_counts[idx] = counts[idx] / sum_for_norm;
        }
      }
    }
  }
  
  Rcpp::Rcout << "[C++] do_statistics: normalized_counts range: [" 
              << Rcpp::min(normalized_counts) << ", " << Rcpp::max(normalized_counts) 
              << "], Sum = " << Rcpp::sum(normalized_counts) << "\n";

  NumericMatrix circular_mean_angle(maxdiag, circ_bins);
  NumericMatrix circular_mean_length(maxdiag, circ_bins);
  NumericMatrix shannon(maxdiag, circ_bins);
  NumericMatrix shannon_nan(maxdiag, circ_bins);
  std::fill(shannon_nan.begin(), shannon_nan.end(), NA_REAL);

  double zar_cc = 1.0;
  if (gabor_bins > 1) {
      double dstep = 2.0 * M_PI / (double)gabor_bins;
      double sin_half_step = std::sin(dstep / 2.0);
    if (std::abs(sin_half_step) > 1e-9)
          zar_cc = (dstep / 2.0) / sin_half_step;
  }

  for (int d = 0; d < maxdiag; d++) {
    for (int c = 0; c < circ_bins; c++) {
      double sumx = 0.0, sumy = 0.0;
      NumericVector slice(gabor_bins);
      for (int g = 0; g < gabor_bins; g++) {
        int idx = d + maxdiag * (c + circ_bins * g);
        double val = normalized_counts[idx];
        slice[g] = val;
        sumx += val * std::cos(bins_vec[g]);
        sumy += val * std::sin(bins_vec[g]);
      }
      double mean_sumx = (gabor_bins > 0) ? sumx / gabor_bins : 0.0;
      double mean_sumy = (gabor_bins > 0) ? sumy / gabor_bins : 0.0;
      double ang = std::atan2(mean_sumy, mean_sumx);
      ang = std::fmod(ang + 2.0 * M_PI, 2.0 * M_PI);
      circular_mean_angle(d, c) = ang;
      double r = std::sqrt(mean_sumx * mean_sumx + mean_sumy * mean_sumy);
      circular_mean_length(d, c) = r * zar_cc;
      double e = entropyCpp(slice);
      shannon(d, c) = e;
      double cs = counts_sum[d + maxdiag * c] - 1.0e-9;
      if (cs > 1.0)
        shannon_nan(d, c) = e;
    }
  }
  
  Rcpp::Rcout << "[C++] do_statistics: shannon matrix range: [" 
              << Rcpp::min(shannon) << ", " << Rcpp::max(shannon) << "]\n";

  return List::create(
    _["normalized_counts"]    = normalized_counts,
    _["circular_mean_angle"]  = circular_mean_angle,
    _["circular_mean_length"] = circular_mean_length,
    _["shannon"]              = shannon,
    _["shannon_nan"]          = shannon_nan
  );
}

// -----------------------------------------------------------------------------
// 9) MAIN WRAPPER: edge_entropy_cpp
// -----------------------------------------------------------------------------

// [[Rcpp::export]]
DataFrame edge_entropy_cpp(const NumericMatrix &image,
                           std::string impath = "image",
                           int maxdiag = 500,
                           int gabor_bins = 24,
                           int filter_length = 31,
                           int circ_bins = 48,
                           SEXP rangesSEXP = R_NilValue)
{
  // Handle 'ranges' parameter.
  List ranges;
  if (Rf_isNull(rangesSEXP)) {
    ranges = List::create(
      IntegerVector::create(20, 80),
      IntegerVector::create(80, 160),
      IntegerVector::create(160, 240)
    );
  } else {
    if (!Rf_isNewList(rangesSEXP))
        stop("Parameter 'ranges' must be a list.");
    ranges = as<List>(rangesSEXP);
  }

  if (image.nrow() <= 0 || image.ncol() <= 0)
      stop("Input image has zero dimensions.");
  if (maxdiag <= 0) stop("maxdiag must be positive.");
  if (gabor_bins <= 0) stop("gabor_bins must be positive.");
  if (filter_length <= 0 || filter_length % 2 == 0)
    stop("filter_length must be positive and odd.");
  if (circ_bins <= 0) stop("circ_bins must be positive.");

  // 1) Build the filter bank.
  List fbank = filter_bank_cpp(gabor_bins, filter_length);

  // 2) Run the filter bank on the image (includes border zeroing if nlines > 0).
  List fres = run_filterbank_cpp(image, fbank);

  // 3) Calculate First-order entropy *after* border zeroing, *before* thresholding.
  // Need to access the resp_val modified by run_filterbank_cpp (borders zeroed). 
  // Need to copy it before passing to do_counting_cpp if modification is an issue.
  NumericMatrix resp_val_for_fo = clone(as<NumericMatrix>(fres["resp_val"]));
  double fo = first_order_entropyCpp(
    fres["resp_bin"],
    resp_val_for_fo, // Use the border-zeroed version
    gabor_bins
  );

  // 4) Counting: compute pairwise interactions (thresholds resp_val_mat in place).
  NumericMatrix resp_val_mat = as<NumericMatrix>(fres["resp_val"]); // Get potentially non-cloned version
  List cts = do_counting_cpp(fres["resp_bin"],
                              resp_val_mat, // note: this is modified in place
    maxdiag,
    circ_bins,
                              fbank["bins_vec"]);

  // 5) Compute statistics.
  List stats = do_statistics_cpp(cts["counts"], fbank["bins_vec"]);

  // 6) Summarize Shannon entropy over distance ranges.
  NumericMatrix shannon = stats["shannon"];
  int d = shannon.nrow();  // maxdiag
  int a = shannon.ncol();  // circ_bins

  NumericVector rowmeans(d);
  if (a > 0) {
      for (int rr = 0; rr < d; rr++) {
          double sumv = 0.0;
          for (int cc = 0; cc < a; cc++) {
              sumv += shannon(rr, cc);
          }
          rowmeans[rr] = sumv / (double)a;
      }
  }
  int nranges = ranges.size();
  NumericVector final_shannon(nranges, NA_REAL);
  for (int i = 0; i < nranges; i++) {
    SEXP range_element = ranges[i];
    if (!Rf_isInteger(range_element) || Rf_length(range_element) != 2) {
        warning("Element %d in 'ranges' list is not an integer vector of length 2. Skipping.", i + 1);
      continue;
    }
    IntegerVector rg = as<IntegerVector>(range_element);
    int start_r = rg[0];
    int end_r = rg[1];
    int start_idx = start_r - 1;
    int end_idx = end_r - 1;
    if (start_r <= 0 || end_r < start_r) {
      warning("Invalid range [%d, %d] at index %d. Skipping.", start_r, end_r, i + 1);
        continue;
    }
    double accum = 0.0;
    int count = 0;
    for (int idx = start_idx; idx <= end_idx; idx++) {
      if (idx >= 0 && idx < d) {
        accum += rowmeans[idx];
        count++;
      }
    }
    if (count > 0)
      final_shannon[i] = accum / (double)count;
  }
  
  Rcpp::Rcout << "[C++] edge_entropy: Calculated fo = " << fo << "\n";
  
  Rcpp::Rcout << "[C++] edge_entropy: rowmeans (shannon) range: [" 
              << Rcpp::min(rowmeans) << ", " << Rcpp::max(rowmeans) << "]\n";
              
  Rcpp::Rcout << "[C++] edge_entropy: final_shannon[0] = " << ((final_shannon.size() >= 1) ? final_shannon[0] : NA_REAL) << "\n";
  Rcpp::Rcout << "[C++] edge_entropy: final_shannon[1] = " << ((final_shannon.size() >= 2) ? final_shannon[1] : NA_REAL) << "\n";
  Rcpp::Rcout << "[C++] edge_entropy: final_shannon[2] = " << ((final_shannon.size() >= 3) ? final_shannon[2] : NA_REAL) << "\n";

  double complex_before = cts["complex_before"];

  return DataFrame::create(
    _["im"]               = impath,
    _["entropy"]          = fo,
    _["pentropy_20_80"]   = (final_shannon.size() >= 1) ? final_shannon[0] : NA_REAL,
    _["pentropy_80_160"]  = (final_shannon.size() >= 2) ? final_shannon[1] : NA_REAL,
    _["pentropy_160_240"] = (final_shannon.size() >= 3) ? final_shannon[2] : NA_REAL,
    _["complex_before"]   = complex_before
  );
}