#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h> // For standard math like isnan, isinf, fabs for non-MPFR parts
#include <mpfr.h>
#include <omp.h>

// --- Global MPFR Precision ---
#define MPFR_PRECISION 256 // Bits of precision for MPFR

// --- Helper: Initialize an array of mpfr_t variables ---
void mpfr_array_init(mpfr_t *arr, int size) {
    for (int i = 0; i < size; ++i) {
        mpfr_init2(arr[i], MPFR_PRECISION);
    }
}

// --- Helper: Clear an array of mpfr_t variables ---
void mpfr_array_clear(mpfr_t *arr, int size) {
    for (int i = 0; i < size; ++i) {
        mpfr_clear(arr[i]);
    }
}

// --- MPFR Logistic Map: x_next = r * x_curr * (1 - x_curr) ---
// All arguments are mpfr_t. temp1 is a pre-initialized temporary.
static inline void logistic_map_mpfr(mpfr_t x_next, const mpfr_t x_curr, const mpfr_t r_param, mpfr_t temp1) {
    mpfr_set_ui(temp1, 1, MPFR_RNDN);                 // temp1 = 1
    mpfr_sub(temp1, temp1, x_curr, MPFR_RNDN);        // temp1 = 1 - x_curr
    mpfr_mul(x_next, x_curr, temp1, MPFR_RNDN);       // x_next = x_curr * (1 - x_curr)
    mpfr_mul(x_next, x_next, r_param, MPFR_RNDN);     // x_next = r * x_curr * (1 - x_curr)
}

// --- "Standard" Logistic Map (double x, long double r) for endpoints data ---
static inline double logistic_map_double_x_ld_r(double x, long double r_ld) {
    return (double)(r_ld * x * (1.0L - (long double)x));
}


// --- Generate Endpoints Data (using OpenMP, double x, long double r) ---
void generate_endpoints_data(const char *filename,
                             long double r_min_ld, long double r_max_ld, long double r_step_ld,
                             int n_trajectories_per_r,
                             int m_length_of_trajectory,
                             int use_random_x0) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open endpoints data file");
        return;
    }
    fprintf(fp, "r,x\n");
    printf("Generating endpoints data for r in [%.3Lf, %.3Lf] with step %.4Lf...\n", r_min_ld, r_max_ld, r_step_ld);
    printf("%d trajectories of length %d per r value.\n", n_trajectories_per_r, m_length_of_trajectory);

    // Estimate total r values for progress reporting
    long num_r_values = (r_max_ld - r_min_ld) / r_step_ld + 1;
    long r_values_processed = 0;

    // Buffer for thread-local output strings
    // Max line length: approx. 10 (for r) + 1 (,) + 20 (for x_end) + 1 (\n) + safety = ~35-40 chars
    // Let's say 1000 lines per buffer flush = 40KB buffer per thread
    #define THREAD_BUFFER_LINES 1000
    #define MAX_LINE_LEN 50 // Increased for safety with long double r

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)omp_get_thread_num();
        char* thread_output_buffer = (char*)malloc(THREAD_BUFFER_LINES * MAX_LINE_LEN);
        int current_buffer_pos = 0;

        #pragma omp for schedule(dynamic) // Dynamic schedule for uneven r processing times (less likely here)
        for (long i = 0; i < num_r_values; ++i) {
            long double r_ld = r_min_ld + i * r_step_ld;
            if (r_ld > r_max_ld + r_step_ld*0.5L) continue; // Ensure we don't overshoot due to fp issues

            for (int traj = 0; traj < n_trajectories_per_r; ++traj) {
                double x;
                if (use_random_x0) {
                    x = (double)rand_r(&seed) / (double)RAND_MAX;
                    if (x < 1e-6) x = 1e-6;
                    if (x > 1.0 - 1e-6) x = 1.0 - 1e-6;
                } else {
                    x = 0.5 + ((double)rand_r(&seed) / (double)RAND_MAX * 1e-3 - 0.5e-3);
                }
                 if (x <=0.0) x = 0.01; if (x >=1.0) x = 0.99;


                for (int iter = 0; iter < m_length_of_trajectory; ++iter) {
                    x = logistic_map_double_x_ld_r(x, r_ld);
                }
                
                int len = snprintf(thread_output_buffer + current_buffer_pos, MAX_LINE_LEN, "%.6Lf,%.8f\n", r_ld, x);
                if (len > 0 && len < MAX_LINE_LEN) {
                    current_buffer_pos += len;
                }

                if (current_buffer_pos >= (THREAD_BUFFER_LINES -1) * MAX_LINE_LEN || (traj == n_trajectories_per_r -1 && i % 10 == 0) ) { // Flush buffer
                    #pragma omp critical
                    {
                        fwrite(thread_output_buffer, 1, current_buffer_pos, fp);
                    }
                    current_buffer_pos = 0;
                }
            }
            #pragma omp atomic
            r_values_processed++;
            if (omp_get_thread_num() == 0 && (r_values_processed % (num_r_values / 20 + 1)) == 0) {
                 printf("Endpoints generation: %.1f%% complete (r approx %.4Lf)\n", 
                       (double)r_values_processed * 100.0 / num_r_values, r_ld);
            }
        }
        // Flush any remaining buffer content for this thread
        if (current_buffer_pos > 0) {
            #pragma omp critical
            {
                fwrite(thread_output_buffer, 1, current_buffer_pos, fp);
            }
        }
        free(thread_output_buffer);
    } // end parallel region
    fclose(fp);
    printf("Endpoints data written to %s\n", filename);
}

// --- MPFR Helper for qsort ---
// Global temporary for mpfr_cmp, initialized once
// This is NOT thread-safe if qsort itself is parallelized by some C libraries,
// but standard qsort is sequential. If parallel sort is used, this needs rethinking.
// For now, assuming qsort is called sequentially.
// mpfr_t g_mpfr_temp_for_cmp1, g_mpfr_temp_for_cmp2; // Avoid global like this
// Instead, pass struct with pointer for context or use thread local if qsort is tricky.
// Simpler: make comparison take const void* and cast to mpfr_ptr.
int compare_mpfr(const void *a, const void *b) {
    // const mpfr_t *pa = (const mpfr_t *)a; // This is pointer to mpfr_t struct
    // const mpfr_t *pb = (const mpfr_t *)b; // This is pointer to mpfr_t struct
    // We are sorting an array of mpfr_t, so a and b are pointers to mpfr_t elements
    return mpfr_cmp(*(const mpfr_ptr*)a, *(const mpfr_ptr*)b);
}


// --- Get Distinct Attractor Points (MPFR version) ---
int get_distinct_attractor_points_mpfr(const mpfr_t r, const mpfr_t x0_start,
                                       int burn_in, int num_to_collect,
                                       mpfr_t *attractor_points_buffer, // Array of mpfr_t
                                       const mpfr_t distinct_tol_mpfr,
                                       mpfr_t temp_x, mpfr_t temp_map_calc) { // Pre-inited temps
    mpfr_set(temp_x, x0_start, MPFR_RNDN);

    for (int i = 0; i < burn_in; ++i) {
        logistic_map_mpfr(temp_x, temp_x, r, temp_map_calc);
        if (mpfr_nan_p(temp_x) || mpfr_inf_p(temp_x)) return 0; // Diverged
    }

    for (int i = 0; i < num_to_collect; ++i) {
        logistic_map_mpfr(temp_x, temp_x, r, temp_map_calc);
        if (mpfr_nan_p(temp_x) || mpfr_inf_p(temp_x)) { // Diverged during collection
             // Fill rest with NaN to indicate problem for sorting/counting if needed
             for(int j=i; j < num_to_collect; ++j) mpfr_set_nan(attractor_points_buffer[j]);
             break;
        }
        mpfr_set(attractor_points_buffer[i], temp_x, MPFR_RNDN);
    }
    
    // Sort using mpfr_cmp
    // qsort takes array of elements. Here, array of mpfr_t.
    // Each mpfr_t is a struct, but qsort needs pointers to them.
    // So, we need an array of mpfr_ptr.
    mpfr_ptr *ptr_array = (mpfr_ptr*)malloc(num_to_collect * sizeof(mpfr_ptr));
    if (!ptr_array) { perror("malloc for ptr_array failed"); return 0; }
    for(int i=0; i<num_to_collect; ++i) ptr_array[i] = attractor_points_buffer[i];

    qsort(ptr_array, num_to_collect, sizeof(mpfr_ptr), compare_mpfr);
    
    // After sorting ptr_array, copy back to attractor_points_buffer if needed, or work with ptr_array
    // For distinct count, we can iterate through sorted ptr_array.

    if (num_to_collect == 0) { free(ptr_array); return 0; }
    if (mpfr_nan_p(ptr_array[0]) || mpfr_inf_p(ptr_array[0])) { free(ptr_array); return 0;}


    int distinct_count = 1;
    mpfr_t diff;
    mpfr_init2(diff, MPFR_PRECISION);

    for (int i = 1; i < num_to_collect; ++i) {
        if (mpfr_nan_p(ptr_array[i]) || mpfr_inf_p(ptr_array[i])) break; // Stop if diverged
        mpfr_sub(diff, ptr_array[i], ptr_array[i-1], MPFR_RNDN);
        mpfr_abs(diff, diff, MPFR_RNDN);
        if (mpfr_cmp(diff, distinct_tol_mpfr) > 0) {
            distinct_count++;
        }
    }
    mpfr_clear(diff);
    free(ptr_array);
    return distinct_count;
}


// --- Find Bifurcation r Value (MPFR version) ---
void find_bifurcation_r_mpfr(mpfr_t result_r_bif, int current_period,
                             const mpfr_t r_low_guess, const mpfr_t r_high_guess,
                             const mpfr_t x0_search_mpfr,
                             int burn_in_search, int collect_search_points,
                             const mpfr_t distinct_tol_mpfr, int max_bisection_iter,
                             const mpfr_t r_precision_mpfr) {
    mpfr_t low_r, high_r, mid_r, diff_r;
    mpfr_init2(low_r, MPFR_PRECISION); mpfr_init2(high_r, MPFR_PRECISION);
    mpfr_init2(mid_r, MPFR_PRECISION); mpfr_init2(diff_r, MPFR_PRECISION);

    mpfr_set(low_r, r_low_guess, MPFR_RNDN);
    mpfr_set(high_r, r_high_guess, MPFR_RNDN);
    mpfr_set(result_r_bif, r_high_guess, MPFR_RNDN); // Default

    mpfr_t *attractor_points = (mpfr_t*)malloc(collect_search_points * sizeof(mpfr_t));
    if (!attractor_points) { perror("malloc for attractor_points failed"); goto cleanup_bisection; }
    mpfr_array_init(attractor_points, collect_search_points);

    mpfr_t temp_x_map, temp_map_calc_intern; // Temps for get_distinct_attractor_points_mpfr
    mpfr_init2(temp_x_map, MPFR_PRECISION); mpfr_init2(temp_map_calc_intern, MPFR_PRECISION);

    for (int i = 0; i < max_bisection_iter; ++i) {
        mpfr_sub(diff_r, high_r, low_r, MPFR_RNDN);
        if (mpfr_cmp(diff_r, r_precision_mpfr) <= 0) break;

        mpfr_add(mid_r, low_r, high_r, MPFR_RNDN);
        mpfr_div_ui(mid_r, mid_r, 2, MPFR_RNDN);

        if (mpfr_cmp(mid_r, low_r) <= 0 || mpfr_cmp(mid_r, high_r) >=0) { // Interval stalled
             //printf("Bisection interval stalled. mid_r, low_r, high_r: ");
             //mpfr_printf("%.30Rf, %.30Rf, %.30Rf\n", mid_r, low_r, high_r);
             break;
        }

        int num_distinct = get_distinct_attractor_points_mpfr(mid_r, x0_search_mpfr,
                                                              burn_in_search, collect_search_points,
                                                              attractor_points, distinct_tol_mpfr,
                                                              temp_x_map, temp_map_calc_intern);
        
        if (num_distinct == 0 && mpfr_cmp_d(mid_r, 3.0) > 0) { // Likely divergence or error
            //mpfr_printf("Warning: num_distinct is 0 for r = %.30Rf. Assuming it's past bifurcation or diverged.\n", mid_r);
             mpfr_set(result_r_bif, mid_r, MPFR_RNDN);
             mpfr_set(high_r, mid_r, MPFR_RNDN); // Try lower
             continue;
        }


        if (num_distinct > current_period) {
            mpfr_set(result_r_bif, mid_r, MPFR_RNDN);
            mpfr_set(high_r, mid_r, MPFR_RNDN);
        } else {
            mpfr_set(low_r, mid_r, MPFR_RNDN);
        }
    }
    mpfr_set(result_r_bif, high_r, MPFR_RNDN); // Best estimate is high_r driven down

    mpfr_array_clear(attractor_points, collect_search_points);
    free(attractor_points);
    mpfr_clear(temp_x_map); mpfr_clear(temp_map_calc_intern);

cleanup_bisection:
    mpfr_clear(low_r); mpfr_clear(high_r); mpfr_clear(mid_r); mpfr_clear(diff_r);
}

// --- Root Finding (double x, long double r - not using MPFR here for speed unless strictly needed) ---
// ... (Keep the Newton-Raphson from the previous long double version if needed for polynomial roots)
// ... For this example, I'll omit the separate polynomial root finding section to focus on Feigenbaum.
// ... If you need it, it can be adapted. The prompt focused on bifurcations for Feigenbaum.

// --- Main Program ---
int main(int argc, char **argv) {
    mpfr_set_default_prec(MPFR_PRECISION);
    srand((unsigned int)time(NULL));

    // --- Part 1: Generate Endpoints Data (using OpenMP) ---
    printf("Part 1: Generating endpoints data for plotting (using OpenMP)...\n");
    // Use long double for r in this part for speed, range 0-4 doesn't need MPFR for plotting.
    generate_endpoints_data("endpoints_data_3_445_4.csv", // bifurcation_data.csv, endpoints_data (_mpfr)
                            3.445L, 4L, 0.00002L,
                            1000, 5000, 1);

    // --- Part 2: Find Bifurcation Points (MPFR) and Estimate Feigenbaum Constant ---
    printf("\nPart 2: Finding bifurcation points (MPFR) and Feigenbaum delta...\n");
    const int MAX_BIFURCATIONS_TO_FIND = 23; // Reduced for reasonable runtime with MPFR
    mpfr_t r_bif[MAX_BIFURCATIONS_TO_FIND];
    mpfr_t deltas[MAX_BIFURCATIONS_TO_FIND - 2];
    mpfr_array_init(r_bif, MAX_BIFURCATIONS_TO_FIND);
    mpfr_array_init(deltas, MAX_BIFURCATIONS_TO_FIND - 2 > 0 ? MAX_BIFURCATIONS_TO_FIND - 2 : 1);


    mpfr_t x0_fb_search_mpfr, distinct_tol_mpfr, r_precision_fb_mpfr;
    mpfr_init2(x0_fb_search_mpfr, MPFR_PRECISION); mpfr_set_d(x0_fb_search_mpfr, 0.50123456789, MPFR_RNDN);
    mpfr_init2(distinct_tol_mpfr, MPFR_PRECISION); mpfr_set_ld(distinct_tol_mpfr, 1e-25L, MPFR_RNDN); // MPFR tolerance for x
    mpfr_init2(r_precision_fb_mpfr, MPFR_PRECISION); mpfr_set_ld(r_precision_fb_mpfr, 1e-40L, MPFR_RNDN); // MPFR tolerance for r

    int burn_in_fb = 50000;       // Increased burn-in for MPFR precision
    int collect_fb = 1 << 12;     // Collect 2^(k+2) points e.g., for 2^10 cycle, need 2^12 to be safe = 4096
    int max_bisection_fb = 200;   // More bisection iterations

    int actual_bifurcations_found = 0;

    // Temporary MPFR variables for search ranges
    mpfr_t r_low_s, r_high_s;
    mpfr_init2(r_low_s, MPFR_PRECISION); mpfr_init2(r_high_s, MPFR_PRECISION);

    // Find r_0 (1->2 cycle)
    printf("Finding r_0 (1->2 cycles)...\n");
    mpfr_set_d(r_low_s, 2.99, MPFR_RNDN);
    mpfr_set_d(r_high_s, 3.01, MPFR_RNDN); // Known to be 3.0
    find_bifurcation_r_mpfr(r_bif[0], 1, r_low_s, r_high_s, x0_fb_search_mpfr,
                              burn_in_fb, 1 << 4, // Collect 16 for 1->2
                              distinct_tol_mpfr, max_bisection_fb, r_precision_fb_mpfr);
    mpfr_printf("r_0 (onset of 2-cycle): %.40RNf\n", r_bif[0]);
    if (mpfr_cmp_d(r_bif[0], 2.0) > 0) actual_bifurcations_found = 1;
    else { printf("Failed to find r_0. Exiting.\n"); goto end_main; }


    mpfr_t feigenbaum_delta_approx, temp1, temp2;
    mpfr_init2(feigenbaum_delta_approx, MPFR_PRECISION); mpfr_set_str(feigenbaum_delta_approx, "4.6692016091", 10, MPFR_RNDN); // Approx, base 10
    mpfr_init2(temp1, MPFR_PRECISION); mpfr_init2(temp2, MPFR_PRECISION);

    for (int k_idx = 1; k_idx < MAX_BIFURCATIONS_TO_FIND; ++k_idx) {
        int current_period_val = 1 << k_idx; // Period that IS bifurcating (e.g. k_idx=1, period is 2)
        printf("Finding r_%d (%d->%d cycles)...\n", k_idx, current_period_val, 2 * current_period_val);

        mpfr_set(r_low_s, r_bif[k_idx-1], MPFR_RNDN);
        mpfr_add_ui(r_low_s, r_low_s, 0, MPFR_RNDN); // r_low_s = r_bif[k_idx-1]
        mpfr_add(r_low_s, r_low_s, r_precision_fb_mpfr, MPFR_RNDN); // Start just above previous

        if (k_idx == 1) { // r_1 (2->4)
            mpfr_set_d(r_high_s, 3.45, MPFR_RNDN); // Initial guess for r_1 high
        } else {
            mpfr_sub(temp1, r_bif[k_idx-1], r_bif[k_idx-2], MPFR_RNDN); // prev_diff = r_{k-1} - r_{k-2}
            mpfr_div(temp2, temp1, feigenbaum_delta_approx, MPFR_RNDN); // prev_diff / delta_approx
            mpfr_add(r_high_s, r_bif[k_idx-1], temp2, MPFR_RNDN);     // r_high_s = r_{k-1} + estimate
        }
        // Sanity check r_high_s
        mpfr_set_str(temp1, "3.569945671869542", 10, MPFR_RNDN); // Accumulation point approx R_inf, base 10
        if (mpfr_cmp(r_high_s, r_low_s) <= 0) { // If estimate is bad
            mpfr_set(r_high_s, r_low_s, MPFR_RNDN);
            mpfr_add_d(r_high_s, r_high_s, (k_idx <3 ? 0.01 : 0.001) / (1L << (k_idx-1)), MPFR_RNDN); // Add small offset
        }
         if (mpfr_cmp(r_high_s, temp1) >=0) { // if high_s > R_inf
            mpfr_set(r_high_s, temp1, MPFR_RNDN);
            mpfr_sub(r_high_s, r_high_s, r_precision_fb_mpfr, MPFR_RNDN); // slightly less than R_inf
        }
         if (mpfr_cmp(r_low_s, r_high_s) >=0) {
             mpfr_printf("Search range for r_%d invalid. Low: %.30Rf, High: %.30Rf. Stopping.\n", k_idx, r_low_s, r_high_s);
             break;
         }

        int current_collect_pts = 1 << (k_idx + 3); // Collect enough points for 2*current_period_val
        if (current_collect_pts > collect_fb) current_collect_pts = collect_fb; // Cap

        find_bifurcation_r_mpfr(r_bif[k_idx], current_period_val, r_low_s, r_high_s, x0_fb_search_mpfr,
                                  burn_in_fb, current_collect_pts,
                                  distinct_tol_mpfr, max_bisection_fb, r_precision_fb_mpfr);
        mpfr_printf("r_%d (onset of %d-cycle): %.40RNf\n", k_idx, 2*current_period_val, r_bif[k_idx]);

        mpfr_sub(temp1, r_bif[k_idx], r_bif[k_idx-1], MPFR_RNDN); // diff = r_k - r_{k-1}
        if (mpfr_cmp(temp1, r_precision_fb_mpfr) <= 0 || mpfr_sgn(temp1) <=0 ) { // If diff is too small or negative
            mpfr_printf("Bifurcation r_%d too close to r_%d or out of order. Stopping.\n", k_idx, k_idx-1);
            break;
        }
        actual_bifurcations_found++;
         if (mpfr_cmp(r_bif[k_idx], temp1) >=0 && k_idx > 3) { // Approaching accumulation point rapidly
             mpfr_printf("r_%d getting very close to accumulation point. May stop early.\n", k_idx);
             // Could add a check: if r_bif[k_idx] is very close to temp1 (R_inf approx), break.
         }
    }


    printf("\nFeigenbaum Delta Estimates (MPFR):\n");
    if (actual_bifurcations_found >= 3) {
        for (int k = 0; k < actual_bifurcations_found - 2; ++k) {
            // Delta_k = (r_{k+1} - r_k) / (r_{k+2} - r_{k+1})
            mpfr_sub(temp1, r_bif[k+1], r_bif[k], MPFR_RNDN);       // Numerator: r_{k+1} - r_k
            mpfr_sub(temp2, r_bif[k+2], r_bif[k+1], MPFR_RNDN);     // Denominator: r_{k+2} - r_{k+1}

            if (mpfr_sgn(temp1) <=0 || mpfr_sgn(temp2) <=0 ) {
                 mpfr_printf("Delta_%d: Invalid r sequence for calculation.\n", k+1);
                 continue;
            }
            if (mpfr_cmp_ui(temp2, 0) == 0) { // Denominator is zero
                mpfr_printf("Delta_%d: Denominator is zero. Cannot calculate.\n", k+1);
                continue;
            }
            mpfr_div(deltas[k], temp1, temp2, MPFR_RNDN);
            mpfr_printf("Delta_%d = (r_%d - r_%d) / (r_%d - r_%d) = %.35RNf\n",
                   k+1, k+1, k, k+2, k+1, deltas[k]);
        }
    } else {
        printf("Not enough bifurcation points found (%d) to calculate delta values.\n", actual_bifurcations_found);
    }

    mpfr_clear(temp1); mpfr_clear(temp2);
    mpfr_clear(feigenbaum_delta_approx);
    mpfr_clear(r_low_s); mpfr_clear(r_high_s);

end_main:
    mpfr_array_clear(r_bif, MAX_BIFURCATIONS_TO_FIND);
    mpfr_array_clear(deltas, MAX_BIFURCATIONS_TO_FIND - 2 > 0 ? MAX_BIFURCATIONS_TO_FIND - 2 : 1);
    mpfr_clear(x0_fb_search_mpfr); mpfr_clear(distinct_tol_mpfr); mpfr_clear(r_precision_fb_mpfr);

    // --- Cleanup MPFR global state ---
    mpfr_free_cache(); // Frees cache used by MPFR internally.
    // mpfr_clear_default_prec(); // If you changed it from a previous global state

    printf("\nDone.\n");
    return 0;
}