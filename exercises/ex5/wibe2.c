#include <stdio.h>
#include <stdlib.h>
#include <math.h>   // For general math, use fabsl, etc. for long double
#include <string.h> // For memcpy
#include <time.h>   // For seeding random

// Use long double for PI if needed, though not directly used in logistic map
// #define PI_LD 3.14159265358979323846264338327950288L

// --- Core Logistic Map ---
// x remains double, r is long double for precision in parameter space
static inline double logistic_map(double x, long double r) {
    // Ensure calculations involving r are done with r's precision
    // The cast to double for x ensures x stays within typical float range
    return (double)(r * x * (1.0L - (long double)x));
}

// --- Iterated Map and its Derivative (for polynomial root finding) ---
// Computes f^k(x, r)
double iterate_map_k_times(double x, long double r, int k) {
    for (int i = 0; i < k; ++i) {
        x = logistic_map(x, r);
    }
    return x;
}

// Computes derivative d(f^k(x,r))/dx
// f'(x,r) = r*(1-2x)
// (f^k(x))' = f'(f^{k-1}(x)) * f'(f^{k-2}(x)) * ... * f'(x)
double derivative_iterate_map_k_times(double x_start, long double r, int k) {
    if (k == 0) return 1.0;

    long double derivative_val_ld = 1.0L; // Use long double for intermediate product
    double x_current = x_start;
    
    // Max k for storing all iterates could be large, 
    // for very large k, consider an alternative if memory is an issue.
    // However, k for Feigenbaum typically doesn't exceed ~10-15 for f^2^k
    // For f^k for roots, k can be larger. Let's assume k is manageable.
    double x_iterates_stack[k > 0 ? k : 1]; // VLA if k > 0, or fixed size for k=0
    double *x_iterates = x_iterates_stack;
    // A more robust solution for potentially large k might use heap allocation
    // if (k > SOME_THRESHOLD) x_iterates = malloc(k * sizeof(double)); ... free(x_iterates);

    for (int i = 0; i < k; ++i) {
        x_iterates[i] = x_current;
        x_current = logistic_map(x_current, r);
    }

    for (int i = 0; i < k; ++i) {
        derivative_val_ld *= r * (1.0L - 2.0L * (long double)x_iterates[i]);
    }
    return (double)derivative_val_ld;
}


// --- New: Generate Endpoints Data (n trajectories of length m per r) ---
void generate_endpoints_data(const char *filename,
                             long double r_min, long double r_max, long double r_step,
                             int n_trajectories_per_r,
                             int m_length_of_trajectory,
                             int use_random_x0) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open endpoints data file");
        return;
    }
    fprintf(fp, "r,x\n"); // CSV header
    printf("Generating endpoints data for r in [%.3Lf, %.3Lf] with step %.4Lf...\n", r_min, r_max, r_step);
    printf("%d trajectories of length %d per r value.\n", n_trajectories_per_r, m_length_of_trajectory);

    long double current_r_report_threshold = r_min;
    int report_interval_count = 0; // For periodic status updates

    for (long double r = r_min; r <= r_max + r_step*0.5L ; r += r_step) { // Add r_step*0.5L to handle potential fp inaccuracies with <=
        if (r >= current_r_report_threshold) {
             //printf("Processing r = %.4Lf\n", r); // Can be too verbose
             current_r_report_threshold += (r_max - r_min) / 20.0L; // Report progress ~20 times
             if (current_r_report_threshold < r) current_r_report_threshold = r + r_step; // ensure progress
             report_interval_count++;
             if(report_interval_count % 5 == 0) printf("... r up to %.3Lf\n", r);
        }

        for (int traj = 0; traj < n_trajectories_per_r; ++traj) {
            double x;
            if (use_random_x0) {
                x = (double)rand() / (double)RAND_MAX; 
                if (x < 1e-6) x = 1e-6; 
                if (x > 1.0 - 1e-6) x = 1.0 - 1e-6;
            } else { // fixed perturbed start
                x = 0.5 + ((double)rand() / (double)RAND_MAX * 1e-3 - 0.5e-3);
                if (x <=0.0) x = 0.01;
                if (x >=1.0) x = 0.99;
            }

            for (int i = 0; i < m_length_of_trajectory; ++i) {
                x = logistic_map(x, r);
            }
            // Using %.8Lf for r for reasonable precision in file, %.15f for x
            fprintf(fp, "%.6Lf,%.8f\n", r, x);
        }
    }
    fclose(fp);
    printf("Endpoints data written to %s\n", filename);
}


// --- Helper for sorting doubles (for finding distinct points in attractor) ---
int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

// --- Get Attractor Points and Count Distinct Ones ---
// r is long double for precision, x0 and attractor_points are double
int get_distinct_attractor_points(long double r, double x0,
                                  int burn_in, int num_to_collect,
                                  double *attractor_points_buffer, 
                                  double tolerance) {
    double x = x0;
    for (int i = 0; i < burn_in; ++i) {
        x = logistic_map(x, r);
    }

    for (int i = 0; i < num_to_collect; ++i) {
        x = logistic_map(x, r);
        attractor_points_buffer[i] = x;
    }

    qsort(attractor_points_buffer, num_to_collect, sizeof(double), compare_doubles);

    if (num_to_collect == 0) return 0;

    int distinct_count = 1;
    // Handle case where first point is NaN or Inf after burn-in if r is problematic
    if (isnan(attractor_points_buffer[0]) || isinf(attractor_points_buffer[0])) return 0;

    for (int i = 1; i < num_to_collect; ++i) {
        if (isnan(attractor_points_buffer[i]) || isinf(attractor_points_buffer[i])) break; // Stop counting if divergence
        if (fabsl((long double)attractor_points_buffer[i] - (long double)attractor_points_buffer[i - 1]) > (long double)tolerance) {
            distinct_count++;
        }
    }
    return distinct_count;
}

// --- Find Bifurcation r Value (using long double for r) ---
long double find_bifurcation_r(int current_period,
                               long double r_low_guess, long double r_high_guess,
                               double x0_search,
                               int burn_in_search, int collect_search_points,
                               double distinct_tol, int max_bisection_iter, long double r_precision) {
    long double low_r = r_low_guess;
    long double high_r = r_high_guess;
    long double mid_r;
    long double found_r = r_high_guess; // Initialize with a value

    double *attractor_points = (double *)malloc(collect_search_points * sizeof(double));
    if (!attractor_points) {
        perror("Failed to allocate memory for attractor points in find_bifurcation_r");
        return r_high_guess; // Or some error indicator
    }
    
    // Check initial bounds
    int low_points = get_distinct_attractor_points(low_r, x0_search, burn_in_search, collect_search_points, attractor_points, distinct_tol);
    int high_points = get_distinct_attractor_points(high_r, x0_search, burn_in_search, collect_search_points, attractor_points, distinct_tol);

    // We expect low_r to have 'current_period' points and high_r to have '2*current_period' points (or more).
    // If low_r already has more than 'current_period' points, it's too high.
    // If high_r has 'current_period' or fewer, it's too low.
    if (low_points > current_period && current_period > 0) { // current_period=0 case not handled here, assumes >0
         //printf("Warning: r_low_guess %.18Lf has %d points, expected %d. Adjusting search if possible or may fail.\n", low_r, low_points, current_period);
         // This is problematic. The bisection logic assumes low_r is in the current_period regime.
    }
    if (high_points <= current_period && current_period > 0) {
         //printf("Warning: r_high_guess %.18Lf has %d points, expected >%d. Adjusting search if possible or may fail.\n", high_r, high_points, current_period);
         // This is also problematic.
    }


    for (int i = 0; i < max_bisection_iter && fabsl(high_r - low_r) > r_precision; ++i) {
        mid_r = low_r + (high_r - low_r) / 2.0L;
        if (mid_r <= low_r || mid_r >= high_r) { // Interval too small or bisection stalled
            //printf("Bisection interval stalled or invalid for period %d. mid_r=%.18Lf, low_r=%.18Lf, high_r=%.18Lf\n", current_period, mid_r, low_r, high_r);
            break;
        }
        int num_distinct = get_distinct_attractor_points(mid_r, x0_search,
                                                         burn_in_search, collect_search_points,
                                                         attractor_points, distinct_tol);
        
        int target_next_period = 2 * current_period;
        
        // Refined logic:
        // If num_distinct <= current_period, mid_r is before bifurcation (or at it, if stable cycle)
        // If num_distinct > current_period (typically == target_next_period), mid_r is at or after bifurcation
        if (num_distinct > current_period) { // Bifurcation has occurred or is occurring at mid_r
             found_r = mid_r; // mid_r is a candidate for the bifurcation point or is past it
             high_r = mid_r;  // Try to find an even earlier r
        } else { // num_distinct <= current_period (bifurcation has not occurred yet at mid_r)
            low_r = mid_r;    // Need larger r
        }
    }
    free(attractor_points);
    // The loop terminates when high_r - low_r <= r_precision.
    // 'high_r' (or 'found_r' which tracks high_r) should be the best estimate from the high side.
    // A common return is (low_r + high_r) / 2.0L or high_r.
    return high_r; // high_r has been pushed down to the bifurcation point
}


// --- Polynomial Root Finding (f^k(x,r) - x = 0) using Newton-Raphson ---
// r_fixed is long double, x and root_val are double
static inline double g_func(double x, long double r_fixed, int k_period) {
    return iterate_map_k_times(x, r_fixed, k_period) - x;
}

static inline double g_func_derivative(double x, long double r_fixed, int k_period) {
    return derivative_iterate_map_k_times(x, r_fixed, k_period) - 1.0;
}

int newton_raphson_single_root(long double r_fixed, int k_period,
                               double initial_x_guess, double *root_val,
                               double tolerance, int max_iterations) {
    double x = initial_x_guess;
    for (int i = 0; i < max_iterations; ++i) {
        double gx = g_func(x, r_fixed, k_period);
        if (fabsl((long double)gx) < (long double)tolerance) {
            *root_val = x;
            return 1; 
        }
        double g_prime_x = g_func_derivative(x, r_fixed, k_period);
        if (fabsl((long double)g_prime_x) < 1e-12L) { // Avoid division by zero/small
            return 0; 
        }
        double delta_x = gx / g_prime_x;
        x -= delta_x;

        if (x < 0.0 || x > 1.0 || isnan(x) || isinf(x)) return 0; 

        if (fabsl((long double)delta_x) < (long double)tolerance) {
            *root_val = x;
            return 1; 
        }
    }
    return 0; 
}

void find_cycle_roots(long double r_fixed, int k_period,
                      double *found_roots, int *num_found_roots, int max_roots_to_find,
                      int num_initial_guesses, double nr_tolerance, int nr_max_iter) {
    *num_found_roots = 0;
    if (k_period == 0) return;

    for (int i = 0; i < num_initial_guesses && *num_found_roots < max_roots_to_find; ++i) {
        double initial_guess = (double)rand() / RAND_MAX; 
        double root;
        if (newton_raphson_single_root(r_fixed, k_period, initial_guess, &root, nr_tolerance, nr_max_iter)) {
            int already_found = 0;
            for (int j = 0; j < *num_found_roots; ++j) {
                if (fabs(root - found_roots[j]) < nr_tolerance * 10.0) { // Wider tol for uniqueness
                    already_found = 1;
                    break;
                }
            }
            if (!already_found) {
                found_roots[*num_found_roots] = root;
                (*num_found_roots)++;
                // qsort here can be slow if num_initial_guesses is huge and many roots are found early
                // Consider sorting at the end if performance is an issue.
                qsort(found_roots, *num_found_roots, sizeof(double), compare_doubles);
            }
        }
    }
}


// --- Main Program ---
int main() {
    srand((unsigned int)time(NULL)); 

    // Part 1: Generate Endpoints Data as requested
    printf("Part 1: Generating endpoints data for plotting...\n");
    generate_endpoints_data("endpoints_data.csv",
                            0.0L, 4.0L, 0.001L, // r range [0,4], step for dense plot
                            1000,              // n_trajectories_per_r
                            500,               // m_length_of_trajectory (effective burn-in)
                            1);                // use_random_x0 = true

    // Part 2: Find Bifurcation Points r_k and Estimate Feigenbaum Constant with Long Double
    printf("\nPart 2: Finding bifurcation points (using long double for r) and estimating Feigenbaum delta...\n");
    const int MAX_BIFURCATIONS_TO_FIND = 25; 
    long double r_bif[MAX_BIFURCATIONS_TO_FIND];
    long double deltas[MAX_BIFURCATIONS_TO_FIND - 2]; // Need at least 3 r_k for 1 delta

    double x0_fb_search = 0.50123456789; // More specific x0 to avoid symmetries if any
    int burn_in_fb_search = 20000;     // Increased burn-in
    int collect_fb_search = 4096;      // Should be > 2 * max_period_expected (e.g. 2^10=1024, so 2048+ is good)
    double distinct_tol_fb = 1e-8;     // Tighter tolerance for distinguishing x values
    int max_bisection_iter_fb = 200;   // More bisection iterations for r precision
    long double r_precision_fb = 1e-17L; // High precision for r

    // r_bif[i] is the r-value for the onset of the 2^(i+1) cycle.
    // i=0: 1->2 cycle (onset of 2^1 cycle)
    // i=1: 2->4 cycle (onset of 2^2 cycle)
    
    printf("Finding r_0 (1->2 cycles)...\n");
    r_bif[0] = find_bifurcation_r(1, 2.99L, 3.01L, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
    printf("r_0 (onset of 2-cycle): %.18Lf\n", r_bif[0]); // Should be 3.0L

    int actual_bifurcations_found = 0;
    if (r_bif[0] > 2.0L) { // Basic sanity check
        actual_bifurcations_found = 1;
    } else {
        printf("Failed to find r_0 reliably. Exiting Feigenbaum estimation.\n");
        goto skip_feigenbaum;
    }

    printf("Finding r_1 (2->4 cycles)...\n");
    // Start search for r_1 slightly above r_0 and well before known next one.
    r_bif[1] = find_bifurcation_r(2, r_bif[0] + 1e-5L, 3.45L, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
    printf("r_1 (onset of 4-cycle): %.18Lf\n", r_bif[1]); // Around 3.449...L
    if (r_bif[1] > r_bif[0]) {
        actual_bifurcations_found = 2;
    } else {
        printf("Failed to find r_1 reliably. Stopping Feigenbaum sequence here.\n");
    }
    
    long double feigenbaum_approx = 4.66920160910299067185320382046620161L; // Feigenbaum delta

    for (int k_idx = 2; k_idx < MAX_BIFURCATIONS_TO_FIND && actual_bifurcations_found == k_idx; ++k_idx) {
        int current_period_val = 1 << k_idx; // This is the period that is bifurcating (e.g., for k_idx=2, 2^2=4-cycle bifurcates)
        printf("Finding r_%d (%d->%d cycles)...\n", k_idx, current_period_val, 2 * current_period_val);
        
        long double r_low_search = r_bif[k_idx-1] + r_precision_fb * 1000.0L; // Start search just above previous bifurcation
        long double r_high_search_est;
        long double prev_diff = r_bif[k_idx-1] - r_bif[k_idx-2];

        if (prev_diff < r_precision_fb * 1000.0L) { 
             r_high_search_est = r_bif[k_idx-1] + ((k_idx < 4) ? 1e-3L : 1e-5L / (1L << (k_idx-4)));
        } else {
             r_high_search_est = r_bif[k_idx-1] + prev_diff / feigenbaum_approx; // Use known delta
        }
        
        long double accumulation_point_r_inf = 3.56994567186954202L; // More precise approx
        if (r_high_search_est <= r_low_search) {
             r_high_search_est = r_low_search + prev_diff / (feigenbaum_approx * 0.9L); // Widen a bit if est too small
             if (r_high_search_est <= r_low_search) r_high_search_est = r_low_search + 1e-6L / (1L << k_idx); // Ensure some interval
        }
        if (r_high_search_est > accumulation_point_r_inf) r_high_search_est = accumulation_point_r_inf - r_precision_fb*10.L;
        if (r_low_search >= r_high_search_est || r_low_search >= accumulation_point_r_inf - r_precision_fb*100.L) {
            printf("Search range for r_%d too small or too close to accumulation point. Stopping.\n", k_idx);
            break;
        }

        r_bif[k_idx] = find_bifurcation_r(current_period_val, r_low_search, r_high_search_est,
                                          x0_fb_search, burn_in_fb_search, collect_fb_search,
                                          distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
        printf("r_%d (onset of %d-cycle): %.18Lf\n", k_idx, 2*current_period_val, r_bif[k_idx]);
        
        if (fabsl(r_bif[k_idx] - r_bif[k_idx-1]) < r_precision_fb * 10.0L || r_bif[k_idx] <= r_bif[k_idx-1]) {
            printf("Bifurcation points r_%d and r_%d too close (diff %.18Le) or out of order. Stopping.\n", k_idx, k_idx-1, fabsl(r_bif[k_idx] - r_bif[k_idx-1]));
            break; 
        }
        actual_bifurcations_found++;
        if (r_bif[k_idx] >= accumulation_point_r_inf - r_precision_fb*1000.0L) {
             printf("Reached accumulation point vicinity (r_%d = %.18Lf). Stopping.\n", k_idx, r_bif[k_idx]);
             break;
        }
    }

skip_feigenbaum:
    printf("\nFeigenbaum Delta Estimates (based on %d found bifurcation points):\n", actual_bifurcations_found);
    if (actual_bifurcations_found >= 3) {
        for (int k_idx = 1; k_idx < actual_bifurcations_found - 1; ++k_idx) {
            long double r_km1 = r_bif[k_idx-1]; // onset of 2^k cycle
            long double r_k   = r_bif[k_idx];   // onset of 2^(k+1) cycle
            long double r_kp1 = r_bif[k_idx+1]; // onset of 2^(k+2) cycle

            if (fabsl(r_kp1 - r_k) < 1e-25L) { // Denominator too small
                printf("Delta_%d: Denominator (r_%d - r_%d) = %.18Le is too small. Cannot calculate.\n", k_idx, k_idx+1, k_idx, r_kp1 - r_k);
                continue;
            }
            if (r_k <= r_km1 + r_precision_fb || r_kp1 <= r_k + r_precision_fb) { // Check strictly increasing
                printf("Delta_%d: r values not strictly increasing (r[%d-1]=%.18Lf, r[%d]=%.18Lf, r[%d+1]=%.18Lf). Cannot calculate.\n", k_idx, k_idx, r_km1, k_idx, r_k, k_idx, r_kp1);
                continue;
            }

            deltas[k_idx-1] = (r_k - r_km1) / (r_kp1 - r_k);
            printf("Delta_%d = (r_%d - r_%d) / (r_%d - r_%d) = %.15Lf\n",
                   k_idx, k_idx, k_idx-1, k_idx+1, k_idx, deltas[k_idx-1]);
        }
    } else {
        printf("Not enough bifurcation points found to calculate any delta values.\n");
    }


    // Part 3: Finding roots of f^k(x,r) - x = 0 using Newton-Raphson...
    printf("\nPart 3: Finding roots of f^k(x,r) - x = 0...\n");
    long double r_test_ld = 3.5L; 
    int k_test_period = 4; // Look for points of the 4-cycle for f^4(x)-x=0

    // Max roots for f^P(x)-x=0 is 2^P. We are looking for points of a K-cycle.
    // If K is the true period, then f^K(x_i) = x_i for K points.
    // These K points are also roots of f^(nK)(x)-x=0 for integer n.
    // The number of points in an N-cycle is N.
    // So we look for k_test_period roots.
    int max_roots_to_find_nr = k_test_period * 2; // Find a bit more than expected if some are spurious or it's a higher period
    if (k_test_period > 0) {
         int max_poly_roots = 1;
         for(int i=0; i<k_test_period; ++i) { // 2^k_test_period
             if (__builtin_mul_overflow(max_poly_roots, 2, &max_poly_roots)) {
                 max_poly_roots = 1024; // Cap if overflow
                 break;
             }
         }
         if (max_poly_roots > 1024) max_poly_roots = 1024; // Cap for practical buffer size
         if (max_roots_to_find_nr > max_poly_roots) max_roots_to_find_nr = max_poly_roots;
    } else {
        max_roots_to_find_nr = 1;
    }


    double *roots_buffer = (double*)malloc(max_roots_to_find_nr * sizeof(double));
    int num_roots_found = 0;

    if(roots_buffer){
        find_cycle_roots(r_test_ld, k_test_period,
                         roots_buffer, &num_roots_found, max_roots_to_find_nr,
                         5000,   // Number of random initial guesses for Newton-Raphson
                         1e-10,  // Tolerance for Newton-Raphson convergence
                         200);   // Max iterations for Newton-Raphson

        printf("For r = %.6Lf, k_period = %d, found %d distinct roots of f^k_period(x)-x = 0 (expected up to %d for true %d-cycle):\n",
               r_test_ld, k_test_period, num_roots_found, k_test_period, k_test_period);
        for (int i = 0; i < num_roots_found; ++i) {
            printf("  x_%d = %.15f\n", i, roots_buffer[i]);
        }
        free(roots_buffer);
    } else {
        perror("Failed to allocate memory for roots_buffer");
    }
    
    printf("\nDone.\n");
    return 0;
}