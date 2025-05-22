#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // For memcpy
#include <time.h>   // For seeding random

#define PI 3.14159265358979323846

// --- Core Logistic Map ---
static inline double logistic_map(double x, double r) {
    return r * x * (1.0 - x);
}

// --- Iterated Map and its Derivative ---
// Computes f^k(x, r)
double iterate_map_k_times(double x, double r, int k) {
    for (int i = 0; i < k; ++i) {
        x = logistic_map(x, r);
    }
    return x;
}

// Computes derivative d(f^k(x,r))/dx
// f'(x) = r*(1-2x)
// (f^k(x))' = f'(f^{k-1}(x)) * f'(f^{k-2}(x)) * ... * f'(x)
double derivative_iterate_map_k_times(double x_start, double r, int k) {
    if (k == 0) return 1.0; // d(x)/dx = 1

    double derivative_val = 1.0;
    double x_current = x_start;
    double x_iterates[k]; // Store intermediate x values

    for (int i = 0; i < k; ++i) {
        x_iterates[i] = x_current;
        x_current = logistic_map(x_current, r);
    }

    // Apply chain rule in reverse for numerical stability perhaps, or forward
    // f'(f^{k-1}(x)) * f'(f^{k-2}(x)) * ... * f'(x)
    for (int i = 0; i < k; ++i) {
        derivative_val *= r * (1.0 - 2.0 * x_iterates[i]);
    }
    return derivative_val;
}


// --- Bifurcation Diagram Data Generation ---
void generate_bifurcation_data(const char *filename,
                               double r_min, double r_max, double r_step,
                               double initial_x,
                               int n_trajectories_per_r, // Usually 1 for diagram
                               int m_burn_in, int m_collect) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open bifurcation data file");
        return;
    }
    fprintf(fp, "r,x\n"); // CSV header

    for (double r = r_min; r <= r_max; r += r_step) {
        for (int traj = 0; traj < n_trajectories_per_r; ++traj) {
            // Use slightly different x0 for multiple trajectories if desired,
            // but for bifurcation diagram, one good x0 is fine.
            double x = initial_x + traj * 1e-5; // Tiny offset if multiple
            if (x >=1.0) x = 0.99; // ensure x is in (0,1)
            if (x <=0.0) x = 0.01;

            // Burn-in iterations
            for (int i = 0; i < m_burn_in; ++i) {
                x = logistic_map(x, r);
            }
            // Collect and write points
            for (int i = 0; i < m_collect; ++i) {
                x = logistic_map(x, r);
                fprintf(fp, "%.6f,%.15f\n", r, x);
            }
        }
    }
    fclose(fp);
    printf("Bifurcation data written to %s\n", filename);
}

// --- Helper for sorting doubles (for finding distinct points) ---
int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

// --- Get Attractor Points and Count Distinct Ones ---
// Returns number of distinct points
int get_distinct_attractor_points(double r, double x0,
                                  int burn_in, int num_to_collect,
                                  double *attractor_points_buffer, // Pre-allocated buffer
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
    for (int i = 1; i < num_to_collect; ++i) {
        if (fabs(attractor_points_buffer[i] - attractor_points_buffer[i - 1]) > tolerance) {
            distinct_count++;
        }
    }
    return distinct_count;
}

// --- Find Bifurcation r Value ---
// Finds r where cycle period doubles from 'current_period'
double find_bifurcation_r(int current_period,
                          double r_low_guess, double r_high_guess,
                          double x0_search,
                          int burn_in_search, int collect_search_points,
                          double distinct_tol, int max_bisection_iter, double r_precision) {
    double low_r = r_low_guess;
    double high_r = r_high_guess;
    double mid_r = low_r;
    double found_r = r_high_guess; // Default to high if not found better

    // Pre-allocate buffer for attractor points
    double *attractor_points = (double *)malloc(collect_search_points * sizeof(double));
    if (!attractor_points) {
        perror("Failed to allocate memory for attractor points");
        return found_r; // Or some error code
    }

    for (int i = 0; i < max_bisection_iter && (high_r - low_r) > r_precision; ++i) {
        mid_r = (low_r + high_r) / 2.0;
        int num_distinct = get_distinct_attractor_points(mid_r, x0_search,
                                                         burn_in_search, collect_search_points,
                                                         attractor_points, distinct_tol);

        // We are looking for the point where it *just* bifurcates to 2 * current_period
        // So, if num_distinct > current_period, it means bifurcation might have occurred or is occurring
        // We want the smallest r for which this is true.
        if (num_distinct > current_period && (current_period == 0 || num_distinct <= 2 * current_period) ) { // current_period=0 for 0->1 cycle is special
             found_r = mid_r; // This r is a candidate
             high_r = mid_r;   // Try to find an even earlier r
        } else if (num_distinct > 2 * current_period && current_period > 0) { // Overshot, too many points
             found_r = mid_r;
             high_r = mid_r;
        }
        else { // num_distinct <= current_period
            low_r = mid_r;    // Bifurcation hasn't happened yet, need larger r
        }
    }
    free(attractor_points);
    return found_r; // Or (low_r + high_r)/2.0 for best estimate
}


// --- Polynomial Root Finding (f^k(x,r) - x = 0) using Newton-Raphson ---

// g(x) = f^k(x,r) - x
static inline double g_func(double x, double r, int k) {
    return iterate_map_k_times(x, r, k) - x;
}

// g'(x) = d(f^k(x,r))/dx - 1
static inline double g_func_derivative(double x, double r, int k) {
    return derivative_iterate_map_k_times(x, r, k) - 1.0;
}

// Newton-Raphson to find one root
// Returns 1 if root found, 0 otherwise. Root is stored in *root_val.
int newton_raphson_single_root(double r_fixed, int k_period,
                               double initial_x_guess, double *root_val,
                               double tolerance, int max_iterations) {
    double x = initial_x_guess;
    for (int i = 0; i < max_iterations; ++i) {
        double gx = g_func(x, r_fixed, k_period);
        if (fabs(gx) < tolerance) {
            *root_val = x;
            return 1; // Converged
        }
        double g_prime_x = g_func_derivative(x, r_fixed, k_period);
        if (fabs(g_prime_x) < 1e-12) { // Avoid division by zero or near-zero
            return 0; // Derivative too small, won't converge well
        }
        double delta_x = gx / g_prime_x;
        x -= delta_x;

        if (x < 0.0 || x > 1.0) return 0; // Escaped valid range

        if (fabs(delta_x) < tolerance) {
            *root_val = x;
            return 1; // Converged
        }
    }
    return 0; // Max iterations reached
}

// Find multiple roots of f^k(x,r) - x = 0 by trying various initial guesses
void find_cycle_roots(double r_fixed, int k_period,
                      double *found_roots, int *num_found_roots, int max_roots,
                      int num_initial_guesses, double nr_tolerance, int nr_max_iter) {
    *num_found_roots = 0;
    if (k_period == 0) return; // f^0(x)-x = 0 has no solution in typical sense for cycles

    for (int i = 0; i < num_initial_guesses; ++i) {
        double initial_guess = (double)rand() / RAND_MAX; // Random x0 in [0,1)
        double root;
        if (newton_raphson_single_root(r_fixed, k_period, initial_guess, &root, nr_tolerance, nr_max_iter)) {
            // Check if this root (or very close) is already found
            int already_found = 0;
            for (int j = 0; j < *num_found_roots; ++j) {
                if (fabs(root - found_roots[j]) < nr_tolerance * 10.0) { // Wider tolerance for uniqueness
                    already_found = 1;
                    break;
                }
            }
            if (!already_found && *num_found_roots < max_roots) {
                found_roots[*num_found_roots] = root;
                (*num_found_roots)++;
                // Sort to make it easier to find unique ones next time (optional, but good)
                qsort(found_roots, *num_found_roots, sizeof(double), compare_doubles);
            }
        }
    }
}


// --- Main Program ---
int main() {
    srand(time(NULL)); // Seed random number generator

    // Part 1: Generate Bifurcation Diagram Data
    printf("Part 1: Generating bifurcation diagram data...\n");
    generate_bifurcation_data("bifurcation_data.csv",
                              2.8, 4.0, 0.001,  // r range and step
                              0.5,               // initial x
                              1,                 // trajectories per r
                              1000,              // burn-in iterations
                              200);              // points to collect per r

    // Part 2: Find Bifurcation Points r_k and Estimate Feigenbaum Constant
    printf("\nPart 2: Finding bifurcation points and estimating Feigenbaum delta...\n");
    const int MAX_BIFURCATIONS = 7; // Up to 2^7 = 128-cycle
    double r_bif[MAX_BIFURCATIONS];
    double deltas[MAX_BIFURCATIONS - 2];

    // Parameters for find_bifurcation_r
    double x0_fb_search = 0.5;
    int burn_in_fb_search = 5000; // More for stability
    int collect_fb_search = 512;  // Should be > 2 * max_period_expected
    double distinct_tol_fb = 1e-5;
    int max_bisection_iter_fb = 100;
    double r_precision_fb = 1e-9;

    // Known approximate ranges to speed up search
    // r_k is where 2^k cycle becomes 2^{k+1} cycle
    // r_0: 1->2 cycle (period_doubling_from = 1)
    // r_1: 2->4 cycle (period_doubling_from = 2)
    // etc.

    // Bifurcation 1 -> 2 (k=0, current_period=1)
    r_bif[0] = find_bifurcation_r(1, 2.9, 3.1, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
    printf("r_0 (1->2 cycles): %.12f\n", r_bif[0]);

    // Bifurcation 2 -> 4 (k=1, current_period=2)
    r_bif[1] = find_bifurcation_r(2, r_bif[0], 3.5, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
    printf("r_1 (2->4 cycles): %.12f\n", r_bif[1]);

    // Bifurcation 4 -> 8 (k=2, current_period=4)
    r_bif[2] = find_bifurcation_r(4, r_bif[1], 3.56, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
    printf("r_2 (4->8 cycles): %.12f\n", r_bif[2]);

    // Subsequent bifurcations
    int current_period = 4;
    for (int k = 3; k < MAX_BIFURCATIONS; ++k) {
        current_period *= 2;
        // Estimate next r_high_guess based on previous delta, or use a safe upper bound
        double r_low = r_bif[k-1];
        double r_high_est = r_bif[k-1] + (r_bif[k-1] - r_bif[k-2]) / 4.6; // Rough est.
        if (r_high_est > 3.5699456) r_high_est = 3.5699456; // Accumulation point
        if (r_high_est <= r_low) r_high_est = r_low + 0.01; // Ensure range is valid

        r_bif[k] = find_bifurcation_r(current_period, r_low, r_high_est, x0_fb_search, burn_in_fb_search, collect_fb_search, distinct_tol_fb, max_bisection_iter_fb, r_precision_fb);
        printf("r_%d (%d->%d cycles): %.12f\n", k, current_period, 2*current_period, r_bif[k]);
        if (fabs(r_bif[k] - r_bif[k-1]) < r_precision_fb * 10) { // Converged or stuck
            printf("Bifurcation points too close, stopping.\n");
            break;
        }
    }

    printf("\nFeigenbaum Delta Estimates:\n");
    for (int k = 1; k < MAX_BIFURCATIONS - 1; ++k) {
        if (r_bif[k+1] <= r_bif[k] || r_bif[k] <= r_bif[k-1]) { // Check for valid, distinct r values
             printf("Delta_%d: r values not distinct or ordered, cannot calculate.\n", k);
             continue;
        }
        double r_km1 = r_bif[k-1];
        double r_k   = r_bif[k];
        double r_kp1 = r_bif[k+1];
        if (fabs(r_kp1 - r_k) < 1e-12) { // Denominator too small
            printf("Delta_%d: (r_%d - r_%d) / (r_%d - r_%d) denominator too small.\n", k, k, k-1, k+1, k);
            continue;
        }
        deltas[k-1] = (r_k - r_km1) / (r_kp1 - r_k);
        printf("Delta_%d = (r_%d - r_%d) / (r_%d - r_%d) = %.8f\n",
               k, k, k-1, k+1, k, deltas[k-1]);
    }

    // Part 3: Finding roots of f^k(x,r) - x = 0 for a specific r and k
    printf("\nPart 3: Finding roots of f^k(x,r) - x = 0 using Newton-Raphson...\n");
    double r_test = 3.5; // Example r value (in 4-cycle region)
    int k_test = 4;      // Look for points of the 4-cycle

    // Max possible roots for f^k(x)-x=0 is 2^k.
    // However, Newton finds one at a time, and we use random starts.
    int max_expected_roots = 1 << k_test;
    if (max_expected_roots > 256) max_expected_roots = 256; // Cap for sanity
    double *roots_buffer = (double*)malloc(max_expected_roots * sizeof(double));
    int num_roots_found = 0;

    if(roots_buffer){
        find_cycle_roots(r_test, k_test,
                         roots_buffer, &num_roots_found, max_expected_roots,
                         1000,   // Number of random initial guesses for Newton-Raphson
                         1e-9,  // Tolerance for Newton-Raphson convergence
                         100);   // Max iterations for Newton-Raphson

        printf("For r = %.4f, k = %d, found %d distinct roots of f^k(x)-x = 0:\n",
               r_test, k_test, num_roots_found);
        for (int i = 0; i < num_roots_found; ++i) {
            printf("  x_%d = %.15f\n", i, roots_buffer[i]);
        }
        free(roots_buffer);
    } else {
        perror("Failed to allocate memory for roots_buffer");
    }
    
    // Example 2: r in 2-cycle region
    r_test = 3.2; 
    k_test = 2; // Look for points of the 2-cycle
    max_expected_roots = 1 << k_test;
    roots_buffer = (double*)malloc(max_expected_roots * sizeof(double));
    num_roots_found = 0;
    if(roots_buffer){
        find_cycle_roots(r_test, k_test,
                         roots_buffer, &num_roots_found, max_expected_roots,
                         500, 1e-9, 100);
        printf("For r = %.4f, k = %d, found %d distinct roots of f^k(x)-x = 0:\n",
               r_test, k_test, num_roots_found);
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