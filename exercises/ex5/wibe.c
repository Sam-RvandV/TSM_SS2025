#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>

#define PREC 256  // MPFR precision in bits

// Generate bifurcation data
void generate_trajectories(double r_start, double r_end, int r_steps, int n, int m, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file");
        exit(1);
    }

    double dr = (r_end - r_start) / (r_steps - 1);
    unsigned int seed = 12345;

    for (int i = 0; i < r_steps; ++i) {
        double r = r_start + i * dr;
        for (int j = 0; j < n; ++j) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            for (int k = 0; k < m; ++k) {
                x = r * x * (1.0 - x);
            }
            fprintf(fp, "%.15f %.15f\n", r, x);
        }
    }
    fclose(fp);
}

int detect_period(mpfr_t r, int max_period) {
    mpfr_t x, tmp, one, tol;
    mpfr_inits2(PREC, x, tmp, one, tol, NULL);
    mpfr_set_d(x, 0.5, MPFR_RNDN);
    mpfr_set_d(one, 1.0, MPFR_RNDN);
    mpfr_set_d(tol, 1e-30, MPFR_RNDN);

    // Burn-in iterations
    for(int i = 0; i < 4000; i++) {
        mpfr_sub(tmp, one, x, MPFR_RNDN);
        mpfr_mul(tmp, tmp, x, MPFR_RNDN);
        mpfr_mul(x, r, tmp, MPFR_RNDN);
    }

    // Store trajectory
    mpfr_t *traj = malloc(max_period * sizeof(mpfr_t));
    for(int i = 0; i < max_period; i++) {
        mpfr_init2(traj[i], PREC);
        mpfr_set(traj[i], x, MPFR_RNDN);
        mpfr_sub(tmp, one, x, MPFR_RNDN);
        mpfr_mul(tmp, tmp, x, MPFR_RNDN);
        mpfr_mul(x, r, tmp, MPFR_RNDN);
    }

    // Period detection
    int period = -1;
    for(int p = 1; p <= max_period/4; p *= 2) {
        int valid = 1;
        for(int i = 0; i < p; i++) {
            mpfr_sub(tmp, traj[i], traj[i+p], MPFR_RNDN);
            if(mpfr_cmpabs(tmp, tol) > 0) {
                valid = 0;
                break;
            }
        }
        if(valid) period = p;
    }

    // Cleanup
    for(int i = 0; i < max_period; i++) mpfr_clear(traj[i]);
    free(traj);
    mpfr_clears(x, tmp, one, tol, NULL);
    return period;
}

void find_bifurcation(int target_period, mpfr_t r_low, mpfr_t r_high, mpfr_t result) {
    mpfr_t mid, diff;
    mpfr_inits2(PREC, mid, diff, NULL);
    const int max_iter = 200;

    for(int i = 0; i < max_iter; i++) {
        mpfr_add(mid, r_low, r_high, MPFR_RNDN);
        mpfr_div_ui(mid, mid, 2, MPFR_RNDN);
        
        int p = detect_period(mid, target_period * 4);
        if(p >= target_period) mpfr_set(r_low, mid, MPFR_RNDN);
        else mpfr_set(r_high, mid, MPFR_RNDN);
        
        mpfr_sub(diff, r_high, r_low, MPFR_RNDN);
        if(mpfr_cmp_d(diff, 1e-35) < 0) break;
    }
    mpfr_set(result, r_low, MPFR_RNDN);
    mpfr_clears(mid, diff, NULL);
}

void compute_bifurcations(int N, mpfr_t *results) {
    #pragma omp parallel for
    for(int k = 1; k <= N; k++) {
        int target = 1 << k;
        mpfr_t r_low, r_high;
        mpfr_inits2(PREC, r_low, r_high, NULL);

        // Known bifurcation intervals
        switch(k) {
            case 1: mpfr_set_d(r_low, 3.0, MPFR_RNDN);
                    mpfr_set_d(r_high, 3.6, MPFR_RNDN); break;
            case 2: mpfr_set_d(r_low, 3.4, MPFR_RNDN);
                    mpfr_set_d(r_high, 3.6, MPFR_RNDN); break;
            case 3: mpfr_set_d(r_low, 3.54, MPFR_RNDN);
                    mpfr_set_d(r_high, 3.57, MPFR_RNDN); break;
            case 4: mpfr_set_d(r_low, 3.56, MPFR_RNDN);
                    mpfr_set_d(r_high, 3.58, MPFR_RNDN); break;
            default: mpfr_sub(results[k-1], results[k-2], results[k-3], MPFR_RNDN);
                     mpfr_div_ui(results[k-1], results[k-1], 10, MPFR_RNDN);
                     mpfr_add(r_low, results[k-2], results[k-1], MPFR_RNDN);
                     mpfr_sub(r_high, results[k-2], results[k-1], MPFR_RNDN);
        }

        find_bifurcation(target, r_low, r_high, results[k-1]);
        mpfr_clears(r_low, r_high, NULL);
    }
}

int main() {
    generate_trajectories(2.5, 4.0, 1000, 5, 1000, "bifurcation_data.txt");

    const int N = 6;  // Reduced for stability
    mpfr_t results[N];
    
    // Initialize with known values
    for(int i = 0; i < N; i++) {
        mpfr_init2(results[i], PREC);
        mpfr_set_d(results[i], 3.0 + i*0.1, MPFR_RNDN);
    }

    compute_bifurcations(N, results);

    // Print results
    printf("Bifurcation Points:\n");
    for(int k = 0; k < N; k++) {
        printf("r_%d = ", k+1);
        mpfr_out_str(stdout, 10, 15, results[k], MPFR_RNDN);
        printf("\n");
    }

    printf("\nFeigenbaum Constants:\n");
    for(int k = 1; k < N-1; k++) {
        mpfr_t d1, d2, delta;
        mpfr_inits2(PREC, d1, d2, delta, NULL);
        
        mpfr_sub(d1, results[k], results[k-1], MPFR_RNDN);
        mpfr_sub(d2, results[k+1], results[k], MPFR_RNDN);
        if(mpfr_cmp_d(d2, 1e-15) > 0) {
            mpfr_div(delta, d1, d2, MPFR_RNDN);
            printf("δ_%d = ", k);
            mpfr_out_str(stdout, 10, 10, delta, MPFR_RNDN);
            printf("\n");
        }
        mpfr_clears(d1, d2, delta, NULL);
    }

    // Cleanup
    for(int i = 0; i < N; i++) mpfr_clear(results[i]);
    return 0;
}