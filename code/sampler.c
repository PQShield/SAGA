#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

/*
 * Sample z0 from the half-Gaussian with sigma_0 = 1.8205 via CDT inversion.
 *
 * The table below is the ascending CDT: row k holds round(2^72 * P(z0 <= k)),
 * stored as three 24-bit limbs (most significant limb first within the row).
 * It is identical to `halfgaussian_cdt` in sampler.py.
 *
 * CDT inversion draws a uniform 72-bit value v and returns
 *     z0 = #{ k : v >= cdt[k] },
 * i.e. the number of thresholds v meets or exceeds. This matches the Python
 * reference `sampler0()`:  z0 += (r >= elt).
 *
 * NOTE: this is a CDT with a ">=" count, NOT Falcon's reverse CDT (RCDT).
 * The RCDT is descending and pairs with a "v < row" count. Mixing an
 * ascending table with the "v < row" convention samples the mirror image
 * (mass at |z| ~ 18 instead of ~ 0), so the count direction below matters.
 */

int gaussian0()
{
    static const uint32_t cdt[] = {
         6031371U, 13708371U, 13035518U,
        11218132U, 15196352U,  8529022U,
        14516786U,  3108023U, 14040577U,
        16068234U, 12355640U,  6731036U,
        16607867U,  9654540U, 12640401U,
        16746677U,  3713810U,  9126561U,
        16773083U,  2272212U,  8951068U,
        16776798U,     9114U,  5413926U,
        16777184U,  8333173U,  8690648U,
        16777214U,  3932749U, 16511895U,
        16777215U, 15544539U,  3132933U,
        16777215U, 16739168U,  7665377U,
        16777215U, 16776345U, 10638952U,
        16777215U, 16777201U,  4231493U,
        16777215U, 16777215U, 13673090U,
        16777215U, 16777215U, 16748392U,
        16777215U, 16777215U, 16777018U,
        16777215U, 16777215U, 16777215U
    };

    uint32_t v0, v1, v2;
    size_t u;
    int z;

    v0 = lrand48() & 0xFFFFFF;
    v1 = lrand48() & 0xFFFFFF;
    v2 = lrand48() & 0xFFFFFF;

    z = 0;
    for (u = 0; u < (sizeof cdt) / sizeof(cdt[0]); u += 3) {
        uint32_t w0, w1, w2, cc;

        w0 = cdt[u + 2];
        w1 = cdt[u + 1];
        w2 = cdt[u + 0];
        /* Borrow chain: cc = 1 iff v < row (unsigned 72-bit compare). */
        cc = (v0 - w0) >> 31;
        cc = (v1 - w1 - cc) >> 31;
        cc = (v2 - w2 - cc) >> 31;
        /* Count v >= row, matching the Python reference (r >= elt). */
        z += (int)(1 - cc);
    }
    return z;
}


int main(int argc, char **argv)
{
    int b, z0, z;
    /* Sample count: argv[1] if given, else 1,000,000. Output goes to
     * stdout; tests capture it and chi2-test against the folded PDT. */
    int sample_size = (argc > 1) ? atoi(argv[1]) : 1000000;

    srand48(time(NULL));

    for (int i = 0; i < sample_size; i++) {
        z0 = gaussian0();
        b = lrand48() & 1;
        z = b + ((b << 1) - 1) * z0;

        printf("%d\n", z);
    }

    return 0;
}
