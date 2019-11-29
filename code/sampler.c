#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>

// looks like the "cdf" is actually a "pdf"

int gaussian0()
{
    static const uint32_t dist[] = {
         6031371U, 13708371U, 13035518U,
         5186761U,  1487980U, 12270720U,
         3298653U,  4688887U,  5511555U,
         1551448U,  9247616U,  9467675U,
          539632U, 14076116U,  5909365U,
          138809U, 10836485U, 13263376U,
           26405U, 15335617U, 16601723U,
            3714U, 14514117U, 13240074U,
             386U,  8324059U,  3276722U,
              29U, 12376792U,  7821247U,
               1U, 11611789U,  3398254U,
               0U,  1194629U,  4532444U,
               0U,    37177U,  2973575U,
               0U,      855U, 10369757U,
               0U,       14U,  9441597U,
               0U,        0U,  3075302U,
               0U,        0U,    28626U,
               0U,        0U,      197U,
               0U,        0U,        1U
    };

    uint32_t v0, v1, v2;
    size_t u;
    int z;

/*
    for (int i = 0; i < 3 * 19; i += 3) {
        printf("%06X%06X%06X\n",
            dist[i], dist[i + 1], dist[i + 2]);
    }
*/

    /*
     * Get a random 72-bit value, into three 24-bit limbs v0..v2.
     */
    v0 = lrand48() & 0xFFFFFF;
    v1 = lrand48() & 0xFFFFFF;
    v2 = lrand48() & 0xFFFFFF;

    /*
     * Sampled value is z, such that v0..v2 is lower than the first
     * z elements of the table.
     */
    z = 0;
    for (u = 0; u < (sizeof dist) / sizeof(dist[0]); u += 3) {
        uint32_t w0, w1, w2, cc;

        w0 = dist[u + 2];
        w1 = dist[u + 1];
        w2 = dist[u + 0];
        cc = (v0 - w0) >> 31;
        cc = (v1 - w1 - cc) >> 31;
        cc = (v2 - w2 - cc) >> 31;
        z += (int)cc;
    }
    return z;
}



int main()
{
	FILE *f = fopen("samples.txt", "a");

    int b, z0, z, i;
	int sample_size=1000000;

    srand48(time(NULL));

    for (i=1;i<=sample_size;i++) {
		
		i=i+1;
        z0 = gaussian0();
        b = lrand48() & 1;
        z = b + ((b << 1) - 1) * z0;

        printf("%d\n", z);
        fprintf(f,"%d\n", z);
    }	
	fclose(f);

    return 0;
}
