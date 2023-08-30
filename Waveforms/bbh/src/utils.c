/* ===============================================
 * File Name: utils.c
 * Author: ekli
 * Mail: ekli_091@mail.dlut.edu.cn  
 * Created Time: 2022-03-26 14:58:58
 * ===============================================
 */

#include <string.h>
#include <math.h>
#include "constant.h"
#include "utils.h"

// Convert degree to radian
double deg2rad(double deg)
{
    return deg/180.0 * _pi_;
}


// Convert radian to degree
double rad2deg(double rad)
{
    return rad/_pi_ * 180;
}

// Function of sinc
double sinc(double x)
{
    if (x==0) return 1;
    return sin(x)/x;
}

// Rotation around the axis of theta
int Rotation_3D(double **R, char *axis, double theta)
{
    int i, j;

    for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
            if (i==j) {
                R[i][j] = 1.0;
            }else {
                R[i][j] = 0.0;
            }
        }
    }

    if (strcmp(axis,"y") == 0) {
        R[0][0] = cos(theta);
        R[0][2] = -sin(theta);
        R[2][0] = sin(theta);
        R[2][2] = cos(theta);
    }else if(strcmp(axis,"x") == 0) {
        R[1][1] = cos(theta);
        R[1][2] = sin(theta);
        R[2][1] = -sin(theta);
        R[2][2] = cos(theta);
    }else if(strcmp(axis,"z") == 0) {
        R[0][0] = cos(theta);
        R[0][1] = sin(theta);
        R[1][0] = -sin(theta);
        R[1][1] = cos(theta);
    }else {
        printf("Out of range, axis should be x, y, or z\n");
        exit(1);
    }    

    return 0;
}

// Generate a matrix
double ** GenerateMatrix(int N)
{
    int i,j;
    double **mat;
    mat = (double **)malloc(N*sizeof(double*));
    for (i=0;i<N;i++){
        mat[i] = (double*)malloc(N*sizeof(double));
        for (j=0;j<N;j++){
            mat[i][j] = 0.0;
        }
    }
    return mat;
}

/* dfridr(func, x, h, err=1e-14, *args):
    '''
    Parameters:
        func: external function
        x: point or array
        h: initial stepsize
        err: error
    -------------------------------------------------------------------------
    Returns the derivative of a function `func` at a point `x` by Ridders' method
    of polynomial extrapolation. The value `h` is input as an estimated initial
    stepsize; It need not be small, but ratther should be an increment in `x` over 
    which func changes substantially. An estimate of the error in the derivative 
    is returned as err.
    Parameters: Stepsize is decreased by `CON` at each iteration. Max size of 
        tableau is set by `NTAB`. Return when error is `SAFE` worse than the best
        so far.
*/
// double dfridr(double (func)(double *))
//     CON  = 1.4
//     CON2 = CON*CON
//     BIG  = 1e30
//     NTAB = 10
//     SAFE = 2.0
//     
//     a = np.zeros((NTAB, NTAB))
//     if (h == 0):
//         raise ValueError('h must be nonzero in dfridr')
//         #sys.exit(0)
//     hh = h
//     a[0,0] = (func(x+hh, *args) - func(x-hh))/(2.0*hh)
//     err = BIG
//     for i in range(1,NTAB):
//         hh = hh/CON
//         a[0,i] = (func(x+hh, *args) - func(x-hh))/(2.0*hh)
//         fac = CON2
//         for j in range(1,i):
//             a[j,i] = (a[j-1,i] * fac - a[j-1,i-1])/(fac -1)
//             fac = CON2 * fac
//             errt = max(np.abs(a[j,-i] - a[j-1,i]), np.abs(a[j,i] - a[j-1,i-1]))
//             if(errt <= err):
//                 err = errt
//                 df = a[j,i]
//         if(np.abs(a[i,i] - a[i-1,i-1]) >= SAFE * err):
//             return df
//     return df

/* QuadLagrange3(x, y):
    Quadratic Lagrange interpolarion polynomial of degree 2
    -------------------------------------------------------
    Parameters:
        x: length of 3
        y: length of 3
    Return:
        res: array with length of 3
    -------------------------------------------
    Reference:
        https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html
        OR
        http://mathonline.wikidot.com/quadratic-lagrange-interpolating-polynomials
*/
int QuadLagrange3(double *res, double* x, double *y)
{
    // double res[3] = {0.0, 0.0, 0.0};
    double c0, c1, c2;

    c0 = y[0] / ( (x[0]-x[1]) * (x[0]-x[2]) );
    c1 = y[1] / ( (x[1]-x[0]) * (x[1]-x[2]) );
    c2 = y[2] / ( (x[2]-x[0]) * (x[2]-x[1]) );
    res[0] = c0 * x[1] * x[2] + c1 * x[2] * x[0] + c2 * x[0] * x[1];
    res[1] =-c0 * (x[1] + x[2]) - c1 * (x[2] + x[0]) - c2 * (x[0] + x[1]);
    res[2] = c0 + c1 + c2;
    return 0;
}
// Quadratic Lagrange Interpolating Polynomials.
double QuadLagrangeInterpolat(double x, double *res, int n)
{
    double ss = 0;
    int i;

    for (i=0; i< n; i++) {
        printf(" %6.2f x^%d +", res[i], i);
        ss += res[i] * pow(x, i);
    }
    printf(" = \n");
    return ss;
}

// factorial
//    Refs:
//    https://mathworld.wolfram.com/BinomialCoefficient.html
int Factorial(int n)
{
    if (n == 0) {
        return 1;
    }
    return n * Factorial(n-1);
}

/* Binominal coefficient
    $$
    \binom{n}{k} = 
        \begin{cases}
            \frac{n!}{k! (n-k)!} & \text{ for } 0 \leq k < n \\
            0 & \text{otherwise}
        \end{cases}
    $$
    ---------------------------------------------------------
    Refs:
    https://mathworld.wolfram.com/BinomialCoefficient.html
*/
int BinomialCoefficient(int n, int k)
{
    if ( k >= 0 && k <= n) {
        return Factorial(n) / Factorial(k) /Factorial(n-k);
    }else{
        //printf("%d is not less than %d, or %d is smaller than 0\n", k, n, k);
        return 0;
    }
    //return 0;
}

// Spin Weighted Spherical Harmoics
//    the spin-weight
//    {}_s Y_{lm}(\theta, \phi)
//    -->
//    Parameters:
//        s: spin
//        l:
//        m:
//        theta:
//        phi:

double complex sYlm(int s, int l, int m, double theta, double phi)
{
    double tp, tps;
    int k;
    double complex Y; // = 0 + 0 * I;
    //Y = (complex *)malloc(sizeof(complex));

    tp = pow(-1.0, m);
    tp *= sqrt( Factorial(l+m) * Factorial(l-m) * (2*l+1)/(4*_pi_) /Factorial(l+s) / Factorial(l-s) );
    tp *= pow( sin(0.5*theta), 2.0 * l);
    
    double ftps(int k)
    {
        return BinomialCoefficient(l-s, k) * BinomialCoefficient(l+s, k+s-m) * pow(-1., l-k-s);
    }

    tps = 0;
    for (k=0; k <= l-s; k++) {
        tps += ftps(k) * pow(tan(0.5*theta), -2*k -s +m);
    }
    Y = tp * tps * cos(m * phi) + I * tp * tps * sin(m * phi);
    return Y;
}

/* ##+====================================
    epsilon tensor or the permutation symbol or the Levi-Civita symbol
    -------------------------------------------------------------------
    Parameters:
        i,j,k: three int values

    Return:
        epsilon_ijk =  0 if any two labels are the same
                    =  1 if i,j,k is an even permutation of 1,2,3
                    = -1 if i,j,k is an odd permutation of 1,2,3
    Refs:
    https://mathworld.wolfram.com/PermutationSymbol.html
*/
int epsilon(int i, int j, int k)
{
    int ss;
    int sp;

    if (i>3 || j>3 || k>3 || i==j || j==k || k==i) {
        printf("Wrong numbe of i j k, at least two of them are the same");
        exit(0);
    }
    
    ss = 0;
    if (j < i) ss += 1;
    if (k < j) ss += 1;
    if (k < i) ss += 1;

    sp = ss%2;
    if (sp == 0) return 1;
    return -1;
}

