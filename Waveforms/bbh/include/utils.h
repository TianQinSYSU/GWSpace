/* ===============================================
 * File Name: utils.h
 * Author: ekli
 * Mail: ekli_091@mail.dlut.edu.cn  
 * Created Time: 2022-03-29 09:46:47
 * ===============================================
 */

#include <stdio.h>
#include <complex.h>

#ifndef __UTILS__
#define __UTILS__
// complex value
//struct complex {
//    double real, imag;
//};
// len: length of array
#define len(array) (int)(sizeof(array) / sizeof(array[0]))

double deg2rad(double deg);
double rad2deg(double rad);
double sinc(double x);
double **GenerateMatrix(int N);

int Rotation_3D(double ** R, char *axis, double theta);
int QuadLagrange3(double *res, double *x, double *y);
double QuadLagrangeInterpolat(double x, double *res, int n);

int Factorial(int n);
int BinomialCoefficient(int n, int k);
double complex sYlm(int s, int l, int m, double theta, double phi);
int epsilon(int i, int j, int k);

#endif 
