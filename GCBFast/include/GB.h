#ifndef GB_h
#define GB_h


struct GB
{
	double T;			// observation period
	double f0;			// initial GW freq
	double theta, phi;  // sky-location (spherical-polar)
	double amp, iota;   // amplitude and inclination angle
	double psi, phi0;   // polarization angle, initial phase

	double cosiota;
	double costheta;

	double *params;		// vector to store parameters

	long q, N;  		// carrier freq bin, number of samples

	long NP;			// number of parameters
};

struct Waveform
{
	long N;
	long q; // Fgw carrier bin

	int NP;		// number of parameters in signal

	double T; 		// observation period

	double *params;

	double **eplus, **ecross;
	double **dplus, **dcross;

	double DPr, DPi, DCr, DCi;

	// direction vector of GW
	double *k;

	// separation unit vectors between S/C
	double *r12, *r21;
	double *r13, *r31;
	double *r23, *r32;

	double **kdotr;
	double *kdotx;

	double *xi, *f, *fonfs;

	// Time series of slowly evolving terms at each vertex
	// dataij corresponds to fractional arm length difference yij
	double *data12, *data21;
	double *data13, *data31;
	double *data23, *data32;

	// Fourier coefficients of slowly evolving terms (numerical)
	double *a12, *a21;
	double *a13, *a31;
	double *a23, *a32;

	// S/C position
	double *x, *y, *z;

	// Time varrying quantities (Re & Im) broken up into convenient segments
	double **TR, **TI;

	//Package cij's into proper form for TDI subroutines
	double ***d;
};



void Fast_GB(double *params, long N, double Tobs, double dt,  double *XLS, double *YLS, double *ZLS, double* XSL, double* YSL, double* ZSL, int NP);

void XYZ(double ***d, double f0, long q, long M, double dt, double Tobs, double *XLS, double *YLS, double *ZLS, double* XSL, double* YSL, double* ZSL);

void get_basis_vecs(double *params, double *u, double *v, double *k);
void get_basis_tensors(struct Waveform *wfm);
void calc_sep_vecs(struct Waveform *wfm);
void calc_d_matrices(struct Waveform *wfm);
void calc_kdotr(struct Waveform *wfm);

void get_transfer(struct Waveform *wfm, double t);
void set_const_trans(struct Waveform *wfm);
void calc_xi_f(struct Waveform *wfm, double t);

void copy_params(struct Waveform *wfm, double *params);
void alloc_waveform(struct Waveform *wfm);
void free_waveform(struct Waveform *wfm);

void fft_data(struct Waveform *wfm);
void unpack_data(struct Waveform *wfm);
void fill_time_series(struct Waveform *wfm, int n);

long get_N(double *params, double Tobs);

#endif /* GB_h */
