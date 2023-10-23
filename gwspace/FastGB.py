import numpy as np

from gwspace.constants import YRSID_SI
from gwspace.Waveform import GCBWaveform

if __package__ or "." in __name__:
    from gwspace import libFastGB
else:
    import libFastGB


class FastGB(GCBWaveform):
    """ Calculate the GCB waveform using fast/slow """

    def buffer_size(self, oversample=1):
        YEAR = YRSID_SI
        mult = 8
        if (self.T_obs/YEAR) <= 8.0: mult = 8
        if (self.T_obs/YEAR) <= 4.0: mult = 4
        if (self.T_obs/YEAR) <= 2.0: mult = 2
        if (self.T_obs/YEAR) <= 1.0: mult = 1
        N = 32*mult
        if self.f0 > 0.001: N = 64*mult
        if self.f0 > 0.01:  N = 256*mult
        if self.f0 > 0.03:  N = 512*mult
        if self.f0 > 0.1:   N = 1024*mult

        return N*oversample

    def get_fastgb_fd_single(self, simulator='synthlisa', buffer=None, dt=15., oversample=1, detector='TianQin'):
        # FIXME: assume T=T_obs below
        N = self.buffer_size(oversample)

        XLS = np.zeros(2*N, 'd')
        YLS = np.zeros(2*N, 'd')
        ZLS = np.zeros(2*N, 'd')

        XSL = np.zeros(2*N, 'd')
        YSL = np.zeros(2*N, 'd')
        ZSL = np.zeros(2*N, 'd')

        params = np.array([self.f0, self.fdot, self.Beta, self.Lambda, self.amp, self.iota, self.psi, self.phi0])

        if np.all(params) is not None:
            # vector must be ordered as required by Fast_GB
            # Fast_GB(double *params, long N, double *XLS, double *ALS, double *ELS, int NP)

            libFastGB.ComputeXYZ_FD(params, N, self.T_obs, dt, XLS, YLS, ZLS, XSL, YSL, ZSL, len(params), detector=detector)
            # TODO Need to transform to SL if required
            Xf = XLS
            Yf = YLS
            Zf = ZLS
            if simulator == 'synthlisa':
                Xf = XSL
                Yf = YSL
                Zf = ZSL
        else:
            raise ValueError

        f0 = self.f0
        if buffer is None:
            retX, retY, retZ = [a[::2]+1.j*a[1::2] for a in (Xf, Yf, Zf)]
            kmin = int(f0*self.T_obs)-N/2
            df = 1.0/self.T_obs
            f_range = np.linspace(kmin*df, (kmin+len(retX)-1)*df, len(retX))
            return f_range, retX, retY, retZ
        else:
            kmin, blen, alen = 0., len(buffer[0]), 2*N

            beg, end = int(int(f0*self.T_obs)-N/2), int(f0*self.T_obs+N/2)  # for a full buffer, "a" begins and ends at these indices
            begb, bega = (beg-kmin, 0) if beg >= kmin else (
                0, 2*(kmin-beg))  # left-side alignment of partial buffer with "a"
            endb, enda = (end-kmin, alen) if end-kmin <= blen else (blen, alen-2*(end-kmin-blen))
            # the corresponding part of "a" that should be assigned to the partial buffer
            # ...remember "a" is doubled up
            # check: if kmin = 0, then begb = beg, endb = end, bega = 0, enda = alen
            bega = int(bega)
            begb = int(begb)
            enda = int(enda)
            endb = int(endb)
            for i, a in enumerate((Xf, Yf, Zf)):
                buffer[i][begb:endb] += a[bega:enda:2]+1j*a[(bega+1):enda:2]

    def get_fastgb_fd(self, simulator='synthlisa', dt=15., oversample=1):
        length = int(0.5*self.T_obs/dt)+1  # was "NFFT = int(T/dt)", and "NFFT/2+1" passed to numpy.zeros
        buffer = tuple(np.zeros(length, dtype=np.complex128) for _ in range(3))

        # for _ in table:
        self.get_fastgb_fd_single(simulator=simulator, buffer=buffer, dt=dt, oversample=oversample)
        f = np.linspace(0, (len(buffer[0])-1)*1.0/self.T_obs, len(buffer[0]))
        return (f, ) + buffer

    def get_fastgb_td(self, dt=15.0, simulator='synthlisa', oversample=1):
        f, X, Y, Z = self.get_fastgb_fd(simulator, dt=dt, oversample=oversample)
        df = 1.0/self.T_obs
        kmin = round(f[0]/df)

        def ifft(arr):
            # n = int(1.0/(dt*df))
            n = round(1.0/(dt*df))
            # by liyn (in case the int() function would cause loss of n)

            ret = np.zeros(int(n/2+1), dtype=arr.dtype)
            ret[kmin:kmin+len(arr)] = arr[:]
            ret *= n  # normalization, ehm, found empirically

            return np.fft.irfft(ret)

        X, Y, Z = ifft(X), ifft(Y), ifft(Z)
        t = np.arange(len(X))*dt
        return t, X, Y, Z
