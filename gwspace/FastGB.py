import numpy as np

from gwspace.constants import YRSID_SI
from gwspace.Waveform import GCBWaveform

if __package__ or "." in __name__:
    from gwspace import libFastGB
else:
    import libFastGB


class FastGB(GCBWaveform):
    """ Calculate the GCB waveform using fast/slow """

    def _buffer_size(self, oversample=1):
        mult = 1 << int(np.ceil(np.log2(self.T_obs/YRSID_SI)))  # next power of 2
        try:
            N = max(2048 >> int(np.floor(-2*np.log10(self.f0))), 32)
        except ValueError:
            N = 2048
        return N*mult*oversample

    def get_fastgb_fd_single(self, simulator='synthlisa', buffer=None, dt=15., oversample=1, detector='TianQin'):
        # FIXME: assume T=T_obs below
        N = self._buffer_size(oversample)

        XLS = np.zeros(2*N, 'd')
        YLS = np.zeros(2*N, 'd')
        ZLS = np.zeros(2*N, 'd')

        XSL = np.zeros(2*N, 'd')
        YSL = np.zeros(2*N, 'd')
        ZSL = np.zeros(2*N, 'd')

        params = np.array([self.f0, self.fdot, self.Beta, self.Lambda, self.amp, self.iota, self.psi, self.phi0])

        if np.all(params) is not None:
            libFastGB.ComputeXYZ_FD(params, N, self.T_obs, dt, XLS, YLS, ZLS, XSL, YSL, ZSL,
                                    len(params), detector=detector)
            # TODO Need to transform to SL if required
            Xf, Yf, Zf = XLS, YLS, ZLS
            if simulator == 'synthlisa':
                Xf, Yf, Zf = XSL, YSL, ZSL
        else:
            raise ValueError

        Xf, Yf, Zf = Xf.view(np.complex128), Yf.view(np.complex128), Zf.view(np.complex128)
        f0 = self.f0
        if buffer is None:
            kmin = int(f0*self.T_obs)-N/2
            df = 1.0/self.T_obs
            f_range = np.linspace(kmin*df, (kmin+N-1)*df, N)
            return f_range, Xf, Yf, Zf
        else:
            blen, alen = len(buffer[0]), N

            # for a full buffer, "a" begins and ends at these indices
            beg, end = int(int(f0*self.T_obs)-N/2), int(f0*self.T_obs+N/2)
            # alignment of partial buffer with "a"
            begb, bega = max(beg, 0), max(0, -beg)
            endb, enda = min(end, blen), alen-max(0, end-blen)

            for i, a in enumerate((Xf, Yf, Zf)):
                buffer[i][begb:endb] += a[bega:enda]

    def get_fastgb_fd(self, simulator='synthlisa', dt=15., oversample=1):
        length = int(0.5*self.T_obs/dt)+1  # was "NFFT = int(T/dt)", and "NFFT/2+1" passed to numpy.zeros
        buffer = tuple(np.zeros(length, dtype=np.complex128) for _ in range(3))

        # for _ in table: TODO: make it support multiple sources?
        self.get_fastgb_fd_single(simulator=simulator, buffer=buffer, dt=dt, oversample=oversample)
        f = np.linspace(0, (length-1)*1.0/self.T_obs, length)
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
