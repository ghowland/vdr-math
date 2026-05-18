"""
vdr.signal — Exact digital signal processing.

    from vdr.signal.convolution import convolve
    from vdr.signal.dft import exact_dft, exact_idft
    from vdr.signal.filters import iir_filter

All digital signals are rational (ADC outputs are integers over 2^bits).
Every frequency bin, every filter output, every convolution result is exact.
DFT roundtrip: IDFT(DFT(x)) == x exactly.
"""
