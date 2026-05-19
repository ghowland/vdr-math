<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `signal`
vdr.signal — Exact digital signal processing. 

 from vdr.signal.convolution import convolve  from vdr.signal.dft import exact_dft, exact_idft  from vdr.signal.filters import iir_filter 

All digital signals are rational (ADC outputs are integers over 2^bits). Every frequency bin, every filter output, every convolution result is exact. DFT roundtrip: IDFT(DFT(x)) == x exactly. 

**Global Variables**
---------------
- **SIG_DFL**
- **SIG_IGN**
- **NSIG**
- **SIG_BLOCK**
- **SIG_UNBLOCK**
- **SIG_SETMASK**
- **SIGHUP**
- **SIGINT**
- **SIGQUIT**
- **SIGILL**
- **SIGTRAP**
- **SIGIOT**
- **SIGABRT**
- **SIGFPE**
- **SIGKILL**
- **SIGBUS**
- **SIGSEGV**
- **SIGSYS**
- **SIGPIPE**
- **SIGALRM**
- **SIGTERM**
- **SIGUSR1**
- **SIGUSR2**
- **SIGCLD**
- **SIGCHLD**
- **SIGPWR**
- **SIGIO**
- **SIGURG**
- **SIGWINCH**
- **SIGPOLL**
- **SIGSTOP**
- **SIGTSTP**
- **SIGCONT**
- **SIGTTIN**
- **SIGTTOU**
- **SIGVTALRM**
- **SIGPROF**
- **SIGXCPU**
- **SIGXFSZ**
- **SIGRTMIN**
- **SIGRTMAX**
- **ITIMER_REAL**
- **ITIMER_VIRTUAL**
- **ITIMER_PROF**


---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Handlers`
An enumeration. 





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ItimerError`








---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Sigmasks`
An enumeration. 





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Signals`
An enumeration. 





---

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/signal/__init__.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `struct_siginfo`
struct_siginfo: Result from sigwaitinfo or sigtimedwait. 

This object may be accessed either as a tuple of (si_signo, si_code, si_errno, si_pid, si_uid, si_status, si_band), or via the attributes si_signo, si_code, and so on. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
