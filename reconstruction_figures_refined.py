# %% [markdown]
# # Datta–Bondi–Gair reconstruction: corrected density and shell crossing
#
# This notebook reconstructs both examples from the supplied notebooks.
# The oscillating example uses a deliberately cored NFW-like snapshot to delay crossings. It evolves the coordinate transform between particle
# proper time tau and comoving time t. The physical radial separation is
#
# D = (dR/dM)|_t = A + U K,
# A = (dR/dM)|_tau, U = (dR/dtau)|_M, K = (dtau/dM)|_t.
#
# From Gair's comoving metric relations (Lambda=0),
#
# dK/dtau = [2 L² / (R(R²+L²))] D - [(L²)_{,M}/(2(R²+L²))].
#
# The physical energy density is rho = 1/(4 pi R² D). A=0 alone is NOT
# the shell-crossing condition. Both examples initialize K(tau_1,M)=0, i.e.
# tau=tau_1 labels one initial comoving t=constant hypersurface.
#
# The original collapse notebook uses an isothermal-like R1(M), while the
# submitted article defines a stretched Jaffe profile. We default to the
# manuscript's stretched profile; set COLLAPSE_INITIAL='legacy_isothermal'
# below to recreate the older input. The original NFW notebook uses alpha=0.25, but the article says alpha=0.18; we retain 0.18.
# To obtain long-lived oscillations, the finite-time density is NFW for M>=0.18,
# but smoothly approaches the Einasto equilibrium profile in the inner mass range.
#
# With M_tot=1 in geometrized units, axes are labeled M, R, tau, rho directly.
# Discrete oscillation figures omit tau=4450; continuous trajectories run to tau=4400.
# Absolute profile comparisons are paired with percentage differences, avoiding overlaps.
# All figures showing physical densities or shell separation stop before the
# first resolved shell crossing. The stability interval is explicitly monitored.

# %%
from pathlib import Path
from dataclasses import dataclass
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

OUTPUT = Path('figures')
OUTPUT.mkdir(exist_ok=True)
TAU_1 = 500.0
M_TOT = 1.0
COLLAPSE_INITIAL = 'paper_stretched'  # or 'legacy_isothermal'
EINASTO_ALPHA = 0.18  # original notebook: 0.25; latest paper: 0.18
N_MASS_COLLAPSE = 500
N_MASS_OSC = 500
M_MIN_COLLAPSE = 0.01
M_MIN_OSC = 0.002
M_MAX_OSC = 0.998
OSC_TIME_END = 11000.0  # stability scan horizon (ends at first resolved crossing)
DENSITY_CUTOFF = 1e-10

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'axes.linewidth': 0.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'figure.dpi': 120,
})

def nice_figure():
    fig, ax = plt.subplots(figsize=(5.9, 3.45), constrained_layout=True)
    ax.tick_params(direction='in', top=True, right=True)
    return fig, ax

def save_fig(fig, name, legacy=None):
    fig.savefig(OUTPUT/name)
    if legacy is not None and legacy != name:
        fig.savefig(OUTPUT/legacy)
    plt.close(fig)


def k_rate(R, U, A, K, L2, dL2):
    '''Equation from Gair: (d/dtau) (dtau/dM)|_t, evaluated at fixed M.'''
    D = A + U*K
    return 2.0*L2*D/(R*(R*R + L2)) - 0.5*dL2/(R*R + L2)


def fields(R, U, A, K):
    '''Physically relevant radial Jacobian and energy density.'''
    D = A + U*K
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where((D > DENSITY_CUTOFF) & (R > 0),
                       1.0/(4.0*np.pi*R*R*D), np.nan)
    return D, rho


def first_crossing(time, D_min):
    '''First sign change in sampled minimum, refined by caller if possible.'''
    where = np.where(D_min <= 0.0)[0]
    if len(where) == 0:
        return None
    i = where[0]
    if i == 0:
        return float(time[0])
    return float(np.interp(0.0, [D_min[i], D_min[i-1]], [time[i], time[i-1]]))

# %% [markdown]
# ## 1. Outer double-root collapse construction
#
# Use the monotone signed parameter y defined by
# R = R_EC + (R_ta - R_EC) sech²(y).
#
# It is positive before turnaround, zero at turnaround and negative after.
# Unlike a direct second-order integration near an *unstable* repeated root,
# this parameterization preserves the outer branch without accidental
# numerical departures from the exponentially approached endpoint.
# We evolve the mass derivative y_M alongside y, so the physical Jacobian
# D follows from *tangent equations*, not finite differences of adjacent
# evolved shell trajectories.

# %%
@dataclass
class Collapse:
    M: np.ndarray
    rec: np.ndarray
    rt: np.ndarray
    E: np.ndarray
    L2: np.ndarray
    R1: np.ndarray
    R1_M: np.ndarray
    y1: np.ndarray
    y1_M: np.ndarray
    rec_M: np.ndarray
    delta: np.ndarray
    delta_M: np.ndarray
    b: np.ndarray
    b_M: np.ndarray
    L2_M: np.ndarray
    sol: object

    def at(self, tau):
        y, ym, K = self.sol.sol(float(tau)).reshape(3, len(self.M))
        z = np.tanh(y)
        sh2 = 1.0-z*z
        R = self.rec + self.delta*sh2
        A = self.rec_M + self.delta_M*sh2 - 2*self.delta*sh2*z*ym
        dy = -self.b / R**1.5
        U = -2*self.delta*sh2*z*dy
        D, rho = fields(R, U, A, K)
        return dict(R=R, U=U, A=A, K=K, D=D, rho=rho)


def make_collapse(initial=COLLAPSE_INITIAL, n=N_MASS_COLLAPSE, end=752.62):
    M = np.linspace(M_MIN_COLLAPSE, M_TOT, n)
    mj, aj = 6.0, 27.0
    q = aj/(mj-M)
    qp = aj/(mj-M)**2
    rec, rec_M = M*q, q+M*qp
    E = (4-q)/(2*q*(q-3))
    E_M = E*(-qp/(4-q) -qp/q - qp/(q-3))
    rt = 2*q*M/(q-4)
    rt_M = 2*(q/(q-4)-4*M*qp/(q-4)**2)
    delta, delta_M = rt-rec, rt_M-rec_M
    if initial == 'paper_stretched':
        g = np.exp(-((M-.45)/.2)**2)
        S = 1.18+.05*g
        Sp = -.1*(M-.45)/(.2**2)*g
        R1, R1_M = S*rec, Sp*rec + S*rec_M
    elif initial == 'legacy_isothermal':
        R1, R1_M = -44*np.log1p(-M/8), 44/(8-M)
    else:
        raise ValueError(initial)
    assert np.all((rec<R1)&(R1<rt)&(R1_M>0))
    L2 = M*rec**2/(rec-3*M)
    L2_M = (rec**2+2*M*rec*rec_M)/(rec-3*M) - M*rec**2*(rec_M-3)/(rec-3*M)**2
    b = .5*np.sqrt(-2*E*delta)
    b_M = .5*b*(E_M/E+delta_M/delta)
    v = (rt-R1)/delta
    vp = (rt_M-R1_M)/delta -v*delta_M/delta
    y1 = np.arctanh(np.sqrt(v))
    ym1 = vp/(2*np.sqrt(v)*(1-v))

    def rhs(t, Y):
        y, ym, K = Y.reshape(3,n)
        z = np.tanh(y)
        sech2 = 1-z*z
        R = rec+delta*sech2
        A = rec_M+delta_M*sech2-2*delta*sech2*z*ym
        dy = -b/R**1.5
        U = -2*delta*sech2*z*dy
        dym = dy*(b_M/b -1.5*A/R)
        return np.r_[dy, dym, k_rate(R,U,A,K,L2,L2_M)]
    sol=solve_ivp(rhs,(TAU_1,end),np.r_[y1,ym1,np.zeros(n)],
                  rtol=2e-9,atol=2e-10,max_step=1.0,dense_output=True)
    if not sol.success: raise RuntimeError(sol.message)
    return Collapse(M,rec,rt,E,L2,R1,R1_M,y1,ym1,rec_M,delta,delta_M,b,b_M,L2_M,sol)

collapse = make_collapse()
cc0 = collapse.at(TAU_1)
assert np.max(abs(cc0['R']-collapse.R1))<1e-10
assert np.max(abs(cc0['D']-collapse.R1_M))<1e-9
assert np.all(cc0['D']>0)
print('Collapse: initial profile recovered; minimum initial D =', cc0['D'].min())

# %%
COLLAPSE_TIMES = [500.0, 552.07, 602.62, 652.62, 702.62, 752.62]
M=collapse.M
fig,ax=nice_figure()
ax.plot(M,1/(4*np.pi*collapse.rec**2*collapse.rec_M),label=r'$\rho_{\rm EC}$')
ax.plot(M,1/(4*np.pi*collapse.R1**2*collapse.R1_M),label=r'$\rho_1$')
ax.set(yscale='log',xlabel=r'$M$',ylabel=r'$\rho$')
ax.legend(frameon=False);save_fig(fig,'collapse_initial_density.pdf','collapse_jaffe.pdf')

fig,ax=nice_figure()
ax.plot(M,collapse.rec,'k--',lw=1.2,label=r'$R_{\rm EC}$')
ax.plot(M,collapse.rt,'k:',lw=1.3,label=r'$R_{\rm ta}$')
for t in COLLAPSE_TIMES:
    ax.plot(M,collapse.at(t)['R'],lw=1.25,label=rf'$\tau={t:g}$')
ax.set(xlabel=r'$M$',ylabel=r'$R$')
ax.legend(ncol=2,frameon=False,loc='upper left');save_fig(fig,'collapse_radius.pdf','areal_radius.pdf')

fig,ax=nice_figure()
for t in COLLAPSE_TIMES:
    v=collapse.at(t)
    if np.any(v['D']<=0): print('WARNING: collapse crossing by time',t)
    ax.plot(M,v['rho'],lw=1.25,label=rf'$\tau={t:g}$')
ax.set(yscale='log',xlabel=r'$M$',ylabel=r'$\rho(\tau,M)$')
ax.legend(ncol=2,frameon=False);save_fig(fig,'collapse_density_evolution.pdf','density_profiles.pdf')

fig,ax=nice_figure()
for t in COLLAPSE_TIMES:
    ax.plot(M,collapse.at(t)['D'],lw=1.25,label=rf'$\tau={t:g}$')
ax.axhline(0,color='black',ls=':',lw=1)
ax.set(xlabel=r'$M$',ylabel=r'$D=(\partial R/\partial M)_t$')
ax.legend(ncol=2,frameon=False);save_fig(fig,'collapse_shell_separation.pdf','shell_crossing.pdf')

# %% [markdown]
# ## 2. Long-lived Einstein-cluster oscillations with an NFW outer snapshot
#
# An exactly NFW finite-time slice and an Einasto equilibrium with alpha=0.18
# develop a shell crossing at ~766 for eta=0.5 and M in [0.002,0.998].
# To preserve the outer NFW profile while avoiding large inner displacements,
# set R_1=R_0+w(M)*(R_NFW-R_0) and U_1=-eta*omega*(R_1-R_0).
# The quintic transition w is C^2 and **exactly 1** for M >= M_CORE_OUTER.
# We use w=epsilon (epsilon=0.01) for M <= M_CORE_INNER; the innermost
# shells are nearly circular, but not artificially forced to be static.
# The snapshot is therefore **not globally NFW**; it is a smoothly cored,
# NFW-matched density slice. This change needs to be stated in the manuscript.
#
# We integrate (R,U,A,B,K), with A=R_M|tau and B=U_M|tau. The physical
# radial separation is D=A+U*K. The integrator stops at its first detected
# zero; subsequent trajectories have no single-stream physical interpretation.

# %%
ETA = 0.5
M_CORE_INNER = 0.02
M_CORE_OUTER = 0.18
CORE_EPSILON = 0.01
SNAPSHOT_MODE = 'matched_core'  # 'pure_nfw' reproduces early crossing
DISPLAY_TIMES = [500.0, 982.87, 1465.74, 1948.62, 2431.49, 3400.0]
DISPLAY_END = max(DISPLAY_TIMES)
TRAJECTORY_END = 4400.0  # continuous trajectories only; no plotted 4450 snapshot


def matching_weight(M, mode=SNAPSHOT_MODE):
    """Return w(M), dw/dM. Monotone C2 transition with exact NFW outside."""
    if mode == 'pure_nfw':
        return np.ones_like(M), np.zeros_like(M)
    if mode != 'matched_core':
        raise ValueError(f'Unknown SNAPSHOT_MODE={mode!r}')
    x = np.clip((M-M_CORE_INNER)/(M_CORE_OUTER-M_CORE_INNER), 0., 1.)
    smootherstep = x**3*(10. - 15.*x + 6.*x*x)
    w = CORE_EPSILON + (1.-CORE_EPSILON)*smootherstep
    wp = np.where((M>M_CORE_INNER)&(M<M_CORE_OUTER),
                  (1.-CORE_EPSILON)*30.*x*x*(1.-x)**2/(M_CORE_OUTER-M_CORE_INNER), 0.)
    return w, wp


@dataclass
class Oscillations:
    M: np.ndarray
    R0: np.ndarray
    RNFW: np.ndarray
    R1: np.ndarray
    R0_M: np.ndarray
    RNFW_M: np.ndarray
    R1_M: np.ndarray
    weight: np.ndarray
    L2: np.ndarray
    L2_M: np.ndarray
    U1: np.ndarray
    E: np.ndarray
    E0: np.ndarray
    Eu: np.ndarray
    sol: object
    tstop: float
    crossing: object
    crossing_mass: object

    def at(self,tau):
        if not (TAU_1-1e-7 <= tau <= self.tstop+1e-7):
            raise ValueError(f'Outside integrated domain {TAU_1} <= tau <= {self.tstop}: {tau}')
        R,U,A,B,K=self.sol.sol(float(tau)).reshape(5,len(self.M))
        D,rho=fields(R,U,A,K)
        return dict(R=R,U=U,A=A,B=B,K=K,D=D,rho=rho)


def make_oscillations(alpha=EINASTO_ALPHA, n=N_MASS_OSC, end=OSC_TIME_END, mode=SNAPSHOT_MODE):
    M=np.linspace(M_MIN_OSC,M_MAX_OSC,n)
    rv,rs=100.,12.5
    rr=np.r_[0.,np.geomspace(1e-6,rv,50000)]
    rho_ein=np.exp(-2/alpha*((rr/rs)**alpha-1.))
    rho_nfw=np.where(rr>0,1/(np.maximum(rr/rs,1e-100)*(1+rr/rs)**2),0.)

    def mass_table(shape):
        raw=cumulative_trapezoid(4*np.pi*rr**2*shape,rr,initial=0.)
        return raw/raw[-1],1/raw[-1]
    me,ne=mass_table(rho_ein)
    mn,nn=mass_table(rho_nfw)
    R0=PchipInterpolator(me,rr)(M)
    RNFW=PchipInterpolator(mn,rr)(M)
    rhoE=ne*np.exp(-2/alpha*((R0/rs)**alpha-1.))
    rhoN=nn/((RNFW/rs)*(1+RNFW/rs)**2)
    R0_M=1/(4*np.pi*R0**2*rhoE)
    RNFW_M=1/(4*np.pi*RNFW**2*rhoN)
    w,wp=matching_weight(M,mode=mode)
    R1=R0+w*(RNFW-R0)
    R1_M=R0_M+w*(RNFW_M-R0_M)+wp*(RNFW-R0)
    assert np.all(R0>6*M), 'Unstable equilibrium shell.'
    assert np.all(R1_M>0), 'Initial shell crossing in specified slice.'
    assert np.max(abs(R1[M>=M_CORE_OUTER]-RNFW[M>=M_CORE_OUTER]))<1e-12

    L2=M*R0**2/(R0-3*M)
    L2_M=(R0**2+2*M*R0*R0_M)/(R0-3*M)-M*R0**2*(R0_M-3)/(R0-3*M)**2
    omega=np.sqrt(M*(R0-6*M)/(R0**3*(R0-3*M)))
    omega_M=.5*omega*(1/M+(R0_M-6)/(R0-6*M)-3*R0_M/R0-(R0_M-3)/(R0-3*M))
    U1=-ETA*omega*(R1-R0)
    U1_M=-ETA*(omega_M*(R1-R0)+omega*(R1_M-R0_M))
    E=.5*U1**2-M/R1-M*L2/R1**3+L2/(2*R1**2)
    E0=M*(4*M-R0)/(2*R0*(R0-3*M))
    Ru=L2/(2*M)*(1-np.sqrt(1-12*M*M/L2))
    Eu=-M/Ru-M*L2/Ru**3+L2/(2*Ru**2)
    assert np.all(E<0) and np.all(E<Eu) and np.all(E>=E0-5e-14), 'Unbound or inadmissible orbit.'

    def rhs(t,Y):
        R,U,A,B,K=Y.reshape(5,n)
        F=-M/R**2-3*M*L2/R**4+L2/R**3
        FR=2*M/R**3+12*M*L2/R**5-3*L2/R**4
        FM=-1/R**2-3*(L2+M*L2_M)/R**4+L2_M/R**3
        return np.r_[U,F,B,FR*A+FM,k_rate(R,U,A,K,L2,L2_M)]

    def physical_crossing_event(t,Y):
        R,U,A,B,K=Y.reshape(5,n)
        return np.min(A+U*K)
    physical_crossing_event.terminal=True
    physical_crossing_event.direction=-1

    init=np.r_[R1,U1,R1_M,U1_M,np.zeros(n)]
    sol=solve_ivp(rhs,(TAU_1,end),init,rtol=3e-9,atol=2e-10,
                  dense_output=True,max_step=1.5, events=physical_crossing_event)
    if not sol.success: raise RuntimeError(sol.message)
    crossing = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    tstop=float(sol.t[-1])
    if crossing is not None:
        yy=sol.y_events[0][0].reshape(5,n)
        dc=yy[2]+yy[1]*yy[4]
        crossing_mass=float(M[np.argmin(dc)])
    else:
        crossing_mass=None
    return Oscillations(M,R0,RNFW,R1,R0_M,RNFW_M,R1_M,w,L2,L2_M,
                        U1,E,E0,Eu,sol,tstop,crossing,crossing_mass)

osc=make_oscillations()
initial=osc.at(TAU_1)
assert np.max(abs(initial['R']-osc.R1))<1e-10
assert np.max(abs(initial['D']-osc.R1_M))<1e-9
assert osc.tstop>max(DISPLAY_END,TRAJECTORY_END), 'Plot range includes shell crossing!'
print(f'Input: alpha={EINASTO_ALPHA:.3f}, eta={ETA:.3f}, M_core=[{M_CORE_INNER},{M_CORE_OUTER}], epsilon={CORE_EPSILON:g}')
print('Initial slice: exactly NFW outside M >=',M_CORE_OUTER)
print('First resolved crossing:',osc.crossing,'at M=',osc.crossing_mass)
print('Numerically verified pre-crossing display window:',TAU_1,'-',DISPLAY_END)
print('Smallest E-E0:',np.min(osc.E-osc.E0),'   smallest Eu-E:',np.min(osc.Eu-osc.E))

# %% [markdown]
# ## 3. Publication-ready figures
#
# Old figure names are exported as aliases, to simplify updating the existing
# LaTeX. Density plotted at proper-time events uses the physical fixed-t
# derivative D. In the initial slice D=R_{1,M}; the matched profile equals
# the pure NFW slice only for M >= M_CORE_OUTER.

# %%
M=osc.M
rho_initial=initial['rho']
rho_nfw_initial=1/(4*np.pi*osc.RNFW**2*osc.RNFW_M)
rho_ein_initial=1/(4*np.pi*osc.R0**2*osc.R0_M)

# --- Contrast-based input profiles: both an absolute and fractional view ---
def add_core_bounds(ax, label=False):
    ax.axvspan(M_CORE_INNER,M_CORE_OUTER,color='0.92',zorder=0,
               label=('inner transition' if label else None))
    ax.axvline(M_CORE_OUTER,color='0.55',lw=.9,ls=':')

def compare_profiles(kind, outfile, alias=None):
    fig,(ax0,ax1)=plt.subplots(2,1,figsize=(6.25,5.2),sharex=True,
        gridspec_kw={'height_ratios':[1.,1.1]},constrained_layout=True)
    for ax in (ax0,ax1):
        ax.tick_params(direction='in',top=True,right=True)
        add_core_bounds(ax)
    if kind=='radii':
        ax0.plot(M,osc.R0,color='0.15',lw=1.5,ls='--',label=r'Einasto equilibrium $R_0$')
        ax0.plot(M,osc.RNFW,color='#d95f02',lw=1.4,ls=':',label=r'pure NFW $R_{\rm NFW}$')
        ax0.plot(M,osc.R1,color='#0072b2',lw=1.55,label=r'matched snapshot $R_1$')
        ax0.set_ylabel(r'$R$')
        ax1.plot(M,100*(osc.R1/osc.R0-1),lw=1.65,color='#0072b2',
                 label=r'matched $(R_1/R_0-1)\times100\%$')
        ax1.plot(M,100*(osc.RNFW/osc.R0-1),lw=1.15,ls=':',color='#d95f02',
                 label=r'pure NFW $(R_{\rm NFW}/R_0-1)\times100\%$')
        ax1.set_ylabel(r'$\Delta R/R_0$ [\%]')
    else:
        ax0.plot(M,rho_ein_initial,color='0.15',lw=1.5,ls='--',label='Einasto equilibrium')
        ax0.plot(M,rho_nfw_initial,color='#d95f02',lw=1.4,ls=':',label='pure NFW')
        ax0.plot(M,rho_initial,color='#0072b2',lw=1.55,label='matched snapshot')
        ax0.set_yscale('log')
        ax0.set_ylabel(r'$\rho$')
        ax1.plot(M,100*(rho_initial/rho_ein_initial-1),lw=1.65,color='#0072b2',
                 label=r'matched $(\rho_1/\rho_0-1)\times100\%$')
        ax1.axhline(0,color='0.25',ls='--',lw=.8)
        ax1.set_ylabel(r'$\Delta\rho/\rho_0$ [\%]')
        ax1.set_ylim(-9.5,11.5)
    ax1.axhline(0,color='0.3',ls=':',lw=.9) if kind=='radii' else None
    ax0.legend(frameon=False,fontsize=8.5,ncol=2,loc='best')
    ax1.set_xlabel(r'$M$')
    ax1.legend(frameon=False,fontsize=8.4,loc='best')
    save_fig(fig,outfile,alias)

compare_profiles('radii','oscillating_profile_radius_contrast.pdf','oscillations_density.pdf')
compare_profiles('densities','oscillating_profile_density_contrast.pdf')

# Standalone percentage contrasts, useful for a compact two-panel journal figure
fig,(ax0,ax1)=plt.subplots(2,1,figsize=(6.25,5.1),sharex=True,constrained_layout=True)
for a in (ax0,ax1):
    a.tick_params(direction='in',top=True,right=True)
    add_core_bounds(a)
ax0.plot(M,100*(osc.R1/osc.R0-1),color='#0072b2',lw=1.8,
         label=r'matched NFW-like snapshot $R_1$')
ax0.axhline(0,color='0.25',ls='--',lw=.8)
ax0.set_ylabel(r'$(R_1/R_0-1)$ [\%]'); ax0.set_ylim(-2.6,1.5)
ax0.legend(frameon=False,loc='upper right')
ax1.plot(M,100*(rho_initial/rho_ein_initial-1),color='#0072b2',lw=1.6)
ax1.axhline(0,color='0.25',ls='--',lw=.8)
ax1.set_ylabel(r'$(\rho_1/\rho_0-1)$ [\%]')
ax1.set_xlabel(r'$M$')
save_fig(fig,'paper_fig3_oscillating_input_contrasts.pdf')

fig,ax=nice_figure()
ax.plot(M,osc.weight,lw=1.5,label=r'$w(M)$')
ax.axvline(M_CORE_INNER,color='.65',ls=':',lw=1,label=r'$M_a$')
ax.axvline(M_CORE_OUTER,color='.5',ls='--',lw=1,label=r'$M_b$')
ax.set(xlabel=r'$M$',ylabel=r'$w(M)$',ylim=(-.04,1.08))
ax.legend(frameon=False)
save_fig(fig,'oscillating_core_weight.pdf')

# Select admissible snapshots, omitting the visually overcrowded tau=4450 slice
fig,ax=nice_figure()
ax.plot(M,osc.R0,color='k',ls='--',lw=1,label=r'$R_0$')
for t in DISPLAY_TIMES:
    ax.plot(M,osc.at(t)['R'],lw=1.1,label=rf'$\tau={t:g}$')
ax.set(xlabel=r'$M$',ylabel=r'$R$')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_radius_long_lived.pdf','oscillations_radius.pdf')

fig,ax=nice_figure()
for t in DISPLAY_TIMES:
    ax.plot(M,osc.at(t)['R']/osc.R0,lw=1.15,label=rf'$\tau={t:g}$')
ax.axhline(1,color='k',ls=':',lw=1,label=r'$R/R_0=1$')
ax.set(xlabel=r'$M$',ylabel=r'$R(\tau,M)/R_0(M)$')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_fractional_radius_long_lived.pdf','oscillations2.pdf')

fig,ax=nice_figure()
tlong=np.linspace(TAU_1,TRAJECTORY_END,700)
for target in [.07,.13,.25,.50,.75]:
    i=int(np.argmin(np.abs(M-target)))
    trajectory=np.array([osc.at(t)['R'][i]/osc.R0[i] for t in tlong])
    ax.plot(tlong,trajectory,lw=1.15,label=rf'$M={M[i]:.3f}$')
ax.axhline(1,color='k',ls=':',lw=1)
ax.set(xlabel=r'$\tau$',ylabel=r'$R(\tau,M)/R_0(M)$')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_shell_trajectories_long_lived.pdf','oscillations.pdf')

fig,ax=nice_figure()
for t in DISPLAY_TIMES:
    f=osc.at(t)
    assert np.min(f['D'])>0
    ax.plot(M,f['rho'],lw=1.15,label=rf'$\tau={t:g}$')
ax.set(xlabel=r'$M$',ylabel=r'$\rho(\tau,M)$',yscale='log')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_density_long_lived.pdf')

fig,ax=nice_figure()
for t in DISPLAY_TIMES:
    ax.plot(M,osc.at(t)['rho']/rho_initial,lw=1.15,label=rf'$\tau={t:g}$')
ax.axhline(1,color='k',ls=':',lw=1)
ax.set(xlabel=r'$M$',ylabel=r'$\rho(\tau,M)/\rho(\tau_1,M)$')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_density_ratio_long_lived.pdf')

fig,ax=nice_figure()
for t in DISPLAY_TIMES:
    ax.plot(M,osc.at(t)['D'],lw=1.15,label=rf'$\tau={t:g}$')
ax.axhline(0,color='k',ls=':',lw=1)
ax.set(xlabel=r'$M$',ylabel=r'$D=(\partial R/\partial M)_t$')
ax.legend(ncol=2,frameon=False)
save_fig(fig,'oscillating_shell_separation_long_lived.pdf')

# Normalize by the physical shell separation on the synchronized initial slice.
# Since D(tau_1,M)>0, the normalized diagnostic has the same crossing criterion.
fig,ax=nice_figure()
for t in DISPLAY_TIMES:
    ax.plot(M,osc.at(t)['D']/initial['D'],lw=1.2,label=rf'$\tau={t:g}$')
ax.axhline(1,color='0.25',ls=':',lw=.9,label='initial')
ax.axhline(0,color='0.4',ls='--',lw=.75)
ax.set(xlabel=r'$M$',ylabel=r'$D(\tau,M)/D(\tau_1,M)$')
ax.legend(ncol=2,frameon=False,fontsize=8.3)
save_fig(fig,'oscillating_shell_separation_ratio_long_lived.pdf')

# Continuously integrated crossing event, sampled scan only for visualization.
scan_end=min(osc.tstop,10500)
T_SCAN=np.linspace(TAU_1,scan_end,300)
D_MIN=np.array([np.min(osc.at(t)['D']) for t in T_SCAN])
fig,ax=nice_figure()
ax.plot(T_SCAN,D_MIN,lw=1.5,label=r'$\min_M D(\tau,M)$')
ax.axhline(0,color='k',ls=':',lw=1)
ax.axvline(DISPLAY_END,color='.4',ls='--',lw=1.1,label=rf'last snapshot $\tau={DISPLAY_END:g}$')
if osc.crossing is not None:
    ax.axvline(osc.crossing,color='#b2182b',ls='-.',lw=1.1,
               label=rf'first crossing $\tau\simeq {osc.crossing:.0f}$')
ax.set(xlabel=r'$\tau$',ylabel=r'$\min_M (\partial R/\partial M)_t$')
ax.legend(frameon=False)
save_fig(fig,'oscillating_minimum_D_time_scan.pdf')

# --- CQG-oriented paired diagnostic figures; all curves from the same numerical solution ---
def two_panel_setup(height=5.8):
    fig,(a,b)=plt.subplots(2,1,figsize=(6.3,height),constrained_layout=True)
    for x in (a,b): x.tick_params(direction='in',top=True,right=True)
    return fig,a,b

fig,a,b=two_panel_setup()
a.plot(collapse.M,1/(4*np.pi*collapse.rec**2*collapse.rec_M),label=r'$\rho_{\rm EC}$',lw=1.55)
a.plot(collapse.M,1/(4*np.pi*collapse.R1**2*collapse.R1_M),label=r'$\rho_1$',lw=1.5)
a.set(yscale='log',xlabel=r'$M$',ylabel=r'$\rho$');a.legend(frameon=False)
b.plot(collapse.M,collapse.rec,color='black',ls='--',lw=1.,label=r'$R_{\rm EC}$')
b.plot(collapse.M,collapse.rt,color='black',ls=':',lw=1.,label=r'$R_{\rm ta}$')
for t in COLLAPSE_TIMES:
    b.plot(collapse.M,collapse.at(t)['R'],lw=1.1,label=rf'$\tau={t:g}$')
b.set(xlabel=r'$M$',ylabel=r'$R$');b.legend(ncol=3,fontsize=8,frameon=False)
save_fig(fig,'paper_fig1_collapse_input_evolution.pdf')

fig,a,b=two_panel_setup()
for t in COLLAPSE_TIMES:
    f=collapse.at(t)
    a.plot(collapse.M,f['rho'],lw=1.15,label=rf'$\tau={t:g}$')
    b.plot(collapse.M,f['D'],lw=1.15,label=rf'$\tau={t:g}$')
a.set(yscale='log',xlabel=r'$M$',ylabel=r'$\rho(\tau,M)$')
b.set(xlabel=r'$M$',ylabel=r'$D=(\partial R/\partial M)_t$')
b.axhline(0,color='0.2',ls=':',lw=.9)
a.legend(ncol=3,fontsize=8,frameon=False)
save_fig(fig,'paper_fig2_collapse_density_D.pdf')

fig,a,b=two_panel_setup()
for target in [.07,.13,.25,.50,.75]:
    ix=np.argmin(abs(osc.M-target))
    vals=np.array([osc.at(t)['R'][ix]/osc.R0[ix] for t in tlong])
    a.plot(tlong,vals,lw=1.2,label=rf'$M={osc.M[ix]:.3f}$')
a.axhline(1,color='0.2',ls=':',lw=.9)
a.set(xlabel=r'$\tau$',ylabel=r'$R(\tau,M)/R_0(M)$')
a.legend(ncol=3,fontsize=8,frameon=False)
for t in DISPLAY_TIMES:
    b.plot(osc.M,osc.at(t)['R']/osc.R0,lw=1.1,label=rf'$\tau={t:g}$')
b.axhline(1,color='0.2',ls=':',lw=.9)
b.set(xlabel=r'$M$',ylabel=r'$R(\tau,M)/R_0(M)$')
b.legend(ncol=3,fontsize=8,frameon=False)
save_fig(fig,'paper_fig4_oscillating_trajectories_radius.pdf')

fig,a,b=two_panel_setup()
for t in DISPLAY_TIMES:
    f=osc.at(t)
    a.plot(osc.M,f['rho']/rho_initial,lw=1.1,label=rf'$\tau={t:g}$')
    b.plot(osc.M,f['D']/initial['D'],lw=1.1,label=rf'$\tau={t:g}$')
a.axhline(1,color='0.2',ls=':',lw=.9)
b.axhline(0,color='0.2',ls=':',lw=.9)
a.set(xlabel=r'$M$',ylabel=r'$\rho(\tau,M)/\rho(\tau_1,M)$')
b.set(xlabel=r'$M$',ylabel=r'$D(\tau,M)/D(\tau_1,M)$')
b.axhline(1,color='0.3',ls='--',lw=.75)
a.legend(ncol=3,fontsize=8,frameon=False)
save_fig(fig,'paper_fig5_oscillating_density_D.pdf')

# %% [markdown]
# ## 4. Diagnostics and reproducibility
#
# The dense-time integration checks D>0 at the selected times, while a terminal
# scalar event monitors min(D) continuously along the adaptive ODE trajectory.
# Event detection and mass sampling remain numerical, not a continuum proof.

# %%
rows=[]
for t in COLLAPSE_TIMES:
    f=collapse.at(t)
    rhs=-2*collapse.E/f['R']**3*(f['R']-collapse.rec)**2*(collapse.rt-f['R'])
    rows.append(dict(example='collapse',tau=t,min_D=float(f['D'].min()),
        min_A=float(f['A'].min()),fraction_nonpositive_D=float(np.mean(f['D']<=0)),
        max_abs_energy_residual=float(np.max(abs(f['U']**2-rhs)))))
for t in DISPLAY_TIMES:
    f=osc.at(t)
    W=2*osc.E+2*osc.M/f['R']-osc.L2/f['R']**2+2*osc.M*osc.L2/f['R']**3
    rows.append(dict(example='oscillation',tau=t,min_D=float(f['D'].min()),
        min_A=float(f['A'].min()),fraction_nonpositive_D=float(np.mean(f['D']<=0)),
        max_abs_energy_residual=float(np.max(abs(f['U']**2-W)))))
diagnostics=pd.DataFrame(rows)
diagnostics.to_csv('DBG_diagnostics.csv',index=False)
print(diagnostics.to_string(index=False,float_format=lambda x:f'{x:.6g}'))

# Sensitivity ODE vs. finite differences, checking the numerical mass derivative:
mid_time=2431.49
f=osc.at(mid_time)
a_fd=np.gradient(f['R'],osc.M,edge_order=2)
reldiff=np.abs(f['A'][2:-2]-a_fd[2:-2])/np.maximum(abs(f['A'][2:-2]),1e-10)
print(f'Sensitivity check tau={mid_time}: median relative error {np.median(reldiff):.2e}; max={np.max(reldiff):.2e}')
assert np.median(reldiff)<0.01

# The finite-time profile matches the NFW profile EXACTLY outside the core
mask=osc.M>=M_CORE_OUTER
nfw_r_rel=np.max(abs(osc.R1[mask]-osc.RNFW[mask])/osc.RNFW[mask])
nfw_rho_rel=np.max(abs(rho_initial[mask]-rho_nfw_initial[mask])/rho_nfw_initial[mask])
print(f'Outer NFW match: relative radius error {nfw_r_rel:.2e}, relative density error {nfw_rho_rel:.2e}')

# Compare to the unchanged NFW input and test the resolved crossing time
# against a finer mass mesh (not a continuum proof).
baseline = make_oscillations(mode='pure_nfw',n=N_MASS_OSC,end=1500.)
resolution_check = make_oscillations(mode=SNAPSHOT_MODE,n=1000,end=11000.)
print(f'Pure NFW snapshot: first crossing tau={baseline.crossing:.6f}, M={baseline.crossing_mass:.6f}')
print(f'Matched snapshot with 1000 shells: first crossing tau={resolution_check.crossing:.6f}, M={resolution_check.crossing_mass:.6f}')

# Orbital amplitude checks, energy conservation, and actual minimum D in figures:
minD_fig=float(min(np.min(osc.at(t)['D']) for t in DISPLAY_TIMES))
minD_scan=float(np.min(D_MIN))
maxrel=float(np.max(abs(osc.R1-osc.R0)/osc.R0))
print(f'Illustrated window: min sampled D={minD_fig:.6f}; min dense-grid D={minD_scan:.6f}')
print(f'Maximum fractional snapshot displacement {maxrel:.5f}; no sampled crossing through {DISPLAY_END:g}.')
summary={
    'example':'long-lived smoothly matched NFW outer shell',
    'Einasto_alpha':EINASTO_ALPHA,'eta':ETA,'tau_initial':TAU_1,
    'M_min':M_MIN_OSC,'M_max':M_MAX_OSC,'N_mass':N_MASS_OSC,
    'transition_mass_inner':M_CORE_INNER,'transition_mass_outer':M_CORE_OUTER,
    'core_weight_floor':CORE_EPSILON,'snapshot_mode':SNAPSHOT_MODE,
    'snapshot_is_exact_NFW_only_for_M_ge':M_CORE_OUTER,
    'first_resolved_shell_crossing_tau':osc.crossing,
    'first_resolved_shell_crossing_M':osc.crossing_mass,
    'baseline_pure_NFW_first_crossing_tau_500shells':baseline.crossing,
    'matched_first_crossing_tau_1000shells':resolution_check.crossing,
    'displayed_time_slices':DISPLAY_TIMES,
    'continuous_trajectory_end':TRAJECTORY_END,
    'minimum_D_on_displayed_slices':minD_fig,
    'minimum_D_on_sampled_pre_crossing_scan':minD_scan,
    'max_relative_snapshot_displacement':maxrel,
    'max_NFW_density_rel_error_above_M_b':float(nfw_rho_rel),
    'min_Eu_minus_E':float(np.min(osc.Eu-osc.E)),
    'min_E_minus_E0':float(np.min(osc.E-osc.E0)),
    'figures':[p.name for p in sorted(OUTPUT.glob('*.pdf'))],
}
Path('DBG_summary.json').write_text(json.dumps(summary,indent=2))
print('Wrote',len(summary['figures']),'PDFs into',OUTPUT)
