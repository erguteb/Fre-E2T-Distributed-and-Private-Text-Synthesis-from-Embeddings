#!/usr/bin/env python3
import argparse
import mpmath as mp
import numpy as np
from scipy.integrate import quad

def kl_bernoulli(q, p, *, base=np.e, clip=0.0):
    """
    D(q || p) = q * log(q/p) + (1-q) * log((1-q)/(1-p))

    Parameters
    ----------
    q, p : float or array-like
        Bernoulli parameters in [0, 1].
    base : float
        Logarithm base (default e). Use 2 for bits.
    clip : float
        If > 0, values are clipped to [clip, 1-clip] for numerical stability.

    Returns
    -------
    float or ndarray
        KL divergence (>= 0). Returns +inf when p=0 with q>0 or p=1 with q<1.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    if clip > 0:
        q = np.clip(q, clip, 1 - clip)
        p = np.clip(p, clip, 1 - clip)

    log = np.log
    if base != np.e:
        log = lambda x: np.log(x) / np.log(base)

    # Handle the 0·log(0/·) limits correctly via masking
    term1 = np.where(q == 0.0, 0.0, q * log(q / p))
    term2 = np.where(q == 1.0, 0.0, (1.0 - q) * log((1.0 - q) / (1.0 - p)))
    return term1 + term2

def f(mu):
    """
    Computes:
        f(mu) = (2 / sqrt(pi)) * ∫_0^(mu/sqrt(2)) e^(-a^2) da
                - (1/mu) * sqrt(2/pi) * (1 - e^(-mu^2/2))

    Parameters
    ----------
    mu : float or ndarray
        Input value(s) of μ.

    Returns
    -------
    float or ndarray
        Computed value(s) of f(mu).
    """
    mu = np.asarray(mu, dtype=float)

    def integrand(a):
        return np.exp(-a**2)

    # Vectorized integration for multiple μ
    def integral(mu_val):
        upper = mu_val / np.sqrt(2)
        val, _ = quad(integrand, 0, upper)
        return val

    integral_vals = np.vectorize(integral)(mu)

    term1 = (2 / np.sqrt(np.pi)) * integral_vals
    term2 = (1 / mu) * np.sqrt(2 / np.pi) * (1 - np.exp(-mu**2 / 2))
    return term1 - term2


def analytic_gaussian_epsilon(noise_multiplier, delta, tol=1e-10, max_eps=1e6, mp_dps=50):
    """
    Compute ε for the Analytic Gaussian Mechanism given:
        noise_multiplier = σ / S
        delta ∈ (0, 0.5)
    """
    if not (noise_multiplier > 0):
        raise ValueError("Require noise_multiplier > 0.")
    if not (0 < delta < 0.5):
        raise ValueError("delta must be in (0, 0.5).")

    # sets precision, dps=50 means 50 digits
    mp.mp.dps = mp_dps
    target = 1 / noise_multiplier  # S / σ

    def solve_chi(eps):
        eps = mp.mpf(eps)
        def f(chi):
            return mp.erfc(chi) - mp.e**(eps) * mp.erfc(mp.sqrt(chi*chi + eps)) - 2*delta

        hi = mp.mpf(1)
        while f(hi) > 0:
            hi *= 2
            if hi > 1e6:
                raise RuntimeError("Failed to bracket chi root. Try larger noise_multiplier or delta.")

        lo = mp.mpf(0)
        for _ in range(100):
            mid = (lo + hi) / 2
            val = f(mid)
            if abs(val) < tol:
                return mid
            if val > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def rhs(eps):
        chi = solve_chi(eps)
        return mp.sqrt(2) * (mp.sqrt(chi*chi + eps) - chi)

    if rhs(0) >= target:
        return 0.0

    lo, hi = mp.mpf(0), mp.mpf(1)
    while rhs(hi) < target:
        hi *= 2
        if hi > max_eps:
            raise RuntimeError("Could not bracket epsilon. noise_multiplier likely too small for given delta.")

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if rhs(mid) >= target:
            hi = mid
        else:
            lo = mid
    
    chi_hi = solve_chi(hi)
    print('Final chi is: ', chi_hi)
    print('Error compared with 2 delta: ',mp.erfc(chi_hi) - mp.e**(hi) * mp.erfc(mp.sqrt(chi_hi*chi_hi + hi)) - 2*delta)
    multiplier_recip = mp.sqrt(2) * (mp.sqrt(chi_hi*chi_hi + hi) - chi_hi)
    print('Verify noise multiplier: ', 1/multiplier_recip)

    return float(hi)

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute the overall ε and δ for the whole mechanism given subsampling rate ps, threshold t, factor of HH-ε, high-prob sensitivity, gaussian noise multiplier, gaussian δ."
    )
    p.add_argument("--ps", type=float, required=True,
                   help="client subsampling rate for heavy-hitter histogram.")
    p.add_argument("--HH_eps_factor", type=float, required=True,
                   help="ratio between the heavy-hitter histogram ε and log(e) 1/(1-ps).")
    p.add_argument("--HH_thres", type=float, required=True,
                   help="cut-off threshold for HH estimation.")
    p.add_argument("--K", type=int, required=True,
                   help="dimension of the projected space.")
    p.add_argument("--sens_to_r", type=float, required=True,
                   help=f"the sensitivity of sum estimation, bin size is 2*r/sqrt(K).")
    p.add_argument("--noise_multiplier", type=float, required=True,
                   help="Noise multiplier (σ / S).")
    p.add_argument("--delta", type=float, required=True, help="overall δ in (0, 0.5).")
    p.add_argument("--tol", type=float, default=1e-10, help="Bisection tolerance.")
    p.add_argument("--precision", type=int, default=50, help="mpmath decimal precision.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    epsilon_HH = args.HH_eps_factor * np.log(1/(1-args.ps)) / np.log(np.e)
    q = 1- (1-args.ps)**(args.HH_eps_factor+1)
    kl_ber = kl_bernoulli(q, args.ps)
    delta_HH = np.exp(-args.HH_thres / q * kl_ber)    # nats
    print(f"============================")
    print(f"HH:")
    print(f"with subsampling rate ps={args.ps}, cut-off threshold={args.HH_thres}")
    print(f"we set the ratio between epsilon_HH and log(1/(1-ps)) to {args.HH_eps_factor}")
    print(f"we obtain epsilon (heavy hitter) = {epsilon_HH}")
    print(f"we obtain delta (heavy hitter) = {delta_HH}")
    print(f"============================")
    print(f"Projection:")
    print(f"in the {args.K}-dimensional space, bucket size is 2r / sqrt({args.K}).")
    print(f"sensitivity is {args.sens_to_r} times r.")
    f_input = 2 / args.sens_to_r
    err_prob = (f(f_input)) ** args.K 
    # ** (100*args.ps-1)
    print(f"error probability is f({f_input}) to the power of {args.K}")
    print(f"delta (sensitivity) = {err_prob}")
    print(f"============================")
    print(f"Gaussian:")
    gauss_eps = analytic_gaussian_epsilon(
        noise_multiplier=args.noise_multiplier,
        delta=args.delta-delta_HH-err_prob,
        tol=args.tol,
        mp_dps=args.precision,
    )
    print(f"Hence, with noise multiplier {args.noise_multiplier}, and target delta (Gaussian) = {args.delta-delta_HH-err_prob}")
    print(f"we obtain epsilon (Gaussian) = {gauss_eps}")
    print(f"The noise std is {args.noise_multiplier * args.sens_to_r} times r")
    print(f"==========================")
    print(f"Overal eps = {epsilon_HH+gauss_eps}, overall delta={args.delta}")


"""
K=10
python end_to_end_DP_analysis.py --ps 0.15 --HH_eps_factor 5 --HH_thres=15 --K 10 --sens_to_r 4 --noise_multiplier 22.0 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.2 --HH_eps_factor 5 --HH_thres=20 --K 10 --sens_to_r 4 --noise_multiplier 4.71 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres=30 --K 10 --sens_to_r 4 --noise_multiplier 1.78 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres=50 --K 10 --sens_to_r 4 --noise_multiplier .95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres=60 --K 10 --sens_to_r 4 --noise_multiplier .43 --delta 1e-6

sampling noise
--------------
0.15 44.0
0.2 9.42
0.3 3.55
0.5 1.95
0.6 0.86


K=15
python end_to_end_DP_analysis.py --ps 0.15 --HH_eps_factor 5 --HH_thres=15 --K 15 --sens_to_r 3.5 --noise_multiplier 22.4 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.2 --HH_eps_factor 5 --HH_thres=20 --K 15 --sens_to_r 3.5 --noise_multiplier 4.72 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres=30 --K 15 --sens_to_r 3.5 --noise_multiplier 1.77 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres=50 --K 15 --sens_to_r 3.5 --noise_multiplier .95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres=60 --K 15 --sens_to_r 3.5 --noise_multiplier .43 --delta 1e-6

sampling noise
--------------
0.15 38.5
0.2 8.26
0.3 3.1
0.5 1.66
0.6 0.753

K=20
python end_to_end_DP_analysis.py --ps 0.15 --HH_eps_factor 5 --HH_thres=15 --K 20 --sens_to_r 2.4 --noise_multiplier 22.4 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.2 --HH_eps_factor 5 --HH_thres=20 --K 20 --sens_to_r 2.4 --noise_multiplier 4.72 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres=30 --K 20 --sens_to_r 2.4 --noise_multiplier 1.78 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres=50 --K 20 --sens_to_r 2.4 --noise_multiplier .95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres=60 --K 20 --sens_to_r 2.4 --noise_multiplier .43 --delta 1e-6

sampling noise
--------------
0.15 26.88
0.2 5.66
0.3 2.136
0.5 1.14
0.6 0.516

K=25
python end_to_end_DP_analysis.py --ps 0.15 --HH_eps_factor 5 --HH_thres=15 --K 25 --sens_to_r 2 --noise_multiplier 22.4 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.2 --HH_eps_factor 5 --HH_thres=20 --K 25 --sens_to_r 2 --noise_multiplier 4.72 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres=30 --K 25 --sens_to_r 2 --noise_multiplier 1.78 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres=50 --K 25 --sens_to_r 2 --noise_multiplier .95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres=60 --K 25 --sens_to_r 2 --noise_multiplier .43 --delta 1e-6

sampling noise
--------------
0.15 22.4
0.2 4.72
0.3 1.78
0.5 0.95
0.6 0.43

K=30
python end_to_end_DP_analysis.py --ps 0.15 --HH_eps_factor 5 --HH_thres=15 --K 30 --sens_to_r 1.6 --noise_multiplier 22.4 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.2 --HH_eps_factor 5 --HH_thres=20 --K 30 --sens_to_r 1.6 --noise_multiplier 4.72 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.3 --HH_eps_factor 4 --HH_thres=30 --K 30 --sens_to_r 1.6 --noise_multiplier 1.78 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.5 --HH_eps_factor 4 --HH_thres=50 --K 30 --sens_to_r 1.6 --noise_multiplier .95 --delta 1e-6
python end_to_end_DP_analysis.py --ps 0.6 --HH_eps_factor 3 --HH_thres=60 --K 30 --sens_to_r 1.6 --noise_multiplier .43 --delta 1e-6

sampling noise
--------------
0.15 17.92
0.2 3.78
0.3 1.424
0.5 0.76
0.6 0.344
"""