import math
from numba import jit

# @jit(nopython=True)
def factorial_moment(moment, dist):
    ans = 0

    for i in range(moment, len(dist)):
        ans += math.comb(i, moment) * dist[i]

    return ans * math.factorial(moment)


# @jit(nopython=True)
def reduced_factorial_moment(moment, dist):
    ans = 0

    for i in range(moment, len(dist)):
        ans += math.comb(i, moment) * dist[i]

    return ans


# @jit(nopython=True)
def factorial_moments(dist, n=3):
    moments = []
    for i in range(1, n + 1):
        moments.append(factorial_moment(i, dist))
    return moments


# @jit(nopython=True)
def reduced_factorial_moments(dist, n=3):
    moments = []
    for i in range(1, n + 1):
        moments.append(reduced_factorial_moment(i, dist))
    return moments


# @jit(nopython=True)
def bohnel_phi1(pf, nur1, pc=0):
    phi1 = (1 - pf - pc) / (1 - nur1 * pf)
    return phi1


# @jit(nopython=True)
def bohnel_phi2(pf, nur1, nur2, pc=0):
    phi2 = (bohnel_phi1(pf, nur1, pc) ** 2) * nur2 * pf / (1 - pf * nur1)
    return phi2


# @jit(nopython=True)
def bohnel_phi3(pf, nur1, nur2, nur3, pc=0):
    phi1 = bohnel_phi1(pf, nur1, pc)
    phi2 = bohnel_phi2(pf, nur1, nur2, pc)
    phi3 = (pf / (1 - nur1 * pf)) * (nur3 * phi1 * phi1 * phi1 + 3 * nur2 * phi1 * phi2)
    return phi3


# @jit(nopython=True)
def bohnel_sdt(pf, nur1, nur2, nur3, F=1, pc=0):
    phi1 = bohnel_phi1(pf, nur1, pc)
    phi2 = bohnel_phi2(pf, nur1, nur2, pc)
    phi3 = bohnel_phi3(pf, nur1, nur3, pc)

    s = F * phi1 / 1.0
    d = F * phi2 / 2.0
    t = F * phi3 / 6.0
    return s, d, t


def vpf_moments_recursive(F, pf, pfs, pc, pcs, nur1, nur2, nur3):
    s_moms = 0
    s_moms_prev = (1 - pf - pc) / (1 - pf * nur1)
    d_moms = 0
    d_moms_prev = (s_moms_prev**2) * pf * nur2 / (1 - pf * nur1)
    t_moms = 0
    t_moms_prev = nur3 * (s_moms_prev**3) + 3 * nur2 * s_moms_prev * d_moms_prev
    t_moms_prev *= pf / (1 - pf * nur1)

    for i in range(len(pfs)):
        s_moms = (1 - pfs[i] - pcs[i]) + pfs[i] * nur1 * s_moms_prev
        d_moms = pfs[i] * (nur2 * (s_moms_prev * s_moms_prev) + nur1 * d_moms_prev)
        t_moms = pfs[i] * (
            nur3 * (s_moms_prev**3)
            + 3 * nur2 * s_moms_prev * d_moms_prev
            + nur1 * t_moms_prev
        )

        s_moms_prev = s_moms
        d_moms_prev = d_moms
        t_moms_prev = t_moms

    s = F * s_moms
    d = F * (1 / 2.0) * d_moms
    t = F * (1 / 6.0) * t_moms

    return s, d, t


def vpf_phi1_r(pf_r, nur1, n=None):
    pf = pf_r[0]
    if n is None:
        n = len(pf_r)

    term1 = 1
    for i in range(n):
        term1 *= nur1 * pf_r[i]

    term2 = 0
    for i in range(n):
        term3 = 1 - pf_r[i]
        for j in range(i + 1, n):
            term3 *= nur1 * pf_r[j]
        term2 += term3

    phi1 = term1 * bohnel_phi1(pf, nur1) + term2
    return phi1


def vpf_phi2_r(pf_r, nur1, nur2, n=None):
    pf = pf_r[0]
    if n is None:
        n = len(pf_r)

    term1 = 1
    for i in range(n):
        term1 *= nur1 * pf_r[i]

    term2 = 0
    for i in range(n):
        term3 = nur2 * pf_r[i] * (vpf_phi1_r(pf_r, nur1, n=i) ** 2)
        for j in range(i + 1, n):
            term3 *= nur1 * pf_r[j]
        term2 += term3

    phi2 = term1 * bohnel_phi2(pf, nur1, nur2) + term2
    return phi2


def vpf_phi3_r(pf_r, nur1, nur2, nur3, n=None):
    pf = pf_r[0]
    if n is None:
        n = len(pf_r)

    term1 = 1
    for i in range(n):
        term1 *= nur1 * pf_r[i]

    term2 = 0
    for i in range(n):
        term3 = nur3 * pf_r[i] * (vpf_phi1_r(pf_r, nur1, n=i) ** 3)
        term3 += (
            3
            * nur2
            * pf_r[i]
            * vpf_phi1_r(pf_r, nur1, n=i)
            * vpf_phi2_r(pf_r, nur1, nur2, n=i)
        )
        for j in range(i + 1, n):
            term3 *= nur1 * pf_r[j]
        term2 += term3

    phi3 = term1 * bohnel_phi3(pf, nur1, nur2, nur3) + term2
    return phi3


def vpf_s_r(F, pf_r, nur1):
    phi1 = vpf_phi1_r(pf_r, nur1)
    return F * phi1 / 1.0


def vpf_d_r(F, pf_r, nur1, nur2):
    phi2 = vpf_phi2_r(pf_r, nur1, nur2)
    return F * phi2 / 2.0


def vpf_t_r(F, pf_r, nur1, nur2, nur3):
    phi3 = vpf_phi3_r(pf_r, nur1, nur2, nur3)
    return F * phi3 / 6.0


def vpf_phi1(pf, nur1, q=0):
    n = len(pf)

    term1 = 1
    for i in range(q, n):
        term1 *= nur1 * pf[i]

    term2 = 0
    for i in range(q, n):
        term3 = 1 - pf[i]
        for j in range(q, i):
            term3 *= nur1 * pf[j]
        term2 += term3

    phi1 = term1 * bohnel_phi1(pf[-1], nur1) + term2
    return phi1


def vpf_phi2(pf, nur1, nur2, q=0):
    n = len(pf)

    term1 = 1
    for i in range(q, n):
        term1 *= nur1 * pf[i]

    term2 = 0
    for i in range(q, n):
        term3 = nur2 * pf[i] * (vpf_phi1(pf, nur1, i + 1) ** 2)
        for j in range(q, i):
            term3 *= nur1 * pf[j]
        term2 += term3

    phi2 = term1 * bohnel_phi2(pf[-1], nur1, nur2) + term2
    return phi2


def vpf_phi3(pf, nur1, nur2, nur3, q=0):
    n = len(pf)

    term1 = 1
    for i in range(q, n):
        term1 *= nur1 * pf[i]

    term2 = 0
    for i in range(q, n):
        term3 = nur3 * pf[i] * (vpf_phi1(pf, nur1, i + 1) ** 3)
        term3 += (
            3
            * nur2
            * pf[i]
            * vpf_phi1(pf, nur1, i + 1)
            * vpf_phi2(pf, nur1, nur2, i + 1)
        )
        for j in range(q, i):
            term3 *= nur1 * pf[j]
        term2 += term3

    phi3 = term1 * bohnel_phi3(pf[-1], nur1, nur2, nur3) + term2
    return phi3


def vpf_s(F, pf, nur1):
    phi1 = vpf_phi1(pf, nur1)
    return F * phi1 / 1.0


def vpf_d(F, pf, nur1, nur2):
    phi2 = vpf_phi2(pf, nur1, nur2)
    return F * phi2 / 2.0


def vpf_t(F, pf, nur1, nur2, nur3):
    phi3 = vpf_phi3(pf, nur1, nur2, nur3)
    return F * phi3 / 6.0
