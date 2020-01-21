from collections import namedtuple
from itertools import repeat
import warnings
import os.path as op
import numpy as np
from mlbspline import load
from lbftd import evalGibbs as eg
from lbftd.statevars import iP, iT

defpath = op.join(op.dirname(op.abspath(__file__)), 'SeaFreeze_Gibbs.mat')

def seafreeze(PT, phase, path=defpath):
    """ Calculates thermodynamic quantities for H2O water or ice polymorphs Ih, III, V, and VI for all phases
        (see lbftd documentation for full list)
        for solid phases only:
            - Vp (compressional wave velocity, in m/s)
            - Vs (shear wave velocity, in m/s)
            - shear (shear modulus, in MPa)
    Requires the SeaFreeze_Gibbs.mat library containing the Gibbs LBF parametrization (installed with this module).

    NOTE:  The authors recommend the use of 'water1' for any application in the 200-355 K range and up to 2300 MPa.
    The ice Gibbs parametrizations are optimized for use with the 'water1' phase for phase equilibrium calculations.
    Using other water parametrizations will lead to incorrect melting curves -- 'water2' and 'water_IAPWS95'
    parametrizations are provided for HP extension up to 100 GPa and comparison only.

    :param PT:      the pressure (MPa) and temperature (K) conditions at which the thermodynamic quantities should be
                    calculated -- the specified units are required, as conversions are built into several calculations.
                    This parameter can have one of the following formats:
                        - a 1-dimensional numpy array of tuples with one or more scattered (P,T) tuples, e.g.
                                PT = np.empty((3,), np.object)
                                PT[0] = (441.0858, 313.95)
                                PT[1] = (478.7415, 313.96)
                                PT[2] = (444.8285, 313.78)
                        - a numpy array with 2 nested numpy arrays, the first with pressures and the second
                          with temperatures -- each inner array must be sorted from low to high values
                          a grid will be constructed from the P and T arrays such that each row of the output
                          will correspond to a pressure and each column to a temperature, e.g.
                                P = np.arange(0.1, 1000.2, 10)
                                T = np.arange(240, 501, 2)
                                PT = np.array([P, T])
    :param phase:   one of the keys of the phases dict, indicating the phase of H2O to be evaluated
    :param path:    an optional path to the SeaFreeze_Gibbs.mat file
                    default value assumes the spline distributed along with the project
    :return:        object containing the calculated thermodynamic quantities (as named properties), as well as
                    a PTM property (a copy of PT)
    """
    try:
        phasedesc = phases[phase]
    except KeyError:
        raise ValueError('The specified phase is not recognized.  Supported phases are ' +
                         ', '.join(phases.keys())+'.')
    sp = load.loadSpline(path, phasedesc.sp_name)
    # calc density and isentropic bulk modulus
    isscatter = _is_scatter(PT)
    tdvs = _get_tdvs(sp, PT, isscatter)
    if phasedesc.shear_mod_parms:
        smg = _get_shear_mod_GPa(phasedesc.shear_mod_parms, tdvs.rho, _get_T(PT, isscatter))
        tdvs.shear = 1e3 * smg  # convert to MPa for consistency with other measures
        tdvs.Vp = _get_Vp(smg, tdvs.rho, tdvs.Ks)
        tdvs.Vs = _get_Vs(smg, tdvs.rho)
    return tdvs


def whichphase(PT, path=defpath):
    """ Determines the most likely phase of water at each pressure/temperature

    :param PT:      the pressure (MPa) and temperature (K) conditions at which the phase should be determined --
                    the specified units are required, as conversions are built into several calculations.
                    This parameter can have one of the following formats:
                        - a 1-dimensional numpy array of tuples with one or more scattered (P,T) tuples, e.g.
                                PT = np.empty((3,), np.object)
                                PT[0] = (441.0858, 313.95)
                                PT[1] = (478.7415, 313.96)
                                PT[2] = (444.8285, 313.78)
                        - a numpy array with 2 nested numpy arrays, the first with pressures and the second
                          with temperatures -- each inner array must be sorted from low to high values
                          a grid will be constructed from the P and T arrays such that each row of the output
                          will correspond to a pressure and each column to a temperature, e.g.
                                P = np.arange(0.1, 1000.2, 10)
                                T = np.arange(240, 501, 2)
                                PT = np.array([P, T])
    :param path:    an optional path to the SeaFreeze_Gibbs.mat file --
                    default value assumes the spline distributed along with the project
    :return:        A numpy.ndarray the same size as PT, where each row corresponds to a pressure,
                    and each column to a temperature.  The phase of each pressure/temperature point is
                    represented by an integer, as shown in phasenum2phase.

    """
    isscatter = _is_scatter(PT)
    phase_sp = _get_phase_splines(phases.values(), path)
    return _which_phase_internal(isscatter, phase_sp, PT, path)


def get_transition_line(reqph, prec=3, path=defpath):
    """
    Identifies the phase transition between two phases, to a given level of resolution, in terms of
    the pressure and temperature points at which one phase becomes more stable than the other.
    Only the P/T regimes where the two phases are contiguously the most stable of the supported
    phases are included in the output.

    :param reqph:       A list of length 2 indicating the phases for which the transition should be identified.
                        Values must be expressed as two distinct keys of the phases dict.
    :param prec:        Indicates the number of decimal places to which the pressure and temperatures
                        corresponding to the phase transition.  In other words, the output values
                        will be specified to precision corresponding to 10^-prec.
                        Accepted values range from 0 to 7.
                        Float values are accepted.  For example, prec=2.3 will result in getting values down
                        to ~ the nearest 0.005.
    :param path:        an optional path to the SeaFreeze_Gibbs.mat file --
                        default value assumes the spline distributed along with the project
    :return:            A list of 2-tuples indicating the (P,T) points corresponding to the transition
                        between the specified phases, if any.  Both pressure (P, in MPa) and
                        temperature (T, in K) will be specified to the requested precision.
                        The list will be empty if there is no phase boundary between the specified phases.
    """
    # check prec values
    if prec > 7 or prec < 0:
        raise ValueError('Unsupported precision')
    # confirm only 2 phases are requested and that all are supported
    if len(reqph) != 2 or len([r for r in reqph if r in phases and not np.isnan(phases[r].phase_num)]) != 2:
        sp = [p for p in phases.keys() if not np.isnan(phases[p].phase_num)]
        raise ValueError('Parameter reqph should be two values, both from supported values '+str(sp))
    phasesn = sorted([phases[p].phase_num for p in reqph]) # phase numbers to match with output
    # get smallest P and T ranges supported by both phases
    phase_sp = _get_phase_splines(phases.values(), path)
    getRangeFromKnots = lambda idx: [(phase_sp[sp]['knots'][idx][0], phase_sp[sp]['knots'][idx][-1])
                                     for sp in phase_sp.keys() if sp in phasesn]
    Prange = getRangeFromKnots(iP)
    Trange = getRangeFromKnots(iT)
    # initialize PT to appropriate precision for both P and T
    P = np.arange(max([pr[0] for pr in Prange]), min([pr[1] for pr in Prange])+1)
    T = np.arange(max([tr[0] for tr in Trange]), min([tr[1] for tr in Trange])+1)
    ptgrids = [np.array([P,T])]
    # iterate over increasing levels of precision
    isTransitionPt = lambda pnp: pnp[0] in phasesn and pnp[1] in phasesn and pnp[0] != pnp[1]
    getGridRange = lambda D, i, ngc, s, ns: np.arange(D[ngc[i]], D[ngc[i]] + s + ns, ns)
    getPt = lambda D, i, ngc: D[ngc[i]]
    for pr in np.arange(0, prec+1):
        nextgrids = []
        nextstep = 10 ** float(((pr + 1) * -1))
        step = 10 ** float(pr * -1)
        for pt in ptgrids:
            # get phases for each cell on P/T grid (first time through will be a grid, the other times will be scatter)
            phdiag = _which_phase_internal(False, phase_sp, pt, path)
            # for each P, find phase transitions between one temperature and the next
            ptranspt = list(np.where(np.apply_along_axis(isTransitionPt, 2, np.dstack((phdiag[:, :-1], phdiag[:, 1:])))))
            # for each T, find phase transitions between one pressure and the next
            ttranspt = list(np.where(np.apply_along_axis(isTransitionPt, 2, np.dstack((phdiag[:-1, :], phdiag[1:, :])))))
            # each (P,T) point will serve as the lower left corner of a new grid to be checked at the next level of precision
            nextgridcorners = sorted({(pi, ti) for pi, ti in zip(*np.concatenate((ptranspt, ttranspt), 1))})
            if pr != prec:      # build ptgrids for next iteration
                nextgrids = nextgrids + [np.array([getGridRange(pt[iP], iP, ngc, step, nextstep),
                                                   getGridRange(pt[iT], iT, ngc, step, nextstep)]) for ngc in nextgridcorners]
            else:               # map nextgridcorners for output
                nextgrids = nextgrids + [(pt[iP][ngc[iP]], pt[iT][ngc[iT]]) for ngc in nextgridcorners]
        if pr != prec:
            ptgrids = nextgrids
    return nextgrids


def _which_phase_internal(isscatter, phase_sp, PT, path):
    ptsh = ((PT.size,) if isscatter else (PT[0].size, PT[1].size))  # reference shape based on PT
    comp = np.full(ptsh + (max_phase_num + 1,), np.nan)  # comparison matrix
    for p in phase_sp.keys():
        sl = tuple(repeat(slice(None), 1 if isscatter else 2)) + (p,)  # slice for this phase
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = phase_sp[p]
            tdvs = _get_tdvs(sp, PT, isscatter, 'G').G
            # wipe out G for PT values that fall outside the knot sequence
            if isscatter:
                extrap = [(pt[iP] < sp['knots'][iP].min()) + (pt[iP] > sp['knots'][iP].max()) +
                          (pt[iT] < sp['knots'][iT].min()) + (pt[iT] > sp['knots'][iT].max()) for pt in PT]
            else:
                pt = np.logical_or(PT[iP] < sp['knots'][iP].min(), PT[iP] > sp['knots'][iP].max())
                tt = np.logical_or(PT[iT] < sp['knots'][iT].min(), PT[iT] > sp['knots'][iT].max())
                extrap = np.logical_or(*np.meshgrid(pt, tt, indexing='ij'))
            tdvs[extrap] = np.nan
            comp[sl] = tdvs
    # output for all-nan slices should be nan
    all_nan_sl = np.all(np.isnan(comp), -1)  # find slices where all values are nan along the innermost axis
    out = np.full(ptsh, np.nan)  # initialize output to nan
    out[~all_nan_sl] = np.nanargmin(comp[~all_nan_sl], -1)  # find min values for other slices
    return out

def _get_phase_splines(phases2load, path):
    return {v.phase_num: load.loadSpline(path, v.sp_name) for v in phases2load if not np.isnan(v.phase_num)}


def _get_tdvs(sp, PT, is_scatter, *tdvSpec):
    """ peeks into PT to see if the PT data is for grid or scatter and calls the appropriate evalGibbs function

    :param sp:          the Gibbs LBF
    :param PT:          the PT data
    :param is_scatter:  Boolean indicating whether the PT data is scatter or not (if not, it is a grid)
    :param tdvSpec:     optional list of thermodynamic variables to calculate (see lbftd documentation)
    :return:            tdv object (see lbftd documentation)
    """
    fn = eg.evalSolutionGibbsScatter if is_scatter else eg.evalSolutionGibbsGrid
    return fn(sp, PT, *tdvSpec, failOnExtrapolate=False)


def _get_shear_mod_GPa(sm, rho, T):
    return None if sm is None else sm[0] + sm[1]*(rho - sm[4]) + sm[2]*(rho-sm[4])**2 + sm[3]*(T-sm[5])


def _get_Vp(smg, rho, Ks):
    return 1e3 * np.sqrt((Ks/1e3 + 4/3*smg)/rho/1e-3)


def _get_Vs(smg, rho):
    return 1e3 * np.sqrt(smg/rho/1e-3)


def _is_scatter(PT):
    return isinstance(PT[0], tuple) or (PT.shape == (1,2) and np.isscalar(PT[0]) and np.isscalar(PT[1]))


def _get_T(PT, is_scatter):
    return np.array([T for P,T in PT]) if is_scatter else PT[1]


#########################################
## Constants
#########################################
PhaseDesc = namedtuple('PhaseDesc', 'sp_name shear_mod_parms phase_num')
phases = {"Ih": PhaseDesc("G_iceIh", [3.04, -0.00462, 0, -0.00607, 1000, 273.15], 1),  # Feistel and Wagner, 2006
          "II": PhaseDesc("G_iceII", [4.1, 0.0175, 0, -0.014, 1100, 273], 2),          # Journaux et al, 2019
          "III": PhaseDesc("G_iceIII", [2.57, 0.0175, 0, -0.014, 1100, 273], 3),       # Journaux et al, 2019
          "V": PhaseDesc("G_iceV", [2.57, 0.0175, 0, -0.014, 1100, 273], 5),           # Journaux et al, 2019
          "VI": PhaseDesc("G_iceVI", [2.57, 0.0175, 0, -0.014, 1100, 273], 6),         # Journaux et al, 2019
          "water1": PhaseDesc("G_H2O_2GPa_500K", None, 0),              # extends to 500 K and 2300 MPa; Bollengier et al 2019
          "water2": PhaseDesc("G_H2O_100GPa_10000K", None, np.nan),     # extends to 100 GPa; Brown 2018
          "water_IAPWS95": PhaseDesc("G_H2O_IAPWS", None, np.nan)       # LBF representation of IAPWS 95; Wagner and Pruß, 2002
          }
max_phase_num = max([p.phase_num for p in phases.values()])
phasenum2phase = {v.phase_num: k for (k,v) in phases.items() if not np.isnan(v.phase_num)}
