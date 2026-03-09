import numpy as np

proton_mass = 0.938272089 # GeV/c^2
neutron_mass = 0.939565421 # GeV/c^2

electron_mass = 0.000510998950 # GeV/c^2
muon_mass = 0.10565837 # GeV/c^2
charged_pion_mass = 0.139570 # GeV/c^2

# from https://www.sciencedirect.com/science/article/pii/0370157372900105?ref=cra_js_challenge&fr=RR-1 equation 3.18
# also see https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/LwlynSmithQELCCPXSec.cxx#L118
# ignores factors that are constant before and after reweighting
def relative_llewelyn_smith_CCQE_xs(Q2, FA, Enu, neutrino=True, lepton="mu"):

    if neutrino: # neutrino + neutron -> muon + proton
        nucleon_mass = proton_mass
        sign = -1
    else:
        nucleon_mass = neutron_mass
        sign = 1

    if lepton == "mu":
        lepton_mass = muon_mass
    else:
        lepton_mass = electron_mass

    A, B, C = ABC(Q2, FA, nucleon_mass, lepton_mass, neutrino)

    s_minus_u = 4 * Enu * nucleon_mass**2 + Q2 - muon_mass**2

    return A + sign * B * s_minus_u / nucleon_mass**2 + C * s_minus_u**2 / nucleon_mass**4


def ABC(Q2, FA, nucleon_mass, lepton_mass, neutrino):

    F1v, chiFv2, Fp = get_non_axial_form_factors(Q2, FA, nucleon_mass, neutrino)

    ml2 = lepton_mass**2
    M2 = nucleon_mass**2
    FA2 = FA**2
    Fp2 = Fp**2
    F1V2 = F1v**2
    xiF2V2 = chiFv2**2
    F1V = F1v * chiFv2
    xiF2V = chiFv2 * chiFv2
    q2_M2 = Q2 / M2


    A = (0.25 * (ml2 - Q2) / M2) * (
        (4 - q2_M2) * FA2 - (4 + q2_M2) * F1V2 - q2_M2 * xiF2V2 * (1 + 0.25 * q2_M2)
        - 4 * q2_M2 * F1V * xiF2V - (ml2 / M2) * (
            (F1V2 + xiF2V2 + 2 * F1V * xiF2V) + (FA2 + 4 * Fp2 + 4 * FA * Fp) + (q2_M2 - 4) * Fp2
        )
    )

    B = -1 * q2_M2 * FA * (F1V + xiF2V)
    C = 0.25 * (FA2 + F1V2 - 0.25 * q2_M2 * xiF2V2)

    return A, B, C


def get_non_axial_form_factors(Q2, FA, nucleon_mass, neutrino):

    # https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/LwlynSmithFF.cxx#L256
    tau = -Q2 / (4 * nucleon_mass**2)

    if neutrino: # neutrino + neutron -> muon + proton
        Ge, Gm = get_Gen_Gmn(Q2)
    else:
        Ge, Gm = get_Gep_Gmp(Q2)


    # https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/LwlynSmithFF.cxx#L146
    F1v = (Ge - tau * Gm) / (1 + tau)

    # https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/LwlynSmithFF.cxx#L156
    xiF2V = (Gm - Ge) / (1 - tau)

    # https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/LwlynSmithFF.cxx#L185
    Fp = 2. * nucleon_mass**2 * FA / (charged_pion_mass**2 - Q2)

    return F1v, xiF2V, Fp


# https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/BBA07ELFormFactorsModel.cxx#L61
fGepa1 = -0.24
fGepb1 = 10.98
fGepb2 = 12.82
fGepb3 = 21.97
fGepp1 = 1.0000
fGepp2 = 0.9927
fGepp3 = 0.9898
fGepp4 = 0.9975
fGepp5 = 0.9812
fGepp6 = 0.9340
fGepp7 = 1.0000

fGmpa1 = 0.1717
fGmpb1 = 11.26
fGmpb2 = 19.32
fGmpb3 = 8.33
fGmpp1 = 1.0000
fGmpp2 = 1.0011
fGmpp3 = 0.9992
fGmpp4 = 0.9974
fGmpp5 = 1.0010
fGmpp6 = 1.0003
fGmpp7 = 1.0000

fGenp1 = 1.0000
fGenp2 = 1.1011
fGenp3 = 1.1392
fGenp4 = 1.0203
fGenp5 = 1.1093
fGenp6 = 1.5429
fGenp7 = 0.9706

fGmnp1 = 1.0000
fGmnp2 = 0.9958
fGmnp3 = 0.9877
fGmnp4 = 1.0193
fGmnp5 = 1.0350
fGmnp6 = 0.9164
fGmnp7 = 0.7300

# https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/config/GEM21_11d/CommonParam.xml#L232
fMuP = 2.7930
fMuN = -1.913042


# https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/BBA07ELFormFactorsModel.cxx#L53

def get_Gep_Gmp(Q2):

    M2 = proton_mass**2
    t = Q2 / (4 * M2)
    xp = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / t))

    GEp = (1.0 + fGepa1 * t) / (1.0 + t * (fGepb1 + t * (fGepb2 + fGepb3 * t)))
    gep = get_An(xp, fGepp1, fGepp2, fGepp3, fGepp4, fGepp5, fGepp6, fGepp7) * GEp

    GMp = (1.0 + fGmpa1 * t) / (1.0 + t * (fGmpb1 + t * (fGmpb2 + fGmpb3 * t)))
    gmp = get_An(xp, fGmpp1, fGmpp2, fGmpp3, fGmpp4, fGmpp5, fGmpp6, fGmpp7) * GMp
    gmp *= fMuP
    return gep, gmp

def get_Gen_Gmn(Q2):

    M2 = neutron_mass**2
    t = Q2 / (4 * M2)
    xn = 2.0 / (1.0 + np.sqrt(1.0 + 1.0 / t))

    gep, gmp = get_Gep_Gmp(Q2)

    gen = get_An(xn, fGenp1, fGenp2, fGenp3, fGenp4, fGenp5, fGenp6, fGenp7) * gep * 1.7 * t / (1 + 3.3 * t)

    gmn = get_An(xn, fGmnp1, fGmnp2, fGmnp3, fGmnp4, fGmnp5, fGmnp6, fGmnp7) * gmp
    gmn *= fMuN / fMuP

    return gen, gmn


# https://github.com/GENIE-MC/Generator/blob/2084cc6b8f25a460ebf4afd6a4658143fa9ce2ff/src/Physics/QuasiElastic/XSection/BBA07ELFormFactorsModel.cxx#L161
def get_An(x, c1, c2, c3, c4, c5, c6, c7):
    d1 = (0.0-1.0/6)*(0.0-2.0/6)*(0.0-3.0/6)*(0.0-4.0/6)*(0.0-5.0/6)*(0.0-1.0)
    d2 = (1.0/6-0.0)*(1.0/6-2.0/6)*(1.0/6-3.0/6)*(1.0/6-4.0/6)*(1.0/6-5.0/6)*(1.0/6-1.0)
    d3 = (2.0/6-0.0)*(2.0/6-1.0/6)*(2.0/6-3.0/6)*(2.0/6-4.0/6)*(2.0/6-5.0/6)*(2.0/6-1.0)
    d4 = (3.0/6-0.0)*(3.0/6-1.0/6)*(3.0/6-2.0/6)*(3.0/6-4.0/6)*(3.0/6-5.0/6)*(3.0/6-1.0)
    d5 = (4.0/6-0.0)*(4.0/6-1.0/6)*(4.0/6-2.0/6)*(4.0/6-3.0/6)*(4.0/6-5.0/6)*(4.0/6-1.0)
    d6 = (5.0/6-0.0)*(5.0/6-1.0/6)*(5.0/6-2.0/6)*(5.0/6-3.0/6)*(5.0/6-4.0/6)*(5.0/6-1.0)
    d7 = (1.0-0.0)*(1.0-1.0/6)*(1.0-2.0/6)*(1.0-3.0/6)*(1.0-4.0/6)*(1.0-5.0/6)

    return (c1*        (x-1.0/6)*(x-2.0/6)*(x-3.0/6)*(x-4.0/6)*(x-5.0/6)*(x-1.0)/d1+
            c2*(x-0.0)*          (x-2.0/6)*(x-3.0/6)*(x-4.0/6)*(x-5.0/6)*(x-1.0)/d2+
            c3*(x-0.0)*(x-1.0/6)*          (x-3.0/6)*(x-4.0/6)*(x-5.0/6)*(x-1.0)/d3+
            c4*(x-0.0)*(x-1.0/6)*(x-2.0/6)*          (x-4.0/6)*(x-5.0/6)*(x-1.0)/d4+
            c5*(x-0.0)*(x-1.0/6)*(x-2.0/6)*(x-3.0/6)*          (x-5.0/6)*(x-1.0)/d5+
            c6*(x-0.0)*(x-1.0/6)*(x-2.0/6)*(x-3.0/6)*(x-4.0/6)*          (x-1.0)/d6+
            c7*(x-0.0)*(x-1.0/6)*(x-2.0/6)*(x-3.0/6)*(x-4.0/6)*(x-5.0/6)          /d7)