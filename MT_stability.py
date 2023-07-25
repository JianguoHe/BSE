import numpy as np
from utils import conditional_njit
from const import mass_accretion_model


@conditional_njit()
def MT_stability_MS(kw_donor, mass1i, mass2i, tbi):
    # Shao & Li 2014, doi:10.1088/0004-637X/796/1/37
    if mass_accretion_model == 1:
        return MT_MS_stability_model_1(kw_donor, mass1i, mass2i, tbi)
    elif mass_accretion_model == 2:
        return MT_MS_stability_model_2(kw_donor, mass1i, mass2i, tbi)
    elif mass_accretion_model == 3:
        return MT_MS_stability_model_3(kw_donor, mass1i, mass2i, tbi)
    else:
        raise ValueError('Please provide an allowed scheme of mass_accretion_model.')


@conditional_njit()
def MT_MS_stability_model_1(kw_donor, mass1i, mass2i, tbi):
    qc = 3
    q1 = mass1i / mass2i
    if kw_donor in {0, 1, 2} or (kw_donor == 4 and mass1i >= 12.0):
        porb15 = 0.548 + 0.0945 * mass1i - 0.001502 * mass1i ** 2 + 1.0184e-5 * mass1i ** 3 - 6.2267e-8 * mass1i ** 4
        porb2 = 0.1958 + 0.3278 * mass1i - 0.01159 * mass1i ** 2 + 0.0001708 * mass1i ** 3 - 9.55e-7 * mass1i ** 4
        porb25 = 6.0143 + 0.01866 * mass1i - 0.0009386 * mass1i ** 2 - 3.709e-5 * mass1i ** 3 + 5.9106e-7 * mass1i ** 4
        porb3 = 24.6 - 1.85 * mass1i + 0.0784 * mass1i ** 2 - 0.0015 * mass1i ** 3 + 1.024e-5 * mass1i ** 4
        if mass1i <= 16.0:
            porb35 = 1772.8 - 551.48 * mass1i + 66.14 * mass1i ** 2 - 3.527 * mass1i ** 3 + 0.07 * mass1i ** 4
        else:
            porb35 = 82.17 - 5.697 * mass1i + 0.17297 * mass1i ** 2 - 0.00238 * mass1i ** 3 + 1.206e-5 * mass1i ** 4
        if mass1i <= 16.0:
            porb4 = 46511.2 - 10670 * mass1i + 919.88 * mass1i ** 2 - 35.25 * mass1i ** 3 + 0.506 * mass1i ** 4
        else:
            porb4 = 153.03 - 8.967 * mass1i + 0.2077 * mass1i ** 2 - 0.00204 * mass1i ** 3 + 6.677e-6 * mass1i ** 4
        if mass1i <= 24.0:
            porb5 = 86434.6 - 15494.3 * mass1i + 1041.4 * mass1i ** 2 - 31.017 * mass1i ** 3 + 0.345 * mass1i ** 4
        else:
            porb5 = 566.5 - 33.123 * mass1i + 0.7589 * mass1i ** 2 - 0.00776 * mass1i ** 3 + 2.94e-5 * mass1i ** 4
        if mass1i <= 40.0:
            porb6 = 219152.8 - 24416.3 * mass1i + 1018.7 * mass1i ** 2 - 18.834 * mass1i ** 3 + 0.1301 * mass1i ** 4
        else:
            porb6 = -10744.14 + 856.43 * mass1i - 24.834 * mass1i ** 2 + 0.3147 * mass1i ** 3 - 0.00148 * mass1i ** 4

        if tbi <= porb15:
            qc = 1e4 if q1 <= 1.5 else 0
        elif tbi <= porb2:
            qc = 1e4 if q1 <= 2 else 0
        elif tbi <= porb25:
            qc = 1e4 if q1 <= 2.5 else 0
        elif tbi <= porb3:
            qc = 1e4 if q1 <= 3 else 0
        elif tbi <= porb35:
            qc = 1e4 if q1 <= 3.5 else 0
        elif tbi <= porb4:
            qc = 1e4 if q1 <= 4 else 0
        elif tbi <= porb5:
            qc = 1e4 if q1 <= 5 else 0
        elif tbi <= porb6:
            qc = 1e4 if q1 <= 6 else 0
        else:
            qc = 1e4 if q1 <= 6 else 0
    return qc


@conditional_njit()
def MT_MS_stability_model_2(kw_donor, mass1i, mass2i, tbi):
    qc = 0
    q1 = mass1i / mass2i
    if kw_donor in {0, 1}:
        porb2 = 0.674 + 0.2149 * mass1i - 0.009817 * mass1i ** 2 + 0.000194 * mass1i ** 3 - 1.365e-6 * mass1i ** 4
        if tbi <= porb2:
            qc = 0.0
        elif mass1i <= 25:
            qc = 1.0e4 if q1 <= 2.0 else 0.0
        elif mass1i <= 35:
            qc = 1.0e4 if q1 <= 2.2 else 0.0
        else:
            qc = 1.0e4 if q1 <= 2.5 else 0.0
    elif kw_donor in {2, 4}:
        if q1 <= 1.8:
            qc = 1.0e4
        elif q1 >= 3.0:
            qc = 0.0
        elif 1.8 < q1 <= 2.8:
            if mass1i <= 10.0 or mass1i >= 20.0:
                qc = 1.0e4
            else:
                qc = 0.0
        elif 2.2 < q1 <= 2.5:
            if mass1i <= 7.0 or mass1i >= 45.0:
                qc = 1.0e4
            else:
                qc = 0.0
        elif 2.8 < q1 < 3.0:
            if mass1i > 45.0:
                qc = 1.0e4
            else:
                qc = 0.0
    return qc


@conditional_njit()
def MT_MS_stability_model_3(kw_donor, mass1i, mass2i, tbi):
    qc = 0
    q1 = mass1i / mass2i
    if kw_donor in {0, 1}:
        if mass1i <= 20.0:
            porb2 = -0.2425 + 0.9489 * mass1i - 0.1448 * mass1i ** 2 + 0.009118 * mass1i ** 3 - 0.0001953 * mass1i ** 4
        else:
            porb2 = -8.3961 + 1.1504 * mass1i - 0.04213 * mass1i ** 2 + 0.0006666 * mass1i ** 3 - 3.689e-6 * mass1i ** 4
        if tbi <= porb2:
            qc = 0.0
        elif mass1i <= 25:
            qc = 1.0e4 if q1 <= 1.6 else 0.0
        elif mass1i <= 30:
            qc = 1.0e4 if q1 <= 2.0 else 0.0
        elif mass1i <= 40:
            qc = 1.0e4 if q1 <= 2.5 else 0.0
        else:
            qc = 1.0e4 if q1 <= 3.0 else 0.0
    elif kw_donor in {2, 4}:
        if mass1i > 20.0:
            qc = 1.0e4 if q1 <= 1.2 else 0.0
        elif mass1i > 15.0:
            qc = 1.0e4 if q1 <= 1.4 else 0.0
        elif mass1i > 10.0:
            qc = 1.0e4 if q1 <= 1.5 else 0.0
        elif mass1i > 8.0:
            qc = 1.0e4 if q1 <= 1.6 else 0.0
        elif mass1i > 7.0:
            qc = 1.0e4 if q1 <= 1.8 else 0.0
        elif mass1i > 5.0:
            qc = 1.0e4 if q1 <= 2.0 else 0.0
        else:
            qc = 1.0e4 if q1 <= 2.2 else 0.0
    return qc
