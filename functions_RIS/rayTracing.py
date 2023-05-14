"""
Created on November 2, 2021

@author: Sara Garcia Sanchez

This code uses ray tracing to determine the received signal for a given scenario.
"""

from functions_RIS import Scenario
import math
import numpy as np

__all__ = ["dist_points", "dist_chmatrix", "dist_to_reflector", "get_los_ch", "get_reflector_ch", "create_ris", "get_total_ch",
           "get_total_ch_gain", "get_ch_pow", "get_ch_pow_mrc", "get_ch_pow_no_mrc", "set_ris_weights"]

c = 3e8


def dist_points(point1, point2):      # Returns euclidean distance between two points
    eucl_dist = math.sqrt(math.pow(point1[0]-point2[0], 2) + math.pow(point1[1]-point2[1], 2) +
                          math.pow(point1[2]-point2[2], 2))
    return eucl_dist


def dist_chmatrix(tx_coord, rx_coord):      # Returns a matrix of euclidean distance between columns of two matrices
    ch_matrix_dist = np.zeros((len(rx_coord[0]), len(tx_coord[0])))
    for i in range(0, len(rx_coord[0])):
        for j in range(0, len(tx_coord[0])):
            ch_matrix_dist[i, j] = dist_points(rx_coord[:, i], tx_coord[:, j])
    return ch_matrix_dist


# Returns a 2D matrix of euclidean distance. The matrix dist_reflector[:,:,0] contains the distance between Tx and
# reflector for each Tx-Rx pair. The matrix dist_reflector[:,:,1] contains the distance between Rx and reflector
# for each Tx-Rx pair. OK.
def dist_to_reflector(tx_coord, rx_coord, reflection_yaxis, reflection_xaxis):
    dist_reflector = np.zeros((len(rx_coord[0]), len(tx_coord[0]), 2))
    for i in range(0, len(rx_coord[0])):
        for j in range(0, len(tx_coord[0])):
            dist_reflector[i, j, 0] = dist_points(tx_coord[:, j], (reflection_xaxis, reflection_yaxis[i, j], 0))
            dist_reflector[i, j, 1] = dist_points(rx_coord[:, i], (reflection_xaxis, reflection_yaxis[i, j], 0))
    # print(dist_reflector[:, :, 0])
    # print(dist_reflector[:, :, 1])
    return dist_reflector


def get_fspl_ch(dist_ch, freq):  # Returns a matrix of FSPL given a matrix of distances (nat)
    fspl_ch = 1/np.power((4*math.pi*freq/c)*dist_ch, 2)
    return fspl_ch


def get_phase_ch(dist_ch, freq):  # Returns a matrix of phase shift given a matrix of distances (rad)
    d_int_lambdas = np.floor(dist_ch*freq/c)  # Number of complete wavelengths within dist_ch
    d_frc_lambdas = dist_ch - (d_int_lambdas*c/freq)  # Remaining distance (fraction of wavelength)
    phase_ch = (2*math.pi*freq/c)*d_frc_lambdas  # Phase for the remaining distance
    return phase_ch


def get_los_ch(tx_coord, rx_coord, freq):  # Returns a matrix of LoS path loss (nat)
    dist_los_ch = dist_chmatrix(tx_coord, rx_coord)  # Matrix of Tx-Rx distance
    fspl_los_ch = get_fspl_ch(dist_los_ch, freq)
    ph_los_ch = get_phase_ch(dist_los_ch, freq)
    los_ch = fspl_los_ch * np.exp(1j * ph_los_ch)
    return los_ch


def get_reflector_ch(tx_coord, rx_coord, freq, reflector_type, polarization, reflection_yaxis_1,
                     reflection_yaxis_2, envsize, tx_loc_center_y, ris_elem_azi, ris_elem_ele, wavelength, d_min_ff,
                     ris_w, np_reflector_size):
    if polarization == 'vertical':
        phase_shift = 0  # Assuming reflection on PEC (Reflection coefficient = 1)
    elif polarization == 'horizontal':
        phase_shift = math.pi  # Assuming reflection on PEC (Reflection coefficient = -1)
    # Reflector 2: If nonRIS np_reflector_size can be set. If RIS np_reflector_size is always 'sub-wavelength'
    if np_reflector_size == 'inf':
        reflection_xaxis_2 = 2 * envsize  # We are locating reflector 2 in x = 2envSize
        d_reflec_2 = dist_to_reflector(tx_coord, rx_coord, reflection_yaxis_2, reflection_xaxis_2)
        d_reflec_2 = d_reflec_2[:, :, 0] + d_reflec_2[:, :, 1]  # Total reflection distance Tx-noRIS-Rx (ref below*)
        fspl_ref_ch_2 = get_fspl_ch(d_reflec_2, freq)
        phase_ref_ch_2 = get_phase_ch(d_reflec_2, freq) + phase_shift
        reflector_ch_2 = fspl_ref_ch_2 * np.exp(1j * phase_ref_ch_2)
    elif np_reflector_size == 'sub-wavelength':
        reflection_xaxis_2 = 2 * envsize  # We are locating reflector 2 in x = 2envSize
        d_reflec_2 = dist_to_reflector(tx_coord, rx_coord, reflection_yaxis_2, reflection_xaxis_2)
        ref_tx_dist = d_reflec_2[:, :, 0]  # Distances from Tx to RIS element r for all tx-rx pairs
        ref_rx_dist = d_reflec_2[:, :, 1]  # Distances from RIS element r to Rx for all tx-rx pairs
        ref_tx_ch = (get_fspl_ch(ref_tx_dist, freq)) * np.exp(1j * (get_phase_ch(ref_tx_dist, freq) + phase_shift))
        ref_rx_ch = (get_fspl_ch(ref_rx_dist, freq)) * np.exp(1j * (get_phase_ch(ref_rx_dist, freq) + phase_shift))
        reflector_ch_2 = np.multiply(ref_tx_ch, ref_rx_ch)  # Channel for all tx-rx, RIS antenna element r
    if reflector_type == 'noRIS':  # If nonRIS np_reflector_size can be set.
        if np_reflector_size == 'inf':
            reflection_xaxis_1 = 0  # We are locating reflector 1 in x = 0
            d_reflec_1 = dist_to_reflector(tx_coord, rx_coord, reflection_yaxis_1, reflection_xaxis_1)
            d_reflec_1 = d_reflec_1[:, :, 0] + d_reflec_1[:, :, 1]  # Total reflection distance Tx-noRIS-Rx
            fspl_ref_ch_1 = get_fspl_ch(d_reflec_1, freq)
            phase_ref_ch_1 = get_phase_ch(d_reflec_1, freq)+phase_shift
            reflector_ch_1 = fspl_ref_ch_1 * np.exp(1j * phase_ref_ch_1)
        elif np_reflector_size == 'sub-wavelength':
            reflection_xaxis_1 = 0  # We are locating reflector 1 in x = 0
            d_reflec_1 = dist_to_reflector(tx_coord, rx_coord, reflection_yaxis_1, reflection_xaxis_1)
            ref_tx_dist = d_reflec_1[:, :, 0]  # Distances from Tx to RIS element r for all tx-rx pairs
            ref_rx_dist = d_reflec_1[:, :, 1]  # Distances from RIS element r to Rx for all tx-rx pairs
            ref_tx_ch = (get_fspl_ch(ref_tx_dist, freq)) * np.exp(1j * (get_phase_ch(ref_tx_dist, freq) + phase_shift))
            ref_rx_ch = (get_fspl_ch(ref_rx_dist, freq)) * np.exp(1j * (get_phase_ch(ref_rx_dist, freq) + phase_shift))
            reflector_ch_1 = np.multiply(ref_tx_ch, ref_rx_ch)  # Channel for all tx-rx, RIS antenna element r
    elif reflector_type == 'RIS':  # RIS is only set to reflector 1. Reflector 2 is still noRIS.
        ris_center_x = 0  # We are locating center of RIS (reflector 1) in x = 0
        ris_center_y = tx_loc_center_y / 2
        ris1_coord = create_ris(ris_center_x, ris_center_y, ris_elem_azi, ris_elem_ele, wavelength, d_min_ff, envsize)
        ris_ch = np.zeros((len(ris1_coord[0]), len(rx_coord[0]), len(tx_coord[0])), dtype=complex)  # Assuming 2D RIS.
        for r in range(len(ris1_coord[0])):
            # Return a n_rx, n_tx matrix where [:,:,0] has distances from all Tx to RIS element r for all tx-rx pairs
            # and [:,:,1] the distance from RIS element r to the rx for all tx-rx pairs.
            d_tx_rx_r = dist_to_reflector(tx_coord, rx_coord, (ris1_coord[1, r])*np.ones((len(rx_coord[0]), len(tx_coord[0]))), ris_center_x)
            ris_tx_dist = d_tx_rx_r[:, :, 0]  # Distances from Tx to RIS element r for all tx-rx pairs
            ris_rx_dist = d_tx_rx_r[:, :, 1]  # Distances from RIS element r to Rx for all tx-rx pairs
            ris_tx_ch = (get_fspl_ch(ris_tx_dist, freq)) * np.exp(1j * (get_phase_ch(ris_tx_dist, freq)+phase_shift))
            ris_rx_ch = (get_fspl_ch(ris_rx_dist, freq)) * np.exp(1j * (get_phase_ch(ris_rx_dist, freq)+phase_shift))
            ris_ch[r, :, :] = ris_w[r]*(np.multiply(ris_tx_ch, ris_rx_ch))  # Channel for all tx-rx, RIS antenna element r

        reflector_ch_1 = np.sum(ris_ch, 0)  # Channel for all tx-rx with contributions from all RIS antenna elements
    return reflector_ch_1, reflector_ch_2


# Creates a matrix of dimension 4: coord_x = reflection_xaxis_1, coord_y = reflection_yaxis_1 +- antenna
# element separation, coord_z = 0, antenna_n with N antenna elements conforming the RIS
def create_ris(ris_center_x, ris_center_y, ris_elem_azi, ris_elem_ele, wavelength, d_min_ff, envsize):
    ris_coord = np.zeros((3, ris_elem_azi*ris_elem_ele))
    if (ris_elem_azi % 2) == 0:
        ris_coord_azi = np.linspace(-((wavelength / 4) + (((ris_elem_azi / 2) - 1) * (wavelength / 2))),
                                  ((wavelength / 4) + (((ris_elem_azi / 2) - 1) * (wavelength / 2))),
                                  num=ris_elem_azi)
    else:
        ris_coord_azi = np.linspace(-((ris_elem_azi - 1) / 2) * (wavelength / 2),
                                    ((ris_elem_azi - 1) / 2) * (wavelength / 2),
                                    num=ris_elem_azi)

    ris_coord[0, :] = np.ones(ris_elem_azi*ris_elem_ele)*ris_center_x
    ris_coord[1, :] = np.ones(ris_elem_azi*ris_elem_ele)*ris_center_y + ris_coord_azi
    ris_coord[2, :] = np.zeros(ris_elem_azi*ris_elem_ele)
    if ris_coord[1, -1] > envsize * 2 or ris_coord[1, 0] < d_min_ff:
        print('WARNING:  RIS is not entirely located within the far field region OR its size is greater than the '
              'environment size [0, 2*envSize]. Consider increasing envSize.')
        exit()
    return ris_coord


def get_total_ch(tx_coord, rx_coord, freq, reflector_type, polarization, reflection_yaxis_1,
                 reflection_yaxis_2, envsize, alpha_los, alpha_rfl_1, alpha_rfl_2, tx_att, tx_loc_center_y, ris_elem_azi,
                 ris_elem_ele, wavelength, d_min_ff, ris_w, np_reflector_size):
    los_ch = get_los_ch(tx_coord, rx_coord, freq)  # LoS channel matrix
    reflector_ch_1, reflector_ch_2 = get_reflector_ch(tx_coord, rx_coord, freq, reflector_type, polarization,
                                                      reflection_yaxis_1, reflection_yaxis_2, envsize, tx_loc_center_y,
                                                      ris_elem_azi, ris_elem_ele, wavelength, d_min_ff, ris_w,
                                                      np_reflector_size)  # Reflection channel matrix
    total_ch = alpha_los*np.multiply(tx_att, los_ch) + alpha_rfl_1*np.multiply(tx_att, reflector_ch_1) + \
               alpha_rfl_2*np.multiply(tx_att, reflector_ch_2)  # Total channel matrix: Interference between LoS and reflection
    destructive_int1 = 0
    destructive_int2 = 0
    if (abs(np.angle(reflector_ch_1[0, 0]) - np.angle(los_ch[0, 0])) >= math.pi/2) and\
            (abs(np.angle(reflector_ch_1[0, 0]) - np.angle(los_ch[0, 0])) < 3*math.pi/2):
        destructive_int1 = 1
    if (abs(np.angle(reflector_ch_2[0, 0]) - np.angle(los_ch[0, 0])) >= math.pi/2) and\
            (abs(np.angle(reflector_ch_2[0, 0]) - np.angle(los_ch[0, 0])) < 3*math.pi/2):
        destructive_int2 = 1
    return total_ch, destructive_int1, destructive_int2


def get_total_ch_gain(total_ch, tx_gain, rx_gain, tx_antenna_gain, rx_antenna_gain):
    total_gain = 10 ** ((tx_gain + rx_gain + tx_antenna_gain + rx_antenna_gain)/10)
    total_ch_gain = total_gain * total_ch
    return total_ch_gain


# The next code sections performs MRC
def get_ch_pow(total_ch_ite_rx):  # Calculate the power as the squared norm of h
    ch_pow = np.square(np.abs(total_ch_ite_rx))
    if len(ch_pow[0]) != 1:  # If multiple rx, norm over all rx signals
        ch_pow_ite = np.sum(ch_pow, 1)
        return ch_pow_ite
    else:
        return ch_pow


def get_ch_pow_mrc(total_ch_ite_rx_mrc):  # Calculate the power as the squared norm of h. Not used unless multi-rx
    ch_pow_mrc = total_ch_ite_rx_mrc
    if len(ch_pow_mrc[0]) != 1:  # If multiple rx, norm over all rx signals
        ch_pow_ite_mrc = np.mean(ch_pow_mrc, 1)
        return ch_pow_ite_mrc
    else:
        return ch_pow_mrc


def get_ch_pow_no_mrc(total_ch_ite_rx):  # Calculate the power as the squared norm of h
    ch_pow = np.square(np.abs(np.mean(total_ch_ite_rx,1)))
    return ch_pow


# Returns a matrix with all RIS antenna element weights in each column and as many rows as configs.
def set_ris_weights(ris_elem_azi, ris_n_config, RIS_typeConfig):  # Calculate the power as the squared norm of h
    w_mag = np.ones((ris_n_config, ris_elem_azi))  # Assuming 2D RIS. Magnitude of RIS weights for all elements
    if RIS_typeConfig == 'equally_spaced':
        res_azi = 2 * math.pi / ris_n_config  # Angular resolution in azimuth
        w_angle_vec = np.linspace(0, 2 * math.pi - res_azi, ris_n_config)  # Vector of equally spaced phase in angle
        w_angle = np.transpose(np.tile(w_angle_vec, (ris_elem_azi, 1)))  # Matrix to set all antenna element to the same value for a given config
        print(w_angle)
    elif RIS_typeConfig == 'random':
        w_angle = 2 * math.pi * (np.random.rand(ris_n_config, ris_elem_azi)) # Random phase
        print(w_angle)
    print(w_angle)
    ris_all_configs = np.multiply(w_mag, np.exp(1j * w_angle))
    return ris_all_configs

# REF https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8936989
# For a nonRIS reflector, an infinitely large IRS or in the near-field, fspl goes with d+r with d the distance between
#   Tx and reflection point and r the distance between reflection point and rx
# In the far field or for a small RIS, the fspl dependency goes with d*r
