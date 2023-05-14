"""
Created on November 8, 2021

@author: Sara Garcia Sanchez

This code contains metrics to evaluate spatial diversity in a given channel.
"""

import numpy as np
from numpy import linalg

__all__ = ["get_singval", "get_cond_number", "get_rank", "get_xcross", "get_xcross_err", "get_xcross_nrm",
           "get_xcross_div", "get_chard", "get_chard_mean"]


def get_singval(total_ch):  # Returns singular values and vectors of a matrix
    u, s, vh = linalg.svd(total_ch)  # U: left s-vec in columns, Vh: right s-vec in rows, s: s-val in decreasing order
    s_norm = s/s[0]
    return s, s_norm


def get_cond_number(singval):  # Returns condition number given a singular value diagonal matrix T (UTV^H)
    cond_number = singval[0]/singval[-1]
    return cond_number


def get_rank(singval, rank_tol):  # Returns rank given a singular value diagonal matrix T (UTV^H)
    rank_bool = sum([singval > rank_tol])
    rank = sum(rank_bool)
    return rank


def get_xcross(seq_1, seq_2):  # Returns cross-correlation between two sequences
    xcross_len = len(seq_1) + len(seq_2) - 1
    xcross = np.zeros((xcross_len, 1), dtype=complex)
    seq_2_padded = np.concatenate((np.zeros(len(seq_1)-1), seq_2, np.zeros(len(seq_1)-1)))  # Add padding to seq_2
    for i in range(len(seq_2_padded)-len(seq_1)+1):
        xcross[i] = np.sum(seq_1 * seq_2_padded[i: i+len(seq_1)])
    return xcross


def get_xcross_err(max_xcorr):  # Returns cross-correlation error metrics per scenario
    xcross_smp_err = sum(sum(max_xcorr == -100))  # Number of times xcross peak is missdetected for a scenario
    xcross_smp_errtx1 = sum(max_xcorr[0, :] == -100)  # Number of times xcross peak is missdetected for Tx1 at
    # all Rx for a given scenario
    return xcross_smp_err, xcross_smp_errtx1


def get_xcross_nrm(xcrosspeak_tx1, iterations):  # Normalized cross correlation peaks for Tx 1, for all Rx (columns)
    # and all iterations (rows)).
    xcrosspeak_nrm = np.zeros((iterations, len(xcrosspeak_tx1[0])), dtype=complex)
    for mm in range(len(xcrosspeak_tx1[0])):
        xcrosspeak_nrm[:, mm] = xcrosspeak_tx1[:, mm] / max(abs(xcrosspeak_tx1[:, mm]))
    return xcrosspeak_nrm


def get_xcross_div(xcrossnrm_tx1, iterations):  # Exploits diversity of multiple Rx to enhance detection
    xcrossnrm_tx1_div = np.zeros((1, iterations), dtype=complex)
    for i in range(iterations):
        xcrossnrm_tx1_div[0, i] = max(abs(xcrossnrm_tx1[i, :]))
    return xcrossnrm_tx1_div


def get_chard(total_ch_ite_pow, iterations):  # Exploits diversity of multiple Rx to enhance detection
    if len(total_ch_ite_pow) != iterations:
        print('Error in CH calculation. Check Tx, Rx matrix dimensions.')
    chard = np.var(total_ch_ite_pow)/((np.mean(total_ch_ite_pow))**2)
    return chard


def get_chard_mean(total_ch_ite_pow):
    mean = (np.mean(total_ch_ite_pow))
    return mean
