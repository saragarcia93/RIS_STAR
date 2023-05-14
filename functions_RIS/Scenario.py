"""
Created on November 1, 2021

@author: Sara Garcia Sanchez

This code generates a scenario for an 'nRays' ray tracing simulation. The 2D simulation environment extends from
0 to 2*'envSize' in the x-axis and y-axis. The scenario consist of 'nTx' number of
transmitters with a single omnidirectional antenna each and a single receiver equipped with an 'rx_nAnt' phased
array antenna. Each antenna at the receiver is assumed to be omnidirectional. Transmitters are assumed to be
time synchronized. Moreover, transmitters and the receiver are assumed to be frequency synchronized. The
environment has a (multiple) reflector(s), whose 'reflector_type' can be RIS or non-RIS depending on whether
they are configurable.
NOTE: if envSize is too large, the attenuation caused by the channel raised and cross-correlation peak detection
fails as expected.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import random
import math
random.seed(5)

__all__ = ["ScenarioClass"]  # Name of class below


class ScenarioClass:
    def __init__(self):
        self.rx_nAnt = 3  # number of antennas at Rx phased array
        self.nTx = 3  # number of Tx
        self.nRIS = 2  # number of reflectors (maximum 2)
        self.RIS_nConfig = 5  # Number of different config to change the RIS within Ts
        self.freq = 900e6  # frequency
        self.wavelength = 3e8/self.freq  # wavelength
        self.alpha_los = 1  # Attenuation factor for reflector 2. Set to 0 to remove reflector from scenario.
        self.alpha_rfl_1 = 0.8  # Attenuation factor for reflector 1. Set to 0 to remove reflector from scenario.
        self.alpha_rfl_2 = 0.2  # Attenuation factor for reflector 2. Set to 0 to remove reflector from scenario.
        self.envSize = 10  # size of simulation environment, meters
        self.txLocRadius = 1 * self.wavelength  # Circle radius within Tx are randomly located
        self.txLocCenterX = self.envSize  # Center coordinate for circle, x axis
        self.txLocCenterY = self.envSize  # Center coordinate for circle, y axis
        self.rx_coord = (self.envSize, 0, 0)  # initial central coord. of the Rx, meters
        self.tx_coord = np.zeros((3,  self.nTx))  # initial coord. of the Tx, meters
        self.reflector_coord = (-self.envSize, self.envSize, 0)  # initial central coord. of the reflector, meters
        self.RIS_typeConfig = 'equally_spaced'
        self.reflector_type = 'noRIS'  # RIS, noRIS (non configurable reflector)
        self.np_reflector_size = 'inf'  # Non-programmable reflector size
        # 'inf': attenuation goes with d1+d2, 'sub-wavelength' attenuation goes with d1*d2.
        self.ris_elem_azi = 3  # Number of RIS antenna elements, for azimuth
        self.ris_elem_ele = 1  # CHECK BEFORE INCREASING: 3D CASE. Number of RIS antenna elements, for elevation
        self.polarization = 'vertical'  # E polarization (vertical, horizontal)
        self.nRays = 1  # number of rays in ray tracing model
        self.txGain = 20  # Transmitter gain (dB)
        self.rxGain = 20  # Receiver gain (dB)
        self.txAntennaGain = 3  # Transmitter antenna gain (dBi)
        self.rxAntennaGain = 3  # Receiver antenna gain (dBi)

    def get_rx_coord(self):  # Estimates Rx antennas coordinates in Cartesian system (one antenna per column).
        if (self.rx_nAnt % 2) == 0:
            rx_coord_1d = np.linspace(-((self.wavelength/4)+(((self.rx_nAnt/2)-1)*(self.wavelength/2))),
                                      ((self.wavelength/4)+(((self.rx_nAnt/2)-1)*(self.wavelength/2))),
                                      num=self.rx_nAnt)
        else:
            rx_coord_1d = np.linspace(-((self.rx_nAnt-1)/2)*(self.wavelength/2),
                                      ((self.rx_nAnt-1)/2)*(self.wavelength/2),
                                      num=self.rx_nAnt)
        rx_coord = np.array(
            [self.rx_coord[0]+rx_coord_1d, self.rx_coord[1] * np.ones(self.rx_nAnt),   # Rx along x-axis.
             self.rx_coord[2] * np.ones(self.rx_nAnt)])
        return rx_coord

    # Returns min distance to be in far field conditions for the maximum rx array size considered in the simulation run
    # This helps to locate the center of the tx circle in the y-axis at the same location for all rx array sizes.
    def max_ff_cond(self, rxarray_maxsize):
        print(rxarray_maxsize, 'hihe')
        if (rxarray_maxsize % 2) == 0:
            rx_coord_1d = np.linspace(-((self.wavelength/4)+(((rxarray_maxsize/2)-1)*(self.wavelength/2))),
                                      ((self.wavelength/4)+(((rxarray_maxsize/2)-1)*(self.wavelength/2))),
                                      num=rxarray_maxsize)
        else:
            rx_coord_1d = np.linspace(-((rxarray_maxsize-1)/2)*(self.wavelength/2),
                                      ((rxarray_maxsize-1)/2)*(self.wavelength/2),
                                      num=rxarray_maxsize)
        max_rx_coord = np.array(
            [self.rx_coord[0]+rx_coord_1d, self.rx_coord[1] * np.ones(rxarray_maxsize),   # Rx along x-axis.
             self.rx_coord[2] * np.ones(rxarray_maxsize)])

        if len(max_rx_coord[0]) == 1:  # If only 1 rx antenna, larger dimension is antenna size (assumed λ/2)
            max_array_size = self.wavelength/2
        else:  # If the rx has more than 1 antenna, larger dimension is the array size
            max_array_size = max_rx_coord[0, -1]-max_rx_coord[0, 0]
        max_d_min_ff = ((2*(max_array_size**2))/self.wavelength)

        d_miny = self.txLocCenterY - self.txLocRadius
        if d_miny < max_d_min_ff:
            print('WARNING: Txs are not located in the far field region', max_d_min_ff, '(m)'
                  ' Consider decreasing txLocRadius or increasing txLocCenterY.')
            exit()
        return max_d_min_ff

    def get_tx_coord(self, rx_coord, rand_rho_ite, rand_theta_ite, iter):  # Estimates random Txs coordinates in Cartesian system. (one Tx per column)
        # ensure that Txs are located within the far field region. Randomly allocate Txs within a circle of radius
        # txLocRange centered in the x-axis in txLocCenterX
        if len(rx_coord[0]) == 1:  # If only 1 rx antenna, larger dimension is antenna size (assumed λ/2)
            array_size = self.wavelength / 2
        else:  # If the rx has more than 1 antenna, larger dimension is the array size
            array_size = rx_coord[0, -1] - rx_coord[0, 0]
        d_min_ff = ((2 * (array_size ** 2)) / self.wavelength)  # Minimum Tx-Rx distance to ensure far field conditions
        d_maxy = self.txLocCenterY + self.txLocRadius
        d_miny = self.txLocCenterY - self.txLocRadius
        d_maxx = self.txLocCenterX + self.txLocRadius
        d_minx = self.txLocCenterX - self.txLocRadius
        rand_rho = self.txLocRadius * rand_rho_ite[:, iter]
        rand_theta = 2 * math.pi * rand_theta_ite[:, iter]
        coord_x = self.txLocCenterX + rand_rho * np.cos(rand_theta)
        coord_y = self.txLocCenterY + rand_rho * np.sin(rand_theta)
        tx_coord = np.array([coord_x, coord_y, np.zeros(self.nTx)])  # Assuming 2D case, z = 0
        if d_maxy > self.envSize * 2:
            print('WARNING: Distance between Tx and Rx cannot exceed the environment size.'
                  ' Consider decreasing txLocRadius and/or txLocCenterY or increasing envSize.')
            exit()
        if d_miny < d_min_ff:
            print('WARNING: Txs are not located in the far field region', d_min_ff, '(m)'
                                                                                    ' Consider decreasing txLocRadius or increasing txLocCenterY.')
            exit()
        if d_maxx > self.envSize * 2 or d_minx < 0:
            print('WARNING:  Txs are not located within the environment [0, 2*envSize].'
                  'Consider modifying txLocCenterX and txLocRadius.')
            exit()
        if self.txLocRadius > 5 * self.wavelength:
            print('WARNING:  Consider reducing circle_radius to avoid accounting for the effect of channel '
                  'attenuation variance due to large distance different between transmitters to receivers.')
        return tx_coord, d_min_ff

    def get_snellreflection_yaxis(self):
        reflection_yaxis_1 = np.zeros((self.rx_nAnt, self.nTx))
        reflection_yaxis_2 = np.zeros((self.rx_nAnt, self.nTx))
        for i in range(0, self.rx_nAnt):
            for j in range(0, self.nTx):
                tx_xaxis = self.tx_coord[0, j]
                rx_xaxis = self.rx_coord[0, i]
                # Calculates reflection point on y-axis based on Snell's law with equal incident and reflected angles
                reflection_yaxis_1[i, j] = (self.tx_coord[1, j]*rx_xaxis) / (tx_xaxis+rx_xaxis)
                reflection_yaxis_2[i, j] = (self.tx_coord[1, j] * ((2 * self.envSize)-rx_xaxis))\
                    / ((4 * self.envSize) - tx_xaxis - rx_xaxis)
        return reflection_yaxis_1, reflection_yaxis_2
