"""
Created on November 1, 2021

@author: Sara Garcia Sanchez

"""

# Results for paper plots SISO and SIMO: scenario.reflector_type = 'noRIS' , iterations = 1000
# SISO: att_reflector_1 = [0, 0.2, 0.4, 0.6, 0.8, 1], att_reflector_2 = [1], rx_nAnt = [1]
# SIMO: att_reflector_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], att_reflector_2 = [1], rx_nAnt = [10]

# Results for paper plots RIS-ST: scenario.reflector_type = 'RIS' , iterations = 1000, rx_nAnt = [1]
#  att_reflector_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], att_reflector_2 = [1]
#  scenario.RIS_nConfig = 10

import functions_RIS
import numpy as np
import math
import signal
import random
import matplotlib.pyplot as plt


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('PyCharm')


# Simulation config.
# GS (hardcoded, to be replaced!!!)
tx_att = [1, 0, 0]  # Attenuation factor for each Tx. 1: no attenuation 0: full attenuation
whSeq = np.array([[-1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1]])  # Walsh Hadamard sequence for Tx 1
detection_thr = 0.2
iterations = 100  # 10000
num_rx_nAnt = 1   # DO NOT CHANGE
# rx_nAnt = list(range(1, num_rx_nAnt+1))  # 2, 4, 8, 16, 32, 64, 128 list(range(1, 11))
rx_nAnt = [10]  # CHANGE THIS: this is the maximum number of antennas in the SIMO simulation. Plot will include data
# from 1 to rx_nAnt. Plot at the end will be adding data from increasing number of antennas to calculate the CH
att_reflector_1 = [0, 0.2, 0.4, 0.6, 0.8, 1]  # Used for SISO plot
att_reflector_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Used for SIMO and RIS-ST plot
#  att_reflector_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5, 1/0.4, 1/0.3,
#                 1/0.2, 1/0.1]
#  att_reflector_1 = [1]

att_reflector_2 = [1] # For normalization
noise_dev = 0.1

# Metrics config.
rank_tol = 0.01

# Init. parameters
xCross_smpErrTOT = 0  # Number of times xcross peak is missdetected for any Tx-Rx, for all iterations
xCross_smpErrTx1TOT = 0  # Number of times xcross peak is missdetected for Tx1 all Rx, for all iterations
att_ratio = [x / att_reflector_2[0] for x in att_reflector_1]


# Scenario config.
scenario = functions_RIS.ScenarioClass()  # Creates the scenario, an object of class Scenario
scenario.freq = 900e6  # Operation frequency
scenario.wavelength = 3e8 / scenario.freq  # wavelength
scenario.envSize =50*scenario.wavelength  # Scenario size (x, y : [0 2envsize], meters)
scenario.txLocRadius = 1*scenario.wavelength  # Circle radius within Tx are randomly located
scenario.txLocCenterX = scenario.envSize  # Center coordinate for circle, x axis
scenario.txLocCenterY = scenario.envSize#+scenario.wavelength  # Center coordinate for circle, y axis
scenario.rx_coord = (scenario.envSize, 0, 0)  # initial central coord. of the Rx, meters
# scenario.rx_nAnt = MIMO_dimension[kk]  # Number of Rx antennas
scenario.nTx = len(tx_att)  # Number of Tx  MIMO_dimension[kk]
#  scenario.nTx = 100  # Number of Tx
scenario.nRIS = 2   # Number of reflectors
scenario.RIS_nConfig = 10  # Number of different config to change the RIS within Ts
scenario.alpha_los = 0  # Attenuation factor for reflector 2. Set to 0 to remove reflector from scenario.
scenario.alpha_rfl_1 = 1  # Attenuation factor for reflector 1. Set to 0 to remove reflector from scenario. 100
scenario.alpha_rfl_2 = 1  # Attenuation factor for reflector 2. Set to 0 to remove reflector from scenario. 100
scenario.reflector_type = 'RIS'  # Type of reflector
scenario.RIS_typeConfig = 'equally_spaced'  # Type of RIS configurations: 'equally_spaced' or 'random'
scenario.np_reflector_size = 'inf'  # Non-programmable reflector size: 'inf' or 'sub-wavelength'
scenario.ris_elem_azi = 1  # Number of RIS antenna elements, for azimuth
scenario.ris_elem_ele = 1  # CHECK BEFORE INCREASING: 2D to 3D CASE. Number of RIS antenna elements, for elevation
scenario.nRays = 1   # Number of rays in ray-tracing model
scenario.polarization = 'vertical'  # E polarization (vertical, horizontal)
scenario.txGain = 50  # Transmitter gain (dB)
scenario.rxGain = 30  # Receiver gain (dB)
scenario.txAntennaGain = 3  # Transmitter antenna gain (dBi)
scenario.rxAntennaGain = 3  # Receiver antenna gain (dBi)

if scenario.reflector_type == 'noRIS':
    scenario.RIS_nConfig = 1
    CH = np.zeros(len(rx_nAnt))
elif scenario.reflector_type == 'RIS':
    rx_nAnt = [1]  # If RIS is active, we set number of receiver to 1
    scenario.np_reflector_size = 'sub-wavelength'  # For comparison with the RIS component
    total_ch_iteRx = np.zeros((iterations, scenario.RIS_nConfig, len(att_reflector_1)), dtype=complex)

# Init. random parameters for Tx deployment to be same for all Rx array size and RIS configs.
rand_rho_ite = np.random.rand(scenario.nTx, iterations)
rand_theta_ite = np.random.rand(scenario.nTx, iterations)

# Set RIS weights (phase, random uniformly distributed, unit magnitude). As many rows as configs. As many columns
# as antennas in the horizontal direction (y axis) for azimuth steering (2D case)
ris_allConfigs = functions_RIS.set_ris_weights(scenario.ris_elem_azi, scenario.RIS_nConfig, scenario.RIS_typeConfig)
print(ris_allConfigs, 'ris_allConfigs')

# Init. parameters
destructive_int_rf1 = np.zeros(scenario.RIS_nConfig)
destructive_int_rf2 = np.zeros(scenario.RIS_nConfig)
total_ch_itePow = np.zeros((iterations, len(rx_nAnt), scenario.RIS_nConfig))
total_ch_itePow_SISO = np.zeros((iterations, len(att_reflector_1)))
CH = np.zeros((rx_nAnt[0], len(att_reflector_1)))

for rr in range(len(att_reflector_1)):
    print(rr, 'rr')
    scenario.alpha_rfl_1 = att_reflector_1[rr]
    for kk in range(len(rx_nAnt)):
        scenario.rx_nAnt = rx_nAnt[kk]  # Number of Rx antennas  MIMO_dimension[kk]
        scenario.rx_coord = (scenario.envSize, 0, 0)  # initial central coord. of the Rx, meters

        # Init. parameters
        xCrossPeak_Tx1 = np.zeros((iterations, scenario.rx_nAnt), dtype=complex)
        total_ch_ite = np.zeros((iterations, scenario.rx_nAnt, scenario.nTx), dtype=complex)

        # Scenario Rx coordinates
        scenario.rx_coord = scenario.get_rx_coord()

        for cc in range(scenario.RIS_nConfig):
            print(cc, 'cc')
            ris_w = ris_allConfigs[cc, :]
            print(ris_w, 'ris_w')
            # Scenario Tx coordinates
            for i in range(iterations):
                scenario.tx_coord, d_min_ff = scenario.get_tx_coord(scenario.rx_coord, rand_rho_ite, rand_theta_ite, i)
                if i==0:
                    print(scenario.tx_coord[:,0], 'tx_coord')
                # Get point of reflection from each Tx to each Rx antenna on the reflective surface following Snell's law
                reflection_yaxis_1, reflection_yaxis_2 = scenario.get_snellreflection_yaxis()
                # Estimate the channel matrix (FSPL)
                total_ch, destructive_int1, destructive_int2 = functions_RIS.get_total_ch(scenario.tx_coord,
                                                                                          scenario.rx_coord,
                                                                                          scenario.freq,
                                                                                          scenario.reflector_type,
                                                                                          scenario.polarization,
                                                                                          reflection_yaxis_1,
                                                                                          reflection_yaxis_2,
                                                                                          scenario.envSize,
                                                                                          scenario.alpha_los,
                                                                                          scenario.alpha_rfl_1,
                                                                                          scenario.alpha_rfl_2, tx_att,
                                                                                          scenario.txLocCenterY,
                                                                                          scenario.ris_elem_azi,
                                                                                          scenario.ris_elem_ele,
                                                                                          scenario.wavelength, d_min_ff,
                                                                                          ris_w, scenario.np_reflector_size)
                # print(total_ch)
                total_ch_ite[i, :, :] = total_ch
                destructive_int_rf1[cc] = destructive_int_rf1[cc] + destructive_int1
                destructive_int_rf2[cc] = destructive_int_rf2[cc] + destructive_int2

                # Estimate the channel matrix including Tx and Rx gains
                total_ch_gain = functions_RIS.get_total_ch_gain(total_ch, scenario.txGain, scenario.rxGain,
                                                                scenario.txAntennaGain, scenario.rxAntennaGain)
                y = np.zeros((len(whSeq[0]), scenario.rx_nAnt), dtype=complex)

                for mm in range(scenario.rx_nAnt):  # Each column is for one Rx
                    for jj in range(len(whSeq[0])):  # Each row is for one GS symbol (in time)
                        noise_var = noise_dev**2
                        noise = complex(np.random.normal(0, math.sqrt(noise_var/2), size=(1, 1)),
                                        np.random.normal(0, math.sqrt(noise_var / 2), size=(1, 1)))
                        # Received signal, combined from all Txs. One column per Rx. One raw per temporal GS sample.
                        y[jj, mm] = np.dot(total_ch_gain[mm, :], whSeq[:, jj]) + noise

                # We find one correlation peak for each Tx at each Rx in iteration i
                maxXCorr = np.zeros((scenario.nTx, scenario.rx_nAnt), dtype=complex)
                for mm in range(scenario.rx_nAnt):
                    for jj in range(scenario.nTx):
                        xCrossPeak_calc = max(abs(functions_RIS.get_xcross(y[:, mm], whSeq[jj, :])))  # Calculated Xcross peak
                        xCrossPeak_expect = abs(np.sum(y[:, mm]*whSeq[jj, :]))  # Expected Xcross peak
                        if xCrossPeak_calc == xCrossPeak_expect:
                            maxXCorr[jj, mm] = xCrossPeak_calc
                        else:
                            maxXCorr[jj, mm] = -100  # Error tag

                xCross_smpErr, xCross_smpErrTx1 = functions_RIS.get_xcross_err(maxXCorr)  # Get error metrics for iteration i
                # Calculate global simulation error metrics (all iterations)
                xCross_smpErrTOT = xCross_smpErrTOT + xCross_smpErr
                xCross_smpErrTx1TOT = xCross_smpErrTx1TOT + xCross_smpErrTx1
                xCrossPeak_Tx1[i, :] = maxXCorr[0, :]  # We only look at values for Tx1, all Rx

            if scenario.reflector_type == 'noRIS':  # If noRIS, calculate CH for a SIMO system
                total_ch_iteRx = np.sum(total_ch_ite, 2)  # Combined channel from all Tx at each Rx antenna, all iterations
                for ll in range(len(total_ch_iteRx[0])):
                    total_ch_itePow = functions_RIS.get_ch_pow(total_ch_iteRx[:, 0:ll+1])
                    CH[ll, rr] = functions_RIS.get_chard(total_ch_itePow, iterations)
                    if ll == 0:  # CHANGE THIS TO SELECT MIDDLE ANTENNA OR AN ANTENNA CLOSER TO THE CENTER
                        total_ch_itePow_SISO[:, rr] = np.ravel(functions_RIS.get_ch_pow(total_ch_iteRx[:, 0:ll+1]))
            elif scenario.reflector_type == 'RIS':  # If RIS, calculate CH for different RIS configs within Ts and
                # the central receiver.
                total_ch_iteRx[:, cc, rr] = np.ravel(np.sum(total_ch_ite, 2), order='C')  # Combined channel from all Tx at
                # each Rx antenna, all iterations (iterations x RIS_nConfig)

            # Normalized cross correlation peaks for Tx 1, for all Rx (columns) and all iterations (rows))
            xCrossNrm_Tx1 = functions_RIS.get_xcross_nrm(xCrossPeak_Tx1, iterations)
            #  print(len(xCrossNrm_Tx1[0]))

            # Exploits diversity of multiple Rx to enhance detection
            if scenario.rx_nAnt > 1:
                xCrossNrm_Tx1_div = functions_RIS.get_xcross_div(xCrossNrm_Tx1, iterations)
            else:
                xCrossNrm_Tx1_div = np.transpose(xCrossNrm_Tx1)

            # OUTPUTS and WARNINGS

            #print('INFO:', int(destructive_int_rf1[cc]/(kk+1)), 'destructive interferences out of', iterations,
            #      'iterations (', 100*destructive_int_rf1[cc]/(iterations*(kk+1)), '%) from Tx1 at Rx1 caused by multipath.')

            if (scenario.envSize / scenario.wavelength) > 10:
                print('WARNING 1: Consider reducing envSize for removing cross-correlation results '
                      'dependency on channel attenuation. Suggested envSize = 2*wavelength')

            if xCross_smpErrTx1TOT == 0:
                print('Number of times the correlation peak is not correctly found for Tx1:', xCross_smpErrTx1TOT)
            else:
                if sum(tx_att[1:]) == 0:
                    print('WARNING 3: Number of times the correlation peak is not correctly found for Tx1',
                          xCross_smpErrTx1TOT, 'Consider reducing envSize. Suggested value envSize = 2*scenario.wavelength.')
                else:
                    print('WARNING 3: Number of times the correlation peak is not correctly found for Tx1', xCross_smpErrTx1TOT,
                          ' If not 0, probably caused by pilot contamination from other Txs. Consider reducing tx_att for'
                          ' Tx2..N att or increasing pilot sequence length in whSeq.')

            if scenario.reflector_type == 'RIS' and scenario.rx_nAnt > 1:
                print('WARNING: Since RIS is active, consider setting rx_nAnt = 1 to analyse the channel hardening effect '
                      'of the RIS compared to a SIMO case. Set tx_att = [1, 0, 0].')


            data_sorted = np.sort(abs(xCrossNrm_Tx1_div))
            # calculate the proportional values of samples
            p = 1. * np.arange(len(xCrossNrm_Tx1_div[0])) / (len(xCrossNrm_Tx1_div[0]) - 1)

if scenario.ris_elem_ele > 1:
    print('WARNING: Code only supports 2D scenarios. Upgraded is needed to support 3D.')
    exit()

# CH for SISO and SIMO cases
if scenario.reflector_type == 'noRIS':
    for rr in range(len(att_reflector_1)):
        plt.plot(range(rx_nAnt[0]), CH[:, rr])  # Cross-correlation sequence plot
    plt.xlabel('Number of Receiving Antennas')
    plt.ylabel('var(CH)')
    plt.show()

if scenario.reflector_type == 'noRIS':
    np.savetxt('CHH_SIMO.txt', CH)
    X = np.linspace(1, rx_nAnt[0], rx_nAnt[0])
    att_reflector_2np = np.array(np.array(att_reflector_2))
    Y = np.log10(att_reflector_2np[0]/np.array(att_reflector_1))
    Y = np.array(att_reflector_1)/ att_reflector_2np[0]
    plt.pcolormesh(X, Y, np.transpose(CH), cmap='Reds') # , shading='auto' binary
    cbar = plt.colorbar()
    plt.xlabel('Number of Receiving Antennas')
    plt.ylabel(r'$\log (\alpha_2 / \alpha_1)$')
    plt.ylabel(r'$\alpha_1 / \alpha_2$')
    plt.xlim([1, 10])
    plt.ylim([0.1, 1])
    cbar.set_label('var(CH)')
    plt.show()

if scenario.reflector_type == 'noRIS':
    total_ch_itePow_SISO = total_ch_itePow_SISO / np.max(total_ch_itePow_SISO)

    markers = ['g-', 'r--', 'b:', 'k-.', 'm-', 'y--', 'o:', 'k-.', '-v',
               '-x']
    width = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    for rr in range(len(att_reflector_1)):
        #  sort data
        x = np.sort(total_ch_itePow_SISO[:, rr], 0)
        #  print(x,'x')
        #  calculate CDF values
        y = 1. * np.arange(len(total_ch_itePow_SISO)) / (len(total_ch_itePow_SISO) - 1)

        #  plot CDF
        plt.plot(x, y, markers[rr], linewidth=width[rr])

    legend = plt.legend(att_ratio)
    legend = plt.legend([0, 0.2, 0.4, 0.6, 0.8, 1])
    legend.set_title(r'$\alpha_1 / \alpha_2$')
    plt.xlabel('Normalized Channel Power (nat)')
    plt.ylabel('CDF')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # plt.grid()
    plt.show()


# CH for RIS-ST case
if scenario.reflector_type == 'RIS':
    CH = np.zeros((scenario.RIS_nConfig, len(att_reflector_1)))
    for rr in range(len(att_reflector_1)):
        for ff in range(scenario.RIS_nConfig):
            total_ch_itePow = functions_RIS.get_ch_pow(total_ch_iteRx[:, 0:ff + 1, rr])  # Power at the Rx OK
            CH[ff, rr] = functions_RIS.get_chard(total_ch_itePow, iterations)

    plt.plot(range(scenario.RIS_nConfig), CH)  # Cross-correlation sequence plot
    plt.xlabel('Number of RIS Configurations')
    plt.ylabel('var(CH)')
    plt.show()

if scenario.reflector_type == 'RIS':
    np.savetxt('CHH_RIS_ST.txt', CH)
    X = np.linspace(1, scenario.RIS_nConfig, scenario.RIS_nConfig)
    att_reflector_2np = np.array(np.array(att_reflector_2))
    Y = np.log10(att_reflector_2np[0] / np.array(att_reflector_1))
    Y = np.array(att_reflector_1) / att_reflector_2np[0]
    plt.pcolormesh(X, Y, np.transpose(CH), cmap='binary') # , shading='nearest'  Blues
    cbar = plt.colorbar()
    plt.xlabel('Number of RIS Configurations')
    plt.ylabel(r'$\log (\alpha_2 / \alpha_1)$')
    plt.ylabel(r'$\alpha_1 / \alpha_2$')
    cbar.set_label('var(CH)')
    plt.xlim([1, 10])
    plt.ylim([0.1, 1])
    plt.show()


print('-------NOTE: For SISO and SIMO results, run simulation first to generate CHH_SISO.txt and CHH_SIMO.txt. Then, uncomment lines 490-513.')
'''
CHH_SIMO = np.loadtxt('CHH_SIMO.txt', dtype=float)
CHH_RIS_ST = np.loadtxt('CHH_RIS_ST.txt', dtype=float)

#CHH_error = np.abs(CHH_RIS_ST-CHH_SIMO)
CHH_error = (CHH_SIMO-CHH_RIS_ST)

X = np.linspace(1, scenario.RIS_nConfig, scenario.RIS_nConfig)
att_reflector_2np = np.array(np.array(att_reflector_2))
Y = np.log10(att_reflector_2np[0] / np.array(att_reflector_1))
Y = np.array(att_reflector_1) / att_reflector_2np[0]
plt.pcolormesh(X, Y, np.transpose(CHH_error), cmap='coolwarm_r',  shading='gouraud', vmin=-0.02, vmax=0.02)  # , shading='auto' , shading='gouraud'
# plot = plt.pcolormesh(np.transpose(CH), cmap='binary') #RdYlBu_r
cbar = plt.colorbar()
# plt.grid(which='minor', color='k', linestyle = '--', linewidth = 0.5)
# plt.minorticks_on()
# att_reflector_1
plt.xlabel('System Dimension')
plt.ylabel(r'$\log (\alpha_2 / \alpha_1)$')
plt.ylabel(r'$\alpha_1 / \alpha_2$')
cbar.set_label(' $ var(CH_{SIMO}) - var(CH_{RIS-ST})$')
# plt.grid(which='minor', alpha=1,  color='k')
plt.xlim([1, 10])
plt.ylim([0.1, 1])
plt.show()
'''

