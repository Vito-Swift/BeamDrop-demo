from BeamDrop.hw_beamformer.beamctl import BeamCtl
from BeamDrop.hw_80211 import hw_80211
from BeamDrop.Network import BDP, BDPPCSegment
from scapy.compat import raw
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import subprocess
from scipy.constants import c, pi
from math import e
import time



def main():
    # --- radio config ---
    rfreq = 2.437e9
    antenna_spacing = 0.05
    angular_region_num = 3
    tx_num = 4
    tx_positions = np.array([[antenna_spacing * x, 0] for x in range(tx_num)])
    tx_weights = lambda aod: [e ** (1j * 2 * pi * rfreq * i * math.cos(aod) / c)
                              for i in range(tx_num)]

    # --- calculate steering angles of each cell beam ---
    beamformer_angles = np.linspace(20, 160, angular_region_num)
    beamformer_weights = np.array([tx_weights(np.deg2rad(direction)) for direction in beamformer_angles])
    print(f"Angles: {beamformer_angles}")
    print(f"Weights: {beamformer_weights}")

    # --- plot array factor ---
    afs = [get_array_factor(tx_positions, w, rfreq=rfreq)[1] for w in beamformer_weights]
    xs, _ = get_array_factor(tx_positions, beamformer_weights[0], rfreq=rfreq)
    plt_t = multiprocessing.Process(target=plot_afs, args=(afs, xs, beamformer_angles,))
    plt_t.start()

    # --- detach sharer thread ---
    send_t = multiprocessing.Process(target=sender_thread, args=(beamformer_weights,))
    send_t.start()


if __name__ == '__main__':
    main()
