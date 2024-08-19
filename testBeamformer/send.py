from BeamDrop.hw_beamformer.beamctl import BeamCtl
from BeamDrop.hw_80211 import hw_80211
from BeamDrop.Network import BDP, BDPPCSegment
from scapy.compat import raw
import matplotlib.pyplot as plt
import math
import threading
import multiprocessing
import numpy as np
import subprocess
from scipy.constants import c, pi
from math import e
import time


def get_array_factor(tx_positions, tx_weights, rfreq=2.437e9):
    # --- Precomputation ---
    angle_resolution = 1
    angle_range = 180
    far_field_dist = 10 ** 6
    lamb = c / rfreq

    tx_num = tx_positions.shape[0]
    if tx_weights is None:
        tx_weights = np.ones(tx_num, dtype=complex)

    # --- Calculation ---
    incident_angles = np.arange(0, angle_range + angle_resolution, angle_resolution)
    AF = np.zeros(int(angle_range / angle_resolution) + 1, dtype=complex)
    theta_index = 0
    for theta in incident_angles:
        r0 = far_field_dist * np.array(([np.cos(np.deg2rad(theta))], [np.sin(np.deg2rad(theta))]))
        d = ((tx_positions[:, 0] - r0[0]) ** 2 + (tx_positions[:, 1] - r0[1]) ** 2) ** 0.5
        angle = d / lamb * 2 * pi
        AF[theta_index] = np.sum(e ** (-1j * angle) * tx_weights / np.sqrt(tx_num))
        theta_index += 1

    # --- Normalization ---
    AF_log = 20 * np.log10(abs(AF))
    # AF_log -= 10 * np.log10(np.sum(np.abs(tx_weights)))
    AF_log[AF_log < -50] = -50
    return incident_angles, AF_log


def plot_afs(afs, xs, labels):
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for i, l in enumerate(labels):
        ax.plot(xs, afs[i], label=l)
    ax.legend()
    ax.set_ylim([-55, 20])
    plt.show()


def sender_thread(beamformer_weights):
    # --- initialize broadcast interface ---
    monitor_iface = "wlxe84e06952924"
    monitor_channel = 1

    # init phy interface
    subprocess.run(['sudo', 'ifconfig', monitor_iface, 'up'], check=True)
    # iw {dev_name} set monitor none
    subprocess.run(['sudo', 'iw', monitor_iface, 'set', 'monitor', 'none'], check=True)
    # iw dev {dev_name} set txpower fixed 4000
    subprocess.run(['sudo', 'iw', 'dev', monitor_iface, 'set', 'txpower', 'fixed', '3000'], check=True)
    # iw dev {dev_name} set channel <channel> [HT20|HT40+|HT40-]
    subprocess.run(['sudo', 'iw', 'dev', monitor_iface, 'set', 'channel',
                    str(monitor_channel)], check=True)

    iface_mac_int = int("0x" + "e8:4e:06:9c:bd:94".replace(':', ''), 16)
    wiphy = hw_80211.init_phy(iface=monitor_iface,
                              mac_addr=iface_mac_int)

    # --- initialize beamformer interface ---
    enable_bfer = True
    if enable_bfer:
        bf_phy = BeamCtl(bfhw_dev='/dev/ttyUSB0', bfhw_baudrate=1000000, tx_power=3000)
        bf_phy.store_beams(beamformer_weights)
        pass
    beam_num = beamformer_weights.shape[0]

    repetition = 10
    bpp = 1
    point_num = 1250
    while True:
        for i in range(beam_num):
            # round-robin each cell
            if enable_bfer:
                bf_phy.switch_beam(i)
                pass

            batched_msg = b""
            msg_length = []
            for j in range(repetition):
                p_seg = (BDP(type=2, deviceMAC="e8:4e:06:9c:bd:94") /
                         BDPPCSegment(frame_id=0, timestamp_ns=time.time_ns(),
                                      point_num=point_num, point_bpp=bpp,
                                      point_cloud=1250 * str(i),
                                      region_id=int(i)))
                batched_msg += raw(p_seg)
                msg_length.append(len(raw(p_seg)))
            hw_80211.inject_multiple_packets(wiphy, msg_length, batched_msg, 56)
            time.sleep(0.1)
        print("Finish one cycle")
        time.sleep(0.2)


def main():
    # --- radio config ---
    rfreq = 2.437e9
    antenna_spacing = 0.05
    angular_region_num = 3
    tx_num = 4
    tx_positions = np.array([[antenna_spacing * x, 0] for x in range(tx_num)])
    tx_weights = lambda aod: [e ** (1j * 2 * pi * rfreq * i * antenna_spacing * math.cos(aod) / c)
                              for i in range(tx_num)]

    # --- calculate steering angles of each cell beam ---
    beamformer_angles = np.linspace(45, 135, angular_region_num)
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
