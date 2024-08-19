import multiprocessing
import time
import struct
import subprocess
from threading import Thread
from multiprocessing import Process, Queue, Manager
import numpy as np
from scapy.pipetool import QueueSink, PipeEngine
from scapy.scapypipes import SniffSource
from scapy.packet import Raw
from scapy.layers.dot11 import LLC, RadioTap
from scapy.compat import raw
from BeamDrop.Network import (BDP, BDPPCSegment)


def sink_parser(sniff_sink: QueueSink, region_records):
    while True:
        pkt = sniff_sink.recv()
        if pkt.haslayer(LLC) and pkt.haslayer(Raw):
            try:
                rx_pkt = BDP(raw(pkt[Raw])[1:])
            except struct.error:
                print('Received an undecodable packet')
                continue

            if not rx_pkt.haslayer(BDPPCSegment):
                continue

            # parse packet
            region_id = int(rx_pkt[BDPPCSegment].region_id)
            region_records.append([time.time(), region_id, pkt[RadioTap].dBm_AntSignal])


def receiver_thread(monitor_iface: str, region_records):
    sniff_source = SniffSource(iface=monitor_iface, filter='type data')
    sniff_sink = QueueSink()
    t = Thread(target=sink_parser, args=(sniff_sink, region_records,))
    t.start()
    try:
        sniff_source > sniff_sink
        p = PipeEngine(sniff_source)
        p.start()
        p.wait_and_stop()
    except (KeyboardInterrupt, SystemExit):
        p.stop()


def main():
    # --- init monitor interface ---
    monitor_iface = "wlx00c0cab22107"
    monitor_channel = 1

    # init phy interface
    # iw {dev_name} set monitor none
    subprocess.run(['sudo', 'iw', monitor_iface, 'set', 'monitor', 'none'], check=True)
    subprocess.run(['sudo', 'ifconfig', monitor_iface, 'up'], check=True)
    # iw dev {dev_name} set txpower fixed 4000
    # subprocess.run(['sudo', 'iw', 'dev', global_config.monitor_iface, 'set', 'txpower', 'fixed',
    #                 str(global_config.txpower)], check=True)
    # iw dev {dev_name} set channel <channel> [HT20|HT40+|HT40-]
    subprocess.run(['sudo', 'iw', 'dev', monitor_iface, 'set', 'channel',
                    str(monitor_channel)], check=True)

    variable_manager = Manager()
    region_records = variable_manager.list()

    receiver_proc = Process(target=receiver_thread,
                            args=(monitor_iface, region_records,))
    receiver_proc.daemon = True
    receiver_proc.start()

    while True:
        time.sleep(3)

        # drop region records > -0.5 seconds
        rr_array = np.array(region_records)

        if rr_array.size == 0:
            print("No records so far")
            continue

        rr_array = rr_array[time.time() - rr_array[:, 0] < 3]
        if rr_array.size == 0:
            print("No records so far")
            continue

        # statistics over existing region records
        unique, counts = np.unique(rr_array[:, 1], return_counts=True)
        # calculate average RSSI
        cell_rssi = np.zeros(unique.shape[0])
        for enum_i, cell_id in enumerate(unique):
            cell_rssi[enum_i] = np.mean(rr_array[rr_array[:, 1] == cell_id][:, 2])
        max_idx = np.argmax(cell_rssi)
        print(np.array([unique, cell_rssi]).T)
        print(f"PCell ID: {unique[max_idx]}")


if __name__ == '__main__':
    main()
