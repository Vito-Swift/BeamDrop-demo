import os
import math
import sys
import queue
import struct
import numpy as np
import subprocess
import multiprocessing
from scapy.pipetool import QueueSink, PipeEngine
from scapy.scapypipes import SniffSource
from scapy.layers.dot11 import RadioTap, Dot11, LLC
from scapy.packet import Raw
from scapy.compat import raw
from threading import Thread
from multiprocessing import Process, Queue
from BeamDrop.runtime.utils import PCSharingBase
from BeamDrop.runtime.config import D2RReceiverConfig
from BeamDrop.runtime.visualizer import pc_visualize_bev
from BeamDrop.Network import (BDP, BDPPCSegment)


def from_pkts_to_data_dict(current_pc_agg, current_frame_stat):
    vis_pc_agg = current_pc_agg  # np.concatenate([current_pc_agg, *history_pc_list])
    current_cell_id = max(current_frame_stat, key=lambda x: current_frame_stat[x]['rssi'])
    current_cell_rssi = current_frame_stat[current_cell_id]['rssi']
    min_rssi = min([current_frame_stat[x]['rssi'] for x in current_frame_stat.keys()])
    efficient_bw = {}
    filtered_point_list = []
    for cell_id in current_frame_stat.keys():
        if cell_id == current_cell_id:
            filtered_point_list.append(vis_pc_agg[vis_pc_agg[:, 3] == cell_id])
        else:
            keep_ratio = 10 ** -(1 + current_cell_rssi - current_frame_stat[cell_id]['rssi'])
            cell_points = vis_pc_agg[vis_pc_agg[:, 3] == cell_id]
            keep_point_nums = keep_ratio * len(cell_points)
            if keep_point_nums != 0:
                keep_point_index = np.random.choice(cell_points.shape[0], int(keep_point_nums), replace=False)
                keep_points = cell_points[keep_point_index]
            else:
                keep_points = np.zeros((1, 4))
            filtered_point_list.append(keep_points)
        efficient_bw[cell_id] = round(40 * filtered_point_list[-1].shape[0] * 32 * 4 / (1024 * 1024), 2)

    vis_pc_agg = np.concatenate(filtered_point_list)

    return {'pts': vis_pc_agg,
            'gps_coord': (0, 0),
            'state': efficient_bw}


def sink_parser(global_config: D2RReceiverConfig, sniff_sink: QueueSink, vis_queue: Queue):
    current_frame_id = 0
    current_pc_agg = np.zeros((1, 4))
    current_frame_stat = {}
    history_pc_list = []
    history_pc_window_size = 5

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
            # 1. frame index
            frame_i = rx_pkt[BDPPCSegment].frame_id
            # 2. point cloud
            pc = np.frombuffer(rx_pkt[BDPPCSegment].point_cloud, dtype=np.float32)
            pc = pc.reshape((-1, 4)).copy()
            # 3. translation matrix
            trans_mat = np.frombuffer(rx_pkt[BDPPCSegment].transition_mat, dtype=np.float32)
            trans_mat = trans_mat.reshape((4, 4))
            # 4. sharer mac address
            mac_fld, mac_val = rx_pkt.getfield_and_val('deviceMAC')
            mac_bytes = mac_fld.i2m(rx_pkt, mac_val)
            # 5. GPS coordinates
            longitude, latitude = rx_pkt[BDPPCSegment].longitude, rx_pkt[BDPPCSegment].latitude
            # 6. region id
            region_id = rx_pkt[BDPPCSegment].region_id
            # 7. RSSI value
            RSSI = pkt[RadioTap].dBm_AntSignal

            pc[:, 3] = region_id

            if region_id in current_frame_stat:
                avg_rssi = current_frame_stat[region_id]['rssi'] * current_frame_stat[region_id]['count']
                avg_rssi += RSSI
                avg_rssi /= (current_frame_stat[region_id]['count'] + 1)
                current_frame_stat[region_id]['rssi'] = round(avg_rssi, 2)
                current_frame_stat[region_id]['count'] += 1
            else:
                current_frame_stat[region_id] = {'count': 1, 'rssi': RSSI}

            if frame_i != current_frame_id:
                # print(f"get one new frame from  {mac_bytes}, push previous pc frame"
                #      f" (point num {current_pc_agg.shape[0]})")

                if len(history_pc_list) == history_pc_window_size:
                    history_pc_list.pop(0)
                data_dict = from_pkts_to_data_dict(current_pc_agg, current_frame_stat)
                history_pc_list.append(data_dict['pts'])

                if global_config.visualize:
                    try:
                        # data_dict = {
                        #     'pts': current_pc_agg,
                        #     'gps_coord': (longitude, latitude)
                        # }
                        data_dict['pts'] = np.concatenate([data_dict['pts'], *history_pc_list])
                        vis_queue.put(data_dict, block=False)
                    except queue.Full:
                        print("vis queue full!")

                # print(f"region stats: {current_frame_stat}")
                # reset current pc agg
                current_pc_agg = np.zeros((1, 4))
                current_frame_id = frame_i
                current_frame_stat = {region_id: {'count': 1, 'rssi': RSSI}}

            else:
                # append to pc list
                current_pc_agg = np.concatenate([current_pc_agg, pc])


def data_to_region_receiver(global_config: D2RReceiverConfig, vis_queue: Queue):
    sniff_source = SniffSource(iface=global_config.monitor_iface, filter='type data')
    sniff_sink = QueueSink()
    t = Thread(target=sink_parser, args=(global_config, sniff_sink, vis_queue,))
    # t.daemon = True
    t.start()

    try:
        sniff_source > sniff_sink
        p = PipeEngine(sniff_source)
        p.start()
        p.wait_and_stop()
    except (KeyboardInterrupt, SystemExit):
        p.stop()


class D2RReceiver(PCSharingBase):
    def __init__(self, global_config: D2RReceiverConfig):
        self.config = global_config
        print(self.config)

        # init phy interface
        subprocess.run(['sudo', 'iw', global_config.monitor_iface, 'set', 'monitor', 'none'], check=True)
        subprocess.run(['sudo', 'ifconfig', global_config.monitor_iface, 'up'], check=True)
        # iw dev {dev_name} set channel <channel> [HT20|HT40+|HT40-]
        subprocess.run(['sudo', 'iw', 'dev', global_config.monitor_iface, 'set', 'channel',
                        str(global_config.monitor_channel)], check=True)

        # init visualizer
        self.vis_queue = Queue(maxsize=10)
        vis_proc = Process(target=pc_visualize_bev,
                           args=(global_config, self.vis_queue,))

        # init d2r receiver
        d2r_receiver_proc = Process(target=data_to_region_receiver,
                                    args=(global_config, self.vis_queue,))

        self.main_proc = [d2r_receiver_proc]
        self.main_queue = []
        self.main_queue_name = []

        if global_config.visualize:
            # add visualization process and queue to the launch set
            self.main_proc.append(vis_proc)
            self.main_queue.append(self.vis_queue)
            self.main_queue_name.append('vis_queue')


if __name__ == '__main__':
    # set to avoid hanging in laptop
    multiprocessing.set_start_method('spawn')

    global_config = D2RReceiverConfig()
    global_config.parse_args(sys.argv)
    # print(global_config)
    receiver = D2RReceiver(global_config)
    receiver.run()
