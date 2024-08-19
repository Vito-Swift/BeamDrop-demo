import time
import sys
import subprocess
import queue
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue, freeze_support
from scapy.compat import raw
from BeamDrop.hw_beamformer.beamctl import BeamCtl
from BeamDrop.hw_80211 import hw_80211
from BeamDrop.Network import (BDP, BDPPCSegment)
from BeamDrop.D2RConfig import D2RConfig
from BeamDrop.runtime.pc_reader import PCReader
from BeamDrop.runtime.config import D2RSharerConfig
from BeamDrop.runtime.utils import PCSharingBase


def data_to_region_sharer(global_config: D2RSharerConfig, pc_queue: Queue):
    # read d2r config
    d2r_beam_config = D2RConfig()
    d2r_beam_config.load_from_file(global_config.d2r_conf)
    print(d2r_beam_config)

    # init phy interface
    subprocess.run(['sudo', 'ifconfig', global_config.monitor_iface, 'up'], check=True)
    # iw {dev_name} set monitor none
    subprocess.run(['sudo', 'iw', global_config.monitor_iface, 'set', 'monitor', 'none'], check=True)
    # iw dev {dev_name} set txpower fixed 4000
    subprocess.run(['sudo', 'iw', 'dev', global_config.monitor_iface, 'set', 'txpower', 'fixed',
                    str(global_config.txpower)], check=True)
    # iw dev {dev_name} set channel <channel> [HT20|HT40+|HT40-]
    subprocess.run(['sudo', 'iw', 'dev', global_config.monitor_iface, 'set', 'channel',
                    str(global_config.monitor_channel)], check=True)

    iface_mac_int = int("0x" + global_config.iface_mac.replace(':', ''), 16)
    wiphy = hw_80211.init_phy(iface=global_config.monitor_iface,
                              mac_addr=iface_mac_int)

    # init beamformer interface
    bf_phy = BeamCtl(bfhw_dev=global_config.bfhw_dev,
                     bfhw_baudrate=global_config.bfhw_baudrate,
                     tx_power=global_config.txpower)
    bf_phy.store_beams(d2r_beam_config.region_beams)

    while True:
        # 1. Fetch new frame from reader's queue
        if not pc_queue.empty():
            try:
                frame_i, pc, trans_mat = pc_queue.get_nowait()
            except queue.Empty:
                print('No pc frame in reader queue')
                continue
        else:
            continue
        print(pc)        
        trans_mat_byte = trans_mat.astype(np.float32).tobytes()
        pc = pc[:, :4].astype(np.float32)
        d2r_beam_config.load_base_pc(pc)

        # get sender gps coordinates
        x = y = 0

        for region_id in range(d2r_beam_config.region_num):
            # 2. Switch to different beams
            bf_phy.switch_beam(region_id)

            # 3. send data via broadcast interface
            region_pts = d2r_beam_config.get_region_points(region_id)

            # create multiple sub packets to transmit the regional point cloud
            sp_size = 1250
            bytes_per_point = 16
            point_per_sp = int(np.floor(sp_size / bytes_per_point))
            unsent_points = region_pts.shape[0]
            count = 0
            batched_msg = b""
            msg_length = []
            while unsent_points > 0:
                count += 1
                sent_points = point_per_sp if unsent_points >= point_per_sp else unsent_points
                start_idx = region_pts.shape[0] - unsent_points
                p_seg = (BDP(type=2, deviceMAC=global_config.iface_mac) /
                         BDPPCSegment(frame_id=frame_i, timestamp_ns=time.time_ns(),
                                      transition_mat=trans_mat_byte, point_num=sent_points,
                                      point_bpp=bytes_per_point,
                                      longitude=x, latitude=y,
                                      point_cloud=region_pts[start_idx:start_idx + sent_points].tobytes(),
                                      region_id=region_id))
                batched_msg += raw(p_seg)
                msg_length.append(len(raw(p_seg)))
                unsent_points -= sent_points
            hw_80211.inject_multiple_packets(wiphy, msg_length, batched_msg, global_config.monitor_rate)


class D2RSharer(PCSharingBase):
    def __init__(self, global_config: D2RSharerConfig):
        self.config = global_config
        print(self.config)

        # init point cloud reader
        point_cloud_reader = PCReader(config=global_config)
        point_cloud_reader_proc = Process(target=point_cloud_reader.run)
        # init d2r sharer
        d2r_sharer_proc = Process(target=data_to_region_sharer,
                                  args=(global_config, point_cloud_reader.oqueue,))

        self.main_proc = [point_cloud_reader_proc, d2r_sharer_proc]
        self.main_queue = [point_cloud_reader.oqueue]
        self.main_queue_name = ['reader_oqueue']


if __name__ == '__main__':
    freeze_support()
    #multiprocessing.set_start_method('spawn')
    global_config = D2RSharerConfig()
    global_config.parse_args(sys.argv)
    sharer = D2RSharer(global_config)
    sharer.run()
