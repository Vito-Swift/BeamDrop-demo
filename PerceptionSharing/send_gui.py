from BeamDrop.D2RConfig import D2RConfig
from BeamDrop.runtime.pc_reader import PCReader
from BeamDrop.runtime.config import D2RSharerConfig
from BeamDrop.hw_beamformer.beamctl import BeamCtl
from BeamDrop.hw_80211 import hw_80211
from BeamDrop.Network import (BDP, BDPPCSegment)
from BeamDrop.visualization.simple_plot3d.canvas_bev import Canvas_BEV
from scapy.compat import raw
import time
import queue
import subprocess
from multiprocessing import Process, Queue, freeze_support
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from plot_demo_d2r_conf import plot_agg_beam_on_ax, plot_agg_cell_data_on_axes
from PyQt6 import QtCore, QtWidgets
import sys
import cv2


class D2RSharerGUI(object):
    def __init__(self, config: D2RSharerConfig, pc_queue: Queue, gui_queue: Queue):
        super(D2RSharerGUI, self).__init__()
        self.quit_flag = False

        self.config = config
        self.pc_queue = pc_queue
        self.gui_queue = gui_queue

    def run(self):
        # read d2r config
        d2r_beam_config = D2RConfig()
        d2r_beam_config.load_from_file(self.config.d2r_conf)
        print(d2r_beam_config)

        # init phy interface
        subprocess.run(['sudo', 'ifconfig', self.config.monitor_iface, 'up'], check=True)
        # iw {dev_name} set monitor none
        subprocess.run(['sudo', 'iw', self.config.monitor_iface, 'set', 'monitor', 'none'], check=True)
        # iw dev {dev_name} set txpower fixed 4000
        subprocess.run(['sudo', 'iw', 'dev', self.config.monitor_iface, 'set', 'txpower', 'fixed',
                        str(self.config.txpower)], check=True)
        # iw dev {dev_name} set channel <channel> [HT20|HT40+|HT40-]
        subprocess.run(['sudo', 'iw', 'dev', self.config.monitor_iface, 'set', 'channel',
                        str(self.config.monitor_channel)], check=True)

        iface_mac_int = int("0x" + self.config.iface_mac.replace(':', ''), 16)
        wiphy = hw_80211.init_phy(iface=self.config.monitor_iface,
                                  mac_addr=iface_mac_int)

        # init beamformer interface
        bf_phy = BeamCtl(bfhw_dev=self.config.bfhw_dev,
                         bfhw_baudrate=self.config.bfhw_baudrate,
                         tx_power=self.config.txpower)
        bf_phy.store_beams(d2r_beam_config.region_beams)

        while True:
            if self.quit_flag:
                break

            # 1. Fetch new frame from reader's queue
            if not self.pc_queue.empty():
                try:
                    frame_i, pc, trans_mat = self.pc_queue.get_nowait()
                except queue.Empty:
                    print('No pc frame in reader queue')
                    continue
            else:
                continue

            if not self.gui_queue.full():
                self.gui_queue.put(pc, block=False)

            trans_mat_byte = trans_mat.astype(np.float32).tobytes()
            pc = pc.astype(np.float32)
            d2r_beam_config.load_base_pc(pc)

            # get sender gps coordinates
            x = y = 0

            for region_id in range(d2r_beam_config.region_num):
                # 2. Switch to different beams
                # bf_phy.switch_beam(region_id)

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
                # hw_80211.inject_multiple_packets(wiphy, msg_length, batched_msg, global_config.monitor_rate)


class BeamDropSenderApp(QtWidgets.QMainWindow):

    def __init__(self, config: D2RSharerConfig, *args, **kwargs):
        self.config = config
        print(self.config)

        super(BeamDropSenderApp, self).__init__(*args, **kwargs)
        self._main = QtWidgets.QWidget

        self.gui_queue = Queue(maxsize=10)
        self.__launch_pc_reader()
        self.__launch_d2r_sharer()
        self.__init_figure_canvas()

        self.d2r_config_path = self.config.d2r_conf
        self.d2r_config = D2RConfig()
        self.d2r_config.load_from_file(filepath=self.d2r_config_path)

        self.__update_d2r_config_on_canvas()
        self.__init_pcd_stream_on_canvas()
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Q:
            self.close()

    def __init_figure_canvas(self):
        plt.style.use('dark_background')

        self.canvas = FigureCanvas(Figure(layout='constrained', figsize=(12, 12)))
        self.max_cell_num = 3
        self.figure_beam_pattern_rows = 2
        self.figure_max_cell_cols = 4
        self.figure_max_cell_rows = np.ceil(self.max_cell_num / self.figure_max_cell_cols).astype(int)
        self.figure_grid_spec = GridSpec(self.figure_max_cell_rows + self.figure_beam_pattern_rows,
                                         self.figure_max_cell_cols, hspace=1, wspace=1,
                                         height_ratios=[1] * self.figure_beam_pattern_rows +
                                                       [1.5] * self.figure_max_cell_rows)
        # axes for beam img
        self.figure_ax_beamimg = self.canvas.figure.add_subplot(
            self.figure_grid_spec[:self.figure_beam_pattern_rows, :self.figure_beam_pattern_rows]
        )
        # axes for point cloud stream
        self.figure_ax_pcd_stream = self.canvas.figure.add_subplot(
            self.figure_grid_spec[:self.figure_beam_pattern_rows, self.figure_beam_pattern_rows:]
        )
        # axes for data distribution pattern
        self.figure_ax_cells = []
        for cell_id in range(self.max_cell_num):
            row_id = cell_id // self.figure_max_cell_cols + self.figure_beam_pattern_rows
            col_id = cell_id % self.figure_max_cell_cols
            ax_cell = self.canvas.figure.add_subplot(self.figure_grid_spec[row_id, col_id])
            self.figure_ax_cells.append(ax_cell)

        # add the figure canvas to QT backend
        self.setCentralWidget(self.canvas)

    def __update_d2r_config_on_canvas(self):
        plot_agg_beam_on_ax(self.figure_ax_beamimg, self.d2r_config_path)
        plot_agg_cell_data_on_axes(self.figure_ax_cells, self.d2r_config_path)

    def __launch_pc_reader(self):
        self.point_cloud_reader = PCReader(config=self.config)
        self.point_cloud_reader_proc = Process(target=self.point_cloud_reader.run)
        self.point_cloud_reader_proc.daemon = True
        self.point_cloud_reader_proc.start()

    def __launch_d2r_sharer(self):
        self.d2rsharer = D2RSharerGUI(config=self.config, pc_queue=self.point_cloud_reader.oqueue,
                                      gui_queue=self.gui_queue)
        self.d2r_sharer_proc = Process(target=self.d2rsharer.run)
        self.d2r_sharer_proc.daemon = True
        self.d2r_sharer_proc.start()

    def __update_pc_imshow(self):
        if not self.gui_queue.empty():
            pc = self.gui_queue.get_nowait()
            pc_bev_canvas = Canvas_BEV(self.bev_canvas_shape, left_hand=False)
            canvas_xy, valid_mask = pc_bev_canvas.get_canvas_coords(pc[::3])
            pc_bev_canvas.draw_canvas_points(canvas_xy, radius=1)
            _canvas = cv2.resize(pc_bev_canvas.canvas, self.bev_canvas_shape)
            self.figure_pcd_im.set_array(_canvas)
            self.canvas.draw()

    def __init_pcd_stream_on_canvas(self):
        self.figure_ax_pcd_stream.axis('off')
        self._timer = self.canvas.new_timer(50)
        self._timer.add_callback(self.__update_pc_imshow)
        self._timer.start()

        # init BEV canvas
        self.bev_canvas_shape = (800, 800)
        # init imshow
        self.figure_pcd_im = self.figure_ax_pcd_stream.imshow(np.zeros(self.bev_canvas_shape))
        self.figure_ax_pcd_stream.set_title("Point Cloud Stream", weight='bold', fontsize=20)


if __name__ == '__main__':
    freeze_support()
    # multiprocessing.set_start_method('spawn')
    global_config = D2RSharerConfig()
    global_config.parse_args(sys.argv)
    app = QtWidgets.QApplication(sys.argv)
    w = BeamDropSenderApp(global_config)
    app.exec()
