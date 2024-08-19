from BeamDrop.AttentionMap import AttentionMap
from BeamDrop.D2RConfig import D2RConfig
from scipy.constants import c, pi
from math import e, cos
import numpy as np

np.set_printoptions(suppress=True)

# --- radio config ---
rfreq = 2.427e9
antenna_spacing = 0.06
angular_region_num = 2
tx_num = 4
tx_positions = np.array([[antenna_spacing * x, 0] for x in range(tx_num)])
tx_weights = lambda aod: [e ** (1j * 2 * pi * rfreq * i * antenna_spacing * cos(aod) / c)
                          for i in range(tx_num)]

# --- cell area config ---
beamformer_angles = np.linspace(40, 140, angular_region_num)
beamformer_weights = np.array([tx_weights(np.deg2rad(direction)) for direction in beamformer_angles])
print(f"Steering Angles: {beamformer_angles}")
print(f"TX Weights: {beamformer_weights}")

# --- assign data to beams ---
assign_mode = "Nearest"  # "Nearest", "Swap", "Random"
attention_map = AttentionMap(grid_width=1, tx_num=4, antenna_spacing=antenna_spacing, map_range=(-50, 50, -50, 50))
attention_map.tx_positions = np.array([[antenna_spacing * x, 0] for x in range(attention_map.tx_num)])
lamb = c / attention_map.radio_freq
print(attention_map.receiver_mask)

beam_info = {}
cell_grids = []
x_grids, y_grids = 100, 100
beam_img = np.zeros((x_grids * y_grids))

for cell_id in range(angular_region_num):
    # calculate received cells of each cell beam
    attention_map.update_phy_chan(tx_weights=beamformer_weights[cell_id])
    channel_power = np.abs(np.sum(attention_map.channel, axis=1))

    # find the serving grids of each cell
    cutoff_snr = np.percentile(channel_power, 65)
    cell_mask = np.where(channel_power > cutoff_snr)[0]
    beam_img[cell_mask] = cell_id + 1

beam_info = {}
if assign_mode == 'Nearest':
    # assign data grids overlapping with the receiver grids
    from scipy.spatial import cKDTree

    # search the nearest receiver grids of data grids
    tree = cKDTree(attention_map.receiver_grids)
    distances, indices = tree.query(attention_map.data_grids)
    data_to_beam = beam_img[indices]
    for cell_id in range(angular_region_num):
        class_grids = np.where(data_to_beam == cell_id + 1)[0]
        beam_info[cell_id] = {
            "v": beamformer_weights[cell_id],
            "grid_idx": class_grids
        }

d2r_conf = D2RConfig(region_name=f'demo_{assign_mode}')
d2r_conf.load_from_beamformer_output(beam_info, attention_map.data_grids)
print(d2r_conf)
d2r_conf.save_to_file(f"d2r_configs/d2r_demo_{assign_mode}.mat")
