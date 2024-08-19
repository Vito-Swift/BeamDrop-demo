from BeamDrop.D2RConfig import D2RConfig
from BeamDrop.AttentionMap import AttentionMap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np


def plot_agg_cell_data_on_axes(ax_list: plt.axes, d2r_file_path: str):
    d2r_config = D2RConfig()
    d2r_config.load_from_file(filepath=d2r_file_path)

    # plot cell data
    for cell_id in range(d2r_config.region_num):
        ax_cell = ax_list[cell_id]
        cell_img_width = np.sqrt(d2r_config.data_grids.shape[0]).astype(int)
        cell_img = np.zeros(d2r_config.data_grids.shape[0])
        cell_img[d2r_config.region_data_grids[cell_id]] = 1

        ax_cell.set_title(f"Cell Data {cell_id + 1}", weight='bold', fontsize=16)
        ax_cell.imshow(cell_img.reshape((cell_img_width, cell_img_width)).T, cmap='Reds')
        ax_cell.set_xlabel("X (m)")
        ax_cell.set_ylabel("Y (m)")

        # major ticks
        ax_cell.set_xticks([0, cell_img_width])
        ax_cell.set_yticks([0, cell_img_width])
        ax_cell.set_xlim([0, cell_img_width])
        ax_cell.set_ylim([0, cell_img_width])
        ax_cell.set_xticklabels([-50, 50])
        ax_cell.set_yticklabels([-50, 50])

        # minor ticks
        ax_cell.set_xticks(np.arange(-.5, cell_img_width, 1), minor=True)
        ax_cell.set_yticks(np.arange(-.5, cell_img_width, 1), minor=True)

        ax_cell.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax_cell.tick_params(which='minor', bottom=False, left=False)


def plot_agg_beam_on_ax(ax, d2r_file_path: str):
    d2r_config = D2RConfig()
    d2r_config.load_from_file(filepath=d2r_file_path)

    antenna_spacing = 0.06
    attention_map = AttentionMap(grid_width=1, tx_num=4, antenna_spacing=antenna_spacing, map_range=(-50, 50, -50, 50))
    x_grids, y_grids = 100, 100
    beam_img = np.zeros((x_grids * y_grids))
    for cell_id in range(d2r_config.region_num):
        # calculate received cells of each cell beam
        attention_map.update_phy_chan(tx_weights=d2r_config.region_beams[cell_id])
        channel_power = np.abs(np.sum(attention_map.channel, axis=1))

        # find the serving grids of each cell
        cutoff_snr = np.percentile(channel_power, 65)
        cell_mask = np.where(channel_power > cutoff_snr)[0]
        beam_img[cell_mask] = cell_id + 1

    beam_img = beam_img.reshape((x_grids, y_grids))

    # plot beam pattern
    cellbeam_img = ax.imshow(beam_img)
    ax.axis('off')
    ax.set_title('Cell Beam Pattern', weight='bold', fontsize=20)
    cellbeam_colors = cellbeam_img.cmap(cellbeam_img.norm(np.unique(beam_img)))
    ax_beamimg_legend = [
        Patch(facecolor=cb_color, edgecolor='black', label=f'Cell {cell_id + 1}')
        for cell_id, cb_color in enumerate(cellbeam_colors[1:])
    ]
    ax.legend(handles=ax_beamimg_legend, loc='upper right')


def plot_beam_pattern_on_fig(figure: plt.Figure, d2r_file_path: str):
    d2r_config = D2RConfig()
    d2r_config.load_from_file(filepath=d2r_file_path)

    beam_pattern_rows = 3
    max_cols = 3
    max_rows = np.ceil(d2r_config.region_num / max_cols).astype(int) + beam_pattern_rows
    gs = GridSpec(max_rows, max_cols, figure=figure)

    # plot beam pattern
    ax_beamimg = figure.add_subplot(gs[:beam_pattern_rows, :beam_pattern_rows])
    plot_agg_beam_on_ax(ax_beamimg, d2r_file_path)

    ax_cells = []
    for cell_id in range(d2r_config.region_num):
        row_id = cell_id // max_cols + beam_pattern_rows
        col_id = cell_id % max_cols
        ax_cell = figure.add_subplot(gs[row_id, col_id])
        ax_cells.append(ax_cell)
    plot_agg_cell_data_on_axes(ax_cells, d2r_file_path)


if __name__ == '__main__':
    # plot layout
    fig = plt.figure(layout="constrained")
    assign_mode = 'Nearest'
    d2r_config_path = f'd2r_configs/d2r_demo_{assign_mode}.mat'
    plot_beam_pattern_on_fig(fig, d2r_config_path)
    plt.show()
