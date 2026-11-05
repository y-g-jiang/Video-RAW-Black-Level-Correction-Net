import os
os.environ.update(TEMP='E:/Temp/bcm18', TMP='E:/Temp/bcm18', MPLCONFIGDIR='E:/Caches/ble_paper_matplotlib')
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'source'
F = R / 'figures'
E = ROOT / 'evidence'
D = ROOT.parent / 'z63_unclipped_demo_20261029'

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_figure(language):
    verified = json.loads((D / 'dark' / 'VERIFIED.json').read_text('utf8'))
    assert verified['passed'] and verified['no_clipping'] and verified['naf_inference'] is False
    assert sha(D / 'dark' / 'T7_RAW_AND_GT.npz') == verified['raw_sample_sha256']

    spec = importlib.util.spec_from_file_location('paper_demo_source_v9', D / 'build_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DARK = True

    module.SCENE_MEDIAN_DN = 2.0
    source_geometry = module.setup()
    trajectory, _ = module.labels()
    with np.load(D / 'dark' / 'SEQUENCE_METADATA.npz') as source:
        assert np.array_equal(source['delta_dn'], trajectory)
    with np.load(D / 'dark' / 'T7_RAW_AND_GT.npz') as source:
        replay_clean = module.frame_signal(285, source_geometry)
        _, replay_q = module.S.observe(replay_clean[None], trajectory[285:286], 700285)
        replay = (replay_q[0] - module.S.reference[:, None, None]).astype(np.float32)
        assert np.array_equal(replay, source['raw_signed_dn'][2, 0])

    # Four predeclared positions within the empirical Z63-derived interval.
    # The staged zero/constant/step phases are never displayed or scored here.
    frames = np.array([255, 295, 335], dtype=np.int32)
    assert np.all((frames >= 240) & (frames < 360))

    source_scene_median_dn = 8.0
    scene_median_dn = 1.0
    module.SCENE_MEDIAN_DN = scene_median_dn
    geometry = module.setup()
    display_ev = 3
    display_white_dn = 48.0 / (2 ** display_ev)

    clean_frames = []
    raw_frames = []
    offsets = trajectory[frames]
    algebra_error = 0.0
    for t in frames:
        clean = module.frame_signal(int(t), geometry)
        latent, quantized = module.S.observe(clean[None], trajectory[t:t + 1], 700000 + int(t))
        latent = latent[0]
        raw = quantized[0] - module.S.reference[:, None, None]
        read_noise = latent - clean - module.S.reference[:, None, None] - trajectory[t, :, None, None]
        quantization_residual = quantized[0] - latent
        algebra_error = max(algebra_error, float(np.max(np.abs(
            raw - trajectory[t, :, None, None] - (clean + read_noise + quantization_residual)))))
        clean_frames.append(clean.astype(np.float32))
        raw_frames.append(raw.astype(np.float32))
    clean_frames = np.stack(clean_frames)
    raw_frames = np.stack(raw_frames)
    assert algebra_error < 1e-10

    def frame_radiance_rgb(frame):
        canvas, shift, angle, _, _, _, zoom = geometry
        height, width = module.H, module.W
        yy, xx = np.mgrid[:2 * height, :2 * width].astype(float)
        yy -= height - 0.5
        xx -= width - 0.5
        radians = np.deg2rad(angle[frame])
        y = (np.cos(radians) * yy - np.sin(radians) * xx) / zoom[frame] + 1.5 * height - 0.5 + 2 * shift[frame, 0]
        x = (np.sin(radians) * yy + np.cos(radians) * xx) / zoom[frame] + 1.5 * width - 0.5 + 2 * shift[frame, 1]
        sampled = np.stack([map_coordinates(canvas[..., c], [y, x], order=1, mode='nearest', prefilter=False)
                            for c in range(3)], axis=-1)
        return sampled.reshape(height, 2, width, 2, 3).mean((1, 3))

    def to_srgb(rgb):
        rgb = rgb / display_white_dn
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
        return np.clip(rgb, 0, 1)

    def average_native(x, native_size, roi_native=None):
        packed_size = native_size // 2
        if roi_native is not None:
            y0, y1, x0, x1 = roi_native
            x = x[:, y0 // 2:y1 // 2, x0 // 2:x1 // 2]
        _, h, w = x.shape
        assert h % packed_size == 0 and w % packed_size == 0
        return x.reshape(4, h // packed_size, packed_size, w // packed_size, packed_size).mean((2, 4))

    def render_rgb(x, native_size, roi_native=None):
        averaged = average_native(x, native_size, roi_native)
        rgb = np.stack((averaged[0], (averaged[1] + averaged[2]) * 0.5, averaged[3]), axis=-1)
        return to_srgb(rgb)

    detail_roi_native = (144, 288, 210, 462)
    y0, y1, x0, x1 = detail_roi_native
    overview_roi = (x0 - 0.5, y0 - 0.5, x1 - x0, y1 - y0)
    if language == 'en':
        titles = ['Empirical Z63-derived sequence'] * 4
        row_labels = ['Overview', 'Radiance\nproxy', 'Signed noisy\nRAW', 'Known-offset\ncorrection']
        x_label, y_label = 'Frame (29.97 frames/s)', 'Injected offset (DN)'
        output = F / 'video_demo_z63_empirical_en_v12'
    else:
        titles = ['Z63 暗场轨迹衍生序列'] * 4
        row_labels = ['全景', '辐射代理', '带符号含噪\nRAW', '已知偏移\n校正']
        x_label, y_label = '帧（29.97 帧/秒）', '注入偏移（DN）'
        output = F / 'video_demo_z63_empirical_zh_v12'

    font = 'DejaVu Sans' if language == 'en' else 'Microsoft YaHei'
    with plt.rc_context({'font.family': font, 'axes.unicode_minus': False}):
        fig = plt.figure(figsize=(7.16, 5.10))
        grid = fig.add_gridspec(5, 3, height_ratios=[0.9, 0.78, 0.78, 0.78, 0.76],
                                left=0.12, right=0.995, top=0.965, bottom=0.09,
                                wspace=0.035, hspace=0.28)
        overview_frame = 295
        overview_index = int(np.flatnonzero(frames == overview_frame)[0])
        overview_linear = frame_radiance_rgb(overview_frame)
        overview = fig.add_subplot(grid[0, 0])
        overview.imshow(to_srgb(overview_linear), interpolation='nearest')
        overview.add_patch(Rectangle((overview_roi[0], overview_roi[1]), overview_roi[2], overview_roi[3],
                                     fill=False, edgecolor='#d33535', linewidth=0.85))
        overview.set_xticks([])
        overview.set_yticks([])
        overview.set_title(f'Radiance proxy · f = {overview_frame}' if language == 'en'
                           else f'辐射代理 · 第 {overview_frame} 帧', fontsize=7.8, pad=3)
        overview.set_ylabel(row_labels[0], rotation=0, labelpad=0, fontsize=7.2, va='center', ha='right')
        overview.yaxis.set_label_coords(-0.12, 0.5)
        for spine in overview.spines.values():
            spine.set_visible(True)
            spine.set_color('#c9cfd4')
            spine.set_linewidth(0.45)

        note = fig.add_subplot(grid[0, 1:])
        note.axis('off')
        note_title = 'Z63-derived simulation' if language == 'en' else 'Z63 暗场轨迹衍生模拟'
        note_lines = (['Scene median: 1 DN (-3 EV from 8 DN)', 'Fixed display gain: +3 EV',
                       'Red box: 252 x 144 native-pixel ROI', 'Detail blocks: 12 x 12 native pixels']
                      if language == 'en' else
                      ['场景中位数：1 DN（相对 8 DN 源场景为 -3 EV）', '固定显示增益：+3 EV',
                       '红框：252 x 144 原生像素裁区', '细节块：12 x 12 原生像素'])
        note.text(0.05, 0.82, note_title, ha='left', va='top', fontsize=8.2, weight='bold')
        note.text(0.05, 0.63, '\n'.join(note_lines), ha='left', va='top', fontsize=7.6, linespacing=1.5)
        for j, frame in enumerate(frames):
            full_linear = frame_radiance_rgb(int(frame))
            roi = full_linear[y0:y1, x0:x1]
            assert roi.shape[:2] == (144, 252)
            roi_binned = roi.reshape(12, 12, 21, 12, 3).mean((1, 3))
            radiance_detail = to_srgb(roi_binned)
            raw_detail = render_rgb(raw_frames[j], 12, detail_roi_native)
            corrected_detail = render_rgb(raw_frames[j] - offsets[j, :, None, None], 12, detail_roi_native)
            for row, image in [(1, radiance_detail), (2, raw_detail), (3, corrected_detail)]:
                ax = fig.add_subplot(grid[row, j])
                ax.imshow(image, interpolation='nearest')
                if row == 1:
                    ax.set_title(f'f = {frame}', fontsize=7.8, pad=3)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(row_labels[row], rotation=0, labelpad=0, fontsize=7.2, va='center', ha='right')
                    ax.yaxis.set_label_coords(-0.12, 0.5)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#c9cfd4')
                    spine.set_linewidth(0.45)

        ax = fig.add_subplot(grid[4, :])
        colors = ['#ba4039', '#2e8058', '#95a934', '#3574a7']
        for c, color in enumerate(colors):
            ax.plot(np.arange(240, 360), trajectory[240:360, c], lw=0.9, color=color, label=['R', 'G1', 'G2', 'B'][c])
        for frame in frames:
            ax.axvline(frame, color='#44515c', lw=0.65, ls=':')
        ax.set(xlim=(240, 359), ylim=(-2, 2), yticks=[-2, 0, 2], xlabel=x_label, ylabel=y_label)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis='y', alpha=0.18)
        ax.legend(ncol=4, frameon=False, loc='upper right', fontsize=7.5,
                  handlelength=1.2, columnspacing=1)
        fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.045)
        fig.savefig(output.with_suffix('.png'), bbox_inches='tight', pad_inches=0.045,
                    dpi=230, facecolor='white')
        plt.close(fig)

    arrays_path = E / 'FIG2_Z63_EMPIRICAL_FRAMES_V12.npz'
    np.savez_compressed(arrays_path, clean_dn=clean_frames, raw_signed_dn=raw_frames,
                        delta_dn=offsets, frame_indices=frames,
                        read_noise_sigma_dn=module.S.sigma.astype(np.float32))
    record = {
        'revision': 'fig2-z63-empirical-v12',
        'source_generator': str(D / 'build_demo.py'),
        'source_generator_sha256': sha(D / 'build_demo.py'),
        'source_verified': verified,
        'existing_dark_frame285_replayed_exact': True,
        'source_canvas_median_dn': source_scene_median_dn,
        'scene_canvas_median_dn': scene_median_dn,
        'scene_exposure_relative_source_ev': -3,
        'display_gain_ev': display_ev,
        'display_white_source_dn': 48.0,
        'display_white_after_gain_dn': display_white_dn,
        'display_spatial_bin_native_overview': [1, 1],
        'display_spatial_bin_native_detail': [12, 12],
        'overview_representation': 'Native-resolution linear RGB radiance proxy before CFA sampling.',
        'display_interpolation': 'nearest',
        'detail_roi_native_yx': [[y0, y1], [x0, x1]],
        'detail_grid_shape_yx': [12, 21],
        'displayed_cell_area_ratio_to_v9': 1.0 / 9.0,
        'cropped_radiance_and_detail_coordinates_identical': True,
        'detail_roi_selection': 'Fixed shoreline and pier region; identical pixel bounds for the frame-295 overview marker and all three displayed crops.',
        'single_overview_frame': overview_frame,
        'selected_frame_indices': frames.tolist(),
        'selection_policy': 'Three fixed frames 255, 295, 335 within the Z63-derived empirical interval 240-359.',
        'read_noise_sigma_dn': module.S.sigma.tolist(),
        'poisson_noise': False,
        'offset_and_read_noise_scale': 'Original DN; unchanged when scene exposure is reduced.',
        'correction': 'Subtract the known injected offset from each CFA channel; retain the same read-noise and quantization realization.',
        'oracle_algebra_max_error': algebra_error,
        'network_output_included': False,
        'sealed_test_used': False,
        'figure_sha256_en': sha(F / 'video_demo_z63_empirical_en_v12.pdf') if (F / 'video_demo_z63_empirical_en_v12.pdf').exists() else None,
        'figure_sha256_zh': sha(F / 'video_demo_z63_empirical_zh_v12.pdf') if (F / 'video_demo_z63_empirical_zh_v12.pdf').exists() else None,
        'figure_arrays_sha256': sha(arrays_path),
    }
    (E / 'Z63_VIDEO_DEMO_EMPIRICAL_V12.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), 'utf8')


def update_manifest():
    manifest = []
    for directory in [E, F]:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                manifest.append({'path': str(path.relative_to(R)), 'sha256': sha(path), 'bytes': path.stat().st_size})
    (R / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf8')


if __name__ == '__main__':
    build_figure('en')
    build_figure('zh')
    print('Generated Figure 2 from the empirical interval of the verified synthetic source sequence.')
