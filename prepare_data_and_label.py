#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display
import IPython.display as ipd
import medleydb as mdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# # ground truth

# In[2]:


import argparse
from joblib import Parallel, delayed
import librosa
import medleydb as mdb
from medleydb import mix
import numpy as np
import os
from scipy.signal import upfirdn
from scipy.ndimage import filters
import sox


# In[3]:


def get_hcqt_params():
    """Hack to always use the same parameters :)
    """
    bins_per_octave = 60
    n_octaves = 6
    harmonics = [0.5, 1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length

def compute_hcqt(audio_fpath):
    """Compute the harmonic CQT from a given audio file
    """
    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length) = get_hcqt_params()
    y, fs = librosa.load(audio_fpath, sr=sr)

    cqt_list = []
    shapes = []
    for h in harmonics:
        cqt = librosa.cqt(
            y, sr=fs, hop_length=hop_length, fmin=f_min*float(h),
            n_bins=bins_per_octave*n_octaves,
            bins_per_octave=bins_per_octave
        )
        cqt_list.append(cqt)
        shapes.append(cqt.shape)

    shapes_equal = [s == shapes[0] for s in shapes]
    if not all(shapes_equal):
        min_time = np.min([s[1] for s in shapes])
        new_cqt_list = []
        for i in range(len(cqt_list)):
            new_cqt_list.append(cqt_list[i][:, :min_time])
        cqt_list = new_cqt_list

    log_hcqt = ((1.0/80.0) * librosa.core.amplitude_to_db(
        np.abs(np.array(cqt_list)), ref=np.max)) + 1.0

    return log_hcqt

def get_freq_grid():
    """Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies( #Compute the center frequencies of Constant-Q bins.
        bins_per_octave*n_octaves, f_min, bins_per_octave=bins_per_octave
    )
    return freq_grid

def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def create_annotation_target(freq_grid, time_grid, annotation_times,
                             annotation_freqs, gaussian_blur):
    """Create the binary annotation target labels
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1]) # dim += 1 分界线
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1]) # dim += 1 分界线
    # time_grid freq_grid的值表示每个像素代表范围的中间值
    # time_grid牵强，因为它的首元素是0……以后再说

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1

    if not gaussian_blur:
        return annotation_target
        '''can skip now'''
    else:
        annotation_target_blur = filters.gaussian_filter1d(
            annotation_target, 1, axis=0, mode='constant'
        )
        if len(annot_freq_idx) > 0:
            min_target = np.min(
                annotation_target_blur[annot_freq_idx, annot_time_idx]
            )
        else:
            min_target = 1.0

        annotation_target_blur = annotation_target_blur / min_target
        annotation_target_blur[annotation_target_blur > 1.0] = 1.0

        return annotation_target_blur
    

def get_input_output_pairs(audio_fpath, annot_times, annot_freqs,
                           gaussian_blur, precomputed_hcqt=None):
    # audio_fpath: mix path
    if precomputed_hcqt is None or not os.path.exists(precomputed_hcqt):
        print("    > computing CQT for {}".format(os.path.basename(audio_fpath)))
        hcqt = compute_hcqt(audio_fpath)
    else:
        print("    > using precomputed CQT for {}".format(os.path.basename(audio_fpath)))
        hcqt = np.load(precomputed_hcqt, mmap_mode='r')

    freq_grid = get_freq_grid()
    time_grid = get_time_grid(len(hcqt[0][0]))

    annot_target = create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs, gaussian_blur
    )

    return hcqt, annot_target, freq_grid, time_grid


def save_data(save_path, prefix, X, Y, f, t):
    input_path = os.path.join(save_path, 'inputs')
    output_path = os.path.join(save_path, 'outputs')
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    np.save(os.path.join(input_path, "{}_input.hcqt.npy".format(prefix)), X.astype(np.float32))
    np.save(os.path.join(output_path, "{}_output.npy".format(prefix)), Y.astype(np.float32))

    print("    Saved data for {} to {}".format(prefix, save_path))



def compute_melody2(mtrack, save_dir, gaussian_blur, precomputed_hcqt):
    data = mtrack.melody2_annotation
    if data is None:
        print("    {} No melody 2 data".format(mtrack.track_id))
    else:
        prefix = "{}_mel2".format(mtrack.track_id)

        input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
        if os.path.exists(input_path) and os.path.exists(output_path):
            print("    > already done!")
            return

        annot = np.array(data).T
        times = annot[0]
        freqs = annot[1]

        idx = np.where(freqs != 0.0)[0] # 返回满足条件的索引

        times = times[idx]
        freqs = freqs[idx]

        X, Y, f, t = get_input_output_pairs(
            mtrack.mix_path, times, freqs, gaussian_blur,
            precomputed_hcqt
        )
        save_data(save_dir, prefix, X, Y, f, t)


# In[ ]:


if __name__ == '__main__':
    
    list_108 = []
    for track in mdb.load_all_multitracks():
        if track.has_melody:
            list_108.append(track.track_id)
    assert(len(list_108)==108)

    for track_id in tqdm(list_108):
        a = mdb.MultiTrack(track_id)
        compute_melody2(a, '.', gaussian_blur=False,precomputed_hcqt=None)


# In[25]:



# In[ ]:




