#[unet]
'''
unet_floors_num_layers = [2, 2, 2]
unet_first_floor_layers_num_kernels = 64
unet_layers_kernel_shape = [3, 3]
unet_layers_pool_shape = [2, 2]
unet_layers_pool_stride = [2, 2]
unet_training_level_by_level = False
'''

#[dataset]
# dataset_eval_batch_size = 16

#[database]
# database_data_type = 'HCQT_TO_F0'
# database_data_chunks_duration_in_sec = 3.0
# database_data_chunks_overlap_in_sec = [0.5, 0.5]
# database_data_chunks_padding_type = 'SAME'
database_src_data_shape = [258, 360, 6]
database_tgt_data_shape = [258, 360]
# mine
data_chunks_duration_in_bins = 258
data_chunks_overlap_in_bins = (25,25) # beginning and end

#[train - test - validation split]
train_set_fold_index = [0,1,2,3,4,5,6,7]
validation_set_fold_index = [8]
test_set_fold_index = [9]

#[corpus]
# corpus_data_sec_to_bins = 86.1328125

#[hcqt]
hcqt_frequency_min_in_hz = 32.7
hcqt_num_bins = 360
hcqt_num_octaves = 6
hcqt_num_bins_per_octave = 60
hcqt_hop_length_in_bins = 256
hcqt_harmonics = [0.5, 1, 2, 3, 4, 5]
hcqt_f0_gaussian_blur = True

#[audio]
audio_sampling_rate_in_hz = 22050

#[training configuration]
batch_size = 16