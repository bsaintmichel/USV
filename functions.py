import os
import torch 
import json
import glob
import numpy as np
import math

from torch.nn import AvgPool1d
from numpy.lib.stride_tricks import sliding_window_view
from torchaudio.transforms import FFTConvolve, Convolve
from scipy.io import loadmat
from io import BufferedIOBase
from tqdm.notebook import tqdm
from IPython.display import Markdown, display

"""
Quick disclaimer : 

n_t is the number of pulses sent (and the number of received echoes) : usually Nb_tir x Nsequence
n_r or n_j0 or n_pts is the number of data points for each recorded pulse : formerly known as 'A'
n_z or n_i0 is the number of channels : it should be 128

"""

#####################################
############## PRETTY PRINTS #######

def printm(text:str, color=None):
    """ A pretty print using Markdown"""
    if color is None:
        colorstr = text
    else:
        colorstr = f"<span style='color:{color}'>{text}</span>"
    display(Markdown(colorstr))


def print_details(path:str, config:dict):
    """ Prints details about the experiment based on its path"""
    abspath = os.path.abspath(path)
    absrefpath = os.path.abspath(config['ref_path'])
    n_wavelengths = config['window'] / config['fsample'] * config['fpulse'] # There is a 1/2 factor in Vincent's/Thomas's code and I have no idea why...
    overlap = 1 - config['stride'] / config['window']
    rotor = config['right']
    device = config['device']

    printm('-----')
    printm(f'**Processing {abspath} : Using {device}**', color='green' if device == 'cuda' else 'orange')
    printm(f'Ref. used : {absrefpath}')
    printm(f'Window size = {n_wavelengths:.2f} λ | Overlap {overlap:.2f} | Rotor position {rotor} px')

######################################
############ FILE MANAGEMENT #########

def find_files(path:str, ext='dat'):
    """ Finds Speckle files using glob and sorts them numerically"""
    files = np.array(glob.glob(path + '/*'+ ext))
    filenum = lambda file : int(file.split('.')[0].split('_')[-1])
    files = sorted(files, key=filenum)
    return files

    

def update_config(config:dict, prms:dict, save_path=None, ref_path=None) -> dict:
    """ Updates the `config` dictionary
    to include data and information from data processing
    (e.g. correlation window size, rotor position, etc.)
    
    Also performs a few checks, notably with the left and right trims
    for the data"""

    for key, val in prms.items():
        config[key] = val

    config['ref_path'] = ref_path
    config['path'] = save_path

    if prms['right'] >= config['nx']:
        printm(f'update_config: right={prms["right"]} too far, changing it to {config["nx"]-1}')  
        prms['right'] = config['nx'] - 1

    if save_path is not None:
        with open(save_path + '/config_calcul.json', 'w') as myfile:
            json.dump(config, myfile)

    return config

def load_config(config_file_uri:str) -> dict:
    """ Converts the config.mat file into 
    a config.json file that is more humanely readable. 
    Takes the URI of the original config.mat file
    Writes the corresponding JSON dict into config.json
    and returns it.
    """
    conffile = loadmat(config_file_uri)
    old_confdict, new_confdict = {}, {}

    for key, val in conffile.items():
        if (key[:2] == '__') and (key[-2:] == '__'): # I KNOW IT IS UGLY
            pass
        elif isinstance(val, np.ndarray) and val.size == 1:
            old_confdict[key] = val.item()
        elif isinstance(val, np.ndarray):
            old_confdict[key] = list(np.squeeze(val))
        elif isinstance(val, bytes):
            old_confdict[key] = val.decode()


    # Conversion to USI & Putting more consistent variable names (and in English)
    new_confdict['fpulse'] = old_confdict['f0'] * 1e6
    new_confdict['fsample'] = old_confdict['rsf'] * 1e6
    new_confdict['frep'] = old_confdict['f_rec']
    new_confdict['fseq'] = 1/old_confdict['T_rep']

    new_confdict['c'] = old_confdict['C_p']
    new_confdict['time_us'] = [1e-3 * elem for elem in old_confdict['Time_us']]
    new_confdict['space_x'] = [1e-3 * elem for elem in old_confdict['Space_x']]
    new_confdict['space_z'] = [1e-3 * elem for elem in old_confdict['Space_x']]

    new_confdict['xini'] = old_confdict['Dist_acq_ini']
    new_confdict['xend'] = old_confdict['Dist_acq_fin']
    new_confdict['nz'] = old_confdict['nbvoie']
    new_confdict['nx'] = old_confdict['A']
    new_confdict['dz'] = old_confdict['pitch'] * 1e-3
    new_confdict['dx'] = old_confdict['C_p'] / 2 / old_confdict['rsf']

    new_confdict['nseqs'] = old_confdict['Nsequence']
    new_confdict['npulses'] = old_confdict['Nb_tir']
    new_confdict['npulses_total'] = old_confdict['Nsequence'] * old_confdict['Nb_tir']

    return new_confdict


def open_all(file_strs:list[str],  mode='r', skip=120) -> list:
    """ Opens all speckle files. Skips the first
    (useless) 120 bytes and returns the open files
    INPUTS
     * file_strs : list of file STRS
     * mode [optional] [ 'w' or 'r'] : 
     * skip [optional] : number of bytes to skip at beginning of file
     
    OUTPUT : list of FILE HANDLES
     """

    file_handles = []
    for file_str in file_strs:
        if mode == 'r':
            file = open(file_str, 'rb')
            file.read(skip)
        else: 
            file = open(file_str, 'wb')
            file.write(b'\x00'*skip)
        file_handles.append(file)
    return file_handles


def write_map(file_handles:list[BufferedIOBase], us_map:np.ndarray) -> None:
    """Writes a single (or multiple) US maps into files
    NOTE : WE WRITE AS _FLOATS_ instead of int16"""
    if us_map.ndim > 2: # should be (Nt, Nz, Nx)
        nt, nz, nx = np.shape(us_map)
        us_map = np.moveaxis(us_map, 0, 1) # Otherwise reshape fails 
        us_map = us_map.reshape([nz, nt * nx])
    
    for chno, file in enumerate(file_handles):
        file.write(us_map[chno,:].astype(np.float32).tobytes())


def close_all(file_handles:list[BufferedIOBase]) -> None:
    """ Closes all file handles.
    INPUT : list of FILE HANDLES """
    for file in file_handles:
        file.close()

def read_waveform(file_uri:str, config:dict, mode='raw') -> torch.Tensor:
    """Reads one waveform file and returns it as a 2D np.ndarray
    * You need to specify the config
    * You _can_ specify a reading mode (default : raw, reads np.int16 
    and converts the numbers, otherwise reads as float32 [what is typical
    for beamformed files]) """

    nt = config['npulses_total']
    skip = 120  # Number of bytes to skip

    if mode == 'raw':
        dtype = np.int16
    elif mode == 'bf': 
        dtype = np.float32
    else:
        raise ValueError('Please specify a reading mode : either `raw` or `bf` ')

    with open(file_uri, 'rb') as myfile:
        _ = myfile.read(skip)
        data = myfile.read()
    data = np.frombuffer(data, dtype=dtype)

    if mode == 'raw': # Raw speckles
        data = data - (data // 2048) * 4096 # Samesies, specific to Lecoeur
    
    data = np.reshape(data, [nt, -1]) # Somehow the first skip points seem to be rubbish
    
    return data


def read_map_batch(file_handles:list[BufferedIOBase], n_pts:int, ref=None, batch_size=1, mode='dat') -> np.ndarray:
    """ Reads a batch of US maps. NB: mode must be .dat or .dbf !
    
    NOTE: nskip represent the number of bytes skipped at the beginning of the file. We "simulate" the same
    number of metadata bytes as in the original files ==> 30 int16 ==> 120 bytes
    
    NOTE 2 : There is no check whatsoever that the number of bytes requested will be effectively
    loaded, in particular at the end of the file...""" 

    dat_list = []

    if mode == 'dat':
        for file in file_handles: # NOTE  : files need to be numerically sorted
            dat = file.read(batch_size * n_pts * 2 ) # np.int16 : two bytes per point
            dat = np.reshape(np.frombuffer(dat, dtype=np.int16), [-1, n_pts])
            dat = (dat - (dat // 2048) * 4096).astype(float)
            dat_list.append(dat)
        
    if mode == 'dbf':
        for file in file_handles: # NOTE  : files need to be numerically sorted
            dat = file.read(batch_size * n_pts * 4 ) # np.int16 : two bytes per point
            dat = np.reshape(np.frombuffer(dat, dtype=np.float32), [-1, n_pts])
            dat_list.append(dat)

    us = np.stack(dat_list)
    us = np.moveaxis(us, 1, 0)      # us is now (Nt, Nz, Nx)

    if ref is not None: 
        if ref.ndim == 2: 
            ref = ref[np.newaxis, :, :]
            us = us - ref

    return us

######################################
############ GENERAL PROCESSING ######
######################################

def make_ref(ref_path:str, 
             ref_config:dict,
             recompute=False):
    """ Makes a reference file if none can be found. The Ref is not 
    beam-formed (we will beam form the images from which the 
    reference has been  subtracted).
    - Specify the input ref_path
    - Specify the ref_config dictionary of the __REF__ folder
    - [optional] specify if you want to recompute the reference 
    - [optional] left trim 
    - [optional] right trim """
    
    ref_file = glob.glob(ref_path + '/ref.json' )
    dat_files = glob.glob(ref_path + '/*.dat*')
    filenum = lambda file : int(file.split('.')[0].split('_')[-1])
    dat_files = sorted(dat_files, key=filenum)

    if ref_file and not recompute:
        printm(f'make_ref: Loading {ref_file[0]}')
        ref = np.array(json.load(open(ref_file[0])))
        return ref
    
    n_channels = ref_config['nz'] # N_channels (128)
    n_pts = ref_config['nx']      # Length of the signal (~640)
    ref = np.zeros((n_channels, n_pts))

    for chno in tqdm(range(n_channels), desc='make_ref'):
        data = read_waveform(dat_files[chno], ref_config)
        ref[chno, :] = np.mean(data, axis=0)

    with open(ref_path + '/ref.json', 'w') as myfile:
        json.dump(ref.tolist(), myfile)

    return ref

def hilbert(data : np.ndarray, window: int, stride: int) -> torch.Tensor:
    """Computes the Hilbert intensity of a channel. 
    Can (optionally) do a sliding average over them to match
    the size of the cross-correlation maps

    NOTE 1 : I expect data to be of shape (N_waveforms x N_pts_per_waveform)
    NOTE 2 : The reference must be subtracted from the data
     before you do anything """

    data = torch.tensor(data)
    data_hat = torch.fft.fft(data)
    data_freq = torch.tile(torch.fft.fftfreq(data.shape[-1]), [data.shape[0],1])
    data_het = (data_hat * ((2 * (data_freq > 0)) + (data_freq == 0)))
    hil_int = torch.abs(torch.fft.ifft(data_het)).to(float)
    averager = AvgPool1d(kernel_size=stride, stride=stride)
    hil_avg = averager(hil_int)
    _, nhil = hil_avg.shape
    trim = (window - stride) // stride # To match displacement matrix 
    left_trim, right_trim = trim // 2, nhil - (trim - trim // 2) # Dispatching it between left and right
    hil_trim = hil_avg[:, left_trim:right_trim]
    
    return hil_trim


def displacement(data: np.ndarray, window:int, stride:int, max_disp=None) -> tuple[torch.Tensor]:
    """Computes the displacement (in pixel) between successive 1d signals
    for a given channel. Specify the data, then a window (correlation width) 
    and a stride (how much we shift the array indices between two correlations)

    NOTE 1 : I expect data to be of shape (N_t x N_pts_per_waveform)
    NOTE 2 : The reference must be subtracted from the data
     before you do anything 
    NOTE 3 : We have to apply the maximum displacement border here, because it can
    help pick up the right local maximum instead of a spurious one (so we get a OK 
    value instead of a NaN)
    NOTE 4 : you can now specify a `max_disp` to force narrowing down the search location for the 
    correlation maximum
    NOTE 5 : sometimes the correlation coefficient exceeds one (by a small margin), 
    this is due to the fact that the std of the shifted signals (which are truncated so that we
    sum them with a shift) is not exactly equal to 1 even if the entire signal is normed.
    """
    
    if max_disp >= window - 1: 
        printm(f'displacement: correlation max_disp {max_disp} exceeds maximum size {window} - 2')
        max_disp = window - 2

    old = torch.tensor(sliding_window_view(data, window, axis=1)[:-1, ::stride,:].copy())
    new = torch.tensor(sliding_window_view(data, window, axis=1)[1:, ::stride, ::-1].copy())

    fft_convolve = FFTConvolve(mode='valid')

    cvs = []

    for corrshift in range(max_disp,-max_disp-1,-1):

        # Selecting the right parts of the signal for correlation
        if corrshift >= 0:
            old_part = old[:,:,corrshift:]
            new_part = new[:,:,corrshift:]
            nsum = window - corrshift
        elif corrshift < 0: 
            old_part = old[:,:,:corrshift]
            new_part = new[:,:,:corrshift]
            nsum = window + corrshift
        
        # # # Normalising the signals
        new_part = (new_part - torch.nanmean(new_part, dim=-1, keepdim=True)) \
            / torch.std(new_part, dim=-1, unbiased=False, keepdim=True)
        old_part = (old_part - torch.nanmean(old_part, dim=-1, keepdim=True)) \
            / torch.std(old_part, dim=-1, unbiased=False, keepdim=True)

        # Convolving (correlating) on last dimension
        cv = fft_convolve(old_part, new_part) / nsum
        cvs.append(cv)

    # Retrieving the correct maximum index (+ subpixel precision)
    # and the corresponding correlation score (for validation)
    cvs = torch.cat(cvs, dim=-1)
    score_max, ind_max = torch.max(cvs,dim=2)
    ind_max = ind_max.unsqueeze(-1)
    score_max = score_max.unsqueeze(-1)

    ind_left_clip = torch.clamp(ind_max-1, 0, 2*max_disp)
    ind_right_clip = torch.clip(ind_max+1, 0, 2*max_disp)
    score_left = torch.gather(cvs, 2, ind_left_clip)
    score_right = torch.gather(cvs, 2, ind_right_clip)
    R_factor = (score_max - score_right) / (score_max - score_left)
    delta = ((R_factor - 1) / (1 + R_factor)) / 2 
    delta = ind_max.squeeze() - max_disp - delta.squeeze()
    score_max = score_max.squeeze()

    return delta, score_max


def process(bf_files:list[str], config:dict, recompute=True, sep='\\'): 
    """Processes a batch of 128 beam-formed files. You know the drill now,
    you pass the list of file uris, the `config` dict, the window and the stride
    and you will get your precious data.
    
    NOTE: you can now specify a `max_disp` to force narrowing down the search location for the 
    correlation maximum """

    # What to do if data already exists
    match = glob.glob(''.join(bf_files[0].split(sep)[:-1]) + '/processed.npz')
    if match and not recompute:
        printm(f'process: Found {match[0]}, loading it. Set `recompute=True` to reprocess')
        data = np.load(match[0])
        return data['hil'], data['disp'], data['score']

    window = config['window']
    stride = config['stride']
    max_disp = config['max_disp']

    hil_all = []
    disp_all = []
    score_all = []


    for file in tqdm(bf_files, desc= 'process'):
        us = read_waveform(file, config, mode='bf') # Ref already subtracted
        hil_all.append(hilbert(us, window, stride))
        disp, score = displacement(us, window, stride, max_disp)
        disp_all.append(disp)
        score_all.append(score)


    hil_all = torch.stack(hil_all).cpu().numpy()
    disp_all = torch.stack(disp_all).cpu().numpy()
    score_all = torch.stack(score_all).cpu().numpy()


    return hil_all, disp_all, score_all

def calibrate_one(calib:dict, config:dict, disp=None, folder=None):
    """Computes the calibration velocity profiles
    For reasons unknown, there is a factor two (not just due to return trip when particles move,
    that one I took care of) in the computation of the velocity that I have to enforce to match
    Sébastien's code but I don't really understand why. Worst case scenario we have sin(theta) wrong.

    NOTE : theta is in degrees.
    """

    theta = calib['theta']
    t_stator = calib['t_stator']
    c0 = calib['c0']
    r_int = calib['r_int']
    r_ext = calib['r_ext']

    # Loading data
    if disp is None: 
        data = np.load(folder + '/processed.npz')
        disp = data['disp']

    # 
    theta_rad = np.pi * theta / 180

    # Working out the times
    nz, nt, nr = disp.shape
    t_us = np.array(config['time_us']) # us is for ultrasound, not microsecond. Time_us is in ms
    dt_us = np.mean(np.diff(t_us)) # As good as a diff(t_us[:2])
    t_new_ini = t_us[0] + dt_us * ( (1 + config['window']) / 2)
    dt_new =  dt_us * config['stride']
    t_new = t_new_ini + dt_new * np.arange(nr)

    # Working out the space
    r_raw = (t_new - t_stator) * c0 / 2 # Corresponds to (y-y0) in the Gallot 2013 paper
    r_true = (r_ext ** 2 + r_raw ** 2 - 2 * r_ext * r_raw * np.cos(theta_rad))** 0.5 - r_int
    r_true_2d = r_true[np.newaxis, np.newaxis, :]
    r_plot = (r_ext-r_int) - r_true

    # Working out the velocity
    disp_true = (r_int + r_true_2d)/(r_ext * np.sin(theta_rad)) * disp / 2  # geometry x displacement_pixels x 1/2 (if a particle moves it affects both forward and return trip)
    velocity = c0 / config['fsample'] * config['frep'] * disp_true  # Converting displacement pixels into actual velocity (in mm/s) : Length scale / time scale
    v_mf = np.nanmean(velocity, axis=1)
    v_profile = np.nanmean(v_mf, axis=0)
    v_std = np.nanstd(np.reshape(velocity, [nt * nz, nr]), axis=0)

    return r_true, velocity, v_mf, v_profile, v_std


def bf_indices_coeffs(config:dict) -> tuple[torch.tensor]:
    """Computes (once and for all) the delays _dj_ associated to beamforming
    at a position (_,j0) [the delays are the same regardless of i0, the channel number].
    The shifts are then converted to actual indices _i and j_ used for the beamforming sum for a given set of initial
    indices _i0, j0_. We finally build a Nz x Nx x Nbf array of indices to sum in the original speckle file to produce the beamformed 
    signal when we sum over the last dimension. We _actually_ build two of these tables and two tables of 
    weight factors to accommodate for non-integer delays _dj_

    ARGS
    ----
    config : dict with the usual stuff

    RETURNS
    ----
    * flat_idx_left  : left summation indices in the flattened US map [ see np.ravel() ]  to produce the beamformed signal
    * flat_idx_right : right summation indices in the flattened US map to produce the beamformed signam
    * coeff_left : weight coefficient for the summation (left part)
    * coeff_right : weight coefficient for the summation (right part)
    * valid : the valid indices --> useful for the bf signal normalisation !

    NOTE: this means that somewhere later in the code, you do `bf3d = coeff_left * us_flat[flat_idx_left] + coeff_right + us_flat[flat_idx_right]` 
    """

    c = config['c']
    dz = config['dz']
    nx = config['nx']
    nz = config['nz']
    nbf = config['nchan_bf']
    fsample = config['fsample']
    
    x = np.atleast_2d(config['time_us']) * config['c'] / 2 
    di = np.atleast_2d(np.arange(-nbf,nbf+1)).T
    dj = fsample * x/c * ((1 + (dz * di)**2/x**2) ** 0.5 - 1)  # For each j0 (initial), computes the dj((i-i0, j0))
    j = np.moveaxis(np.tile(np.arange(nx) + dj, [nz, 1, 1]), 1,-1) # Build a 3d array with j + dj(i-i0, j0) 
    jint = np.floor(j) # Integer part (used for indices)
    jfrac = j - jint # Fractional part used by the weights
    jint = jint.astype(np.int_)
    coeff_left = (1-jfrac)
    coeff_right = jfrac
    
    i0 = np.moveaxis(np.tile(np.arange(nz), [nx,2*nbf+1,1]), [0,1,2], [1,2,0])  # Build 3d table with i0 indices
    i = i0 + np.tile(np.arange(-nbf,nbf+1), [nz,nx,1]) # Build 3d table with i = i0 + (i-i0) indices (they increase over the 3rd dimension)
    
    # Compute the "flat" indices
    flat_idx_left = i*640 + jint
    flat_idx_right = i*640 + jint + 1

    # Deal with out of bounds
    j_valid = jint + 1 < config['right']
    i_valid = (i >= 0) & (i < 128) # Validate the i's
    valid = i_valid & j_valid
    n_valid = valid.sum(axis=-1)
    n_valid[n_valid == 0] = 1 # To avoid 0/0 issues later on
    flat_idx_left[~valid] = 0
    flat_idx_right[~valid] = 0
    coeff_left[~valid] = 0
    coeff_right[~valid] = 0
    
    return torch.tensor(flat_idx_left), \
        torch.tensor(flat_idx_right), \
        torch.tensor(coeff_left), \
        torch.tensor(coeff_right), \
        torch.tensor(n_valid)


def beamform(file_strs:list[str], config:dict, ref=None, recompute=False, batch_size=100) -> list[str]:
    """Beamforms all ultrasound files. You can trim the files in x (time_us axis)
    with the `left` and `right` indices.
    
    NOTE 1 : Beamformed files are in __float32__ format their name is Speckle_xxx.dbf
    NOTE 2 : Writing in BF files is buffered
    NOTE 3 : We call a numba-accelerated function `beamform_one`
    NOTE 4 : By default we are not recomputing the files !
    """
    
    # Check if we need to work
    bf_file_strs = [f_str.replace('.dat', '.dbf') for f_str in file_strs]
    match = glob.glob(file_strs[0].replace('.dat', '.dbf'))
    if (not recompute) and match:
        printm('beamform : Found .dbf files, not recomputing')
        return bf_file_strs

    # Extracting relevant info from the config dict
    n_pts = config['nx']
    n_batches = np.ceil(config['npulses_total']/batch_size).astype(np.int_)
    
    # Beamforming loop
    orig_files = open_all(file_strs)
    bf_files = open_all(bf_file_strs, mode='w')
    idx_left, idx_right, weight_left, weight_right, n_valid = bf_indices_coeffs(config)
    
    for _ in tqdm(range(n_batches), desc='beamform'):

        bf_batch = []
        us_batch = read_map_batch(orig_files, n_pts=n_pts, ref=ref, batch_size=batch_size)
        us_batch_flat = torch.tensor(np.reshape(us_batch, [batch_size, -1])) # 2d us_batch 

        for us_flat in us_batch_flat:
            bf = (weight_left * us_flat[idx_left] 
                  + weight_right * us_flat[idx_right]) \
                  .sum(dim=-1) / n_valid
            bf_batch.append(bf)

        # When batch is processed, back to numpy
        bf_3d = torch.cat(bf_batch, dim=-1).cpu().numpy()
        bf_batch = []
        write_map(bf_files, bf_3d)

    close_all(bf_files)
    close_all(orig_files)
    return bf_file_strs


