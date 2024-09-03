import torch 
import json
import glob
import numpy as np

from torch.nn import AvgPool1d
from numpy.lib.stride_tricks import sliding_window_view
from torchaudio.transforms import FFTConvolve, Convolve
from scipy.io import loadmat
from numba import njit, cuda
from io import BufferedIOBase
from tqdm import tqdm

from colorama import Fore, Style

"""
Quick disclaimer : 

n_t is the number of pulses sent (and the number of received echoes) : usually Nb_tir x Nsequence
n_r or n_j0 or n_pts is the number of data points for each recorded pulse : formerly known as 'A'
n_z or n_i0 is the number of channels : it should be 128

"""

######################################
############ FILE MANAGEMENT #########
######################################

def find_files(path:str, ext='dat'):
    """ Finds Speckle files using glob and sorts them numerically"""
    files = np.array(glob.glob(path + '\*'+ ext))
    filenum = lambda file : int(file.split('.')[0].split('_')[-1])
    files = sorted(files, key=filenum)
    return files

def print_details(path:str, config:dict):
    """ Prints details about the experiment based on its path"""
    n_wavelengths = config['window'] / config['fsample'] * config['fpulse'] # There is a 1/2 factor in Vincent's/Thomas's code and I have no idea why...
    overlap = 1 - config['stride'] / config['window']
    rotor = config['right']
    ref_path = config['ref_path']
    
    print(Fore.GREEN, end='')
    print(f'\nProcessing {path} ...')
    print(Style.RESET_ALL, end='')
    print(f'Ref. used at : {ref_path}')
    print(f'Window size = {n_wavelengths:.2f} λ | Overlap {overlap:.2f} | Rotor position {rotor} px')
    

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
        print(f'update_config: right={prms["right"]} too far, changing it to {config["nx"]-1}')  
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


######################################
############ GENERAL PROCESSING #########
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
        print(f'make_ref: Loading {ref_file[0]}')
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
    data_het = data_hat * ((2 * (data_freq > 0)) + (data_freq == 0))
    hil_int = torch.abs(torch.fft.ifft(data_het)).to(float)
    averager = AvgPool1d(kernel_size=window, stride=stride)

    return averager(hil_int)


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
        print(f'displacement: correlation max_disp {max_disp} exceeds maximum size {window} - 2')
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
        print(f'process: Found {match[0]}, loading it. Set `recompute=True` to reprocess')
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


    hil_all = np.stack(hil_all)
    disp_all = np.stack(disp_all)
    score_all = np.stack(score_all)

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

###########################################
########## Beamforming : general ##########
###########################################

def beamform(file_strs:list[str], config:dict, ref=None, recompute=False, batch_size=100) -> list[str]:
    """ Just a wrapper between the two beamform (`beamform_gpu` and `beamform_cpu`) programmes """
    if config['do_cuda']: 
        bf_file_strs = beamform_gpu(file_strs, config, ref, recompute, blocks_per_grid=batch_size)
    else:
        bf_file_strs = beamform_cpu(file_strs, config, ref, recompute, batch_size=batch_size)
    return bf_file_strs


def compute_delays(t_us:np.ndarray, c:float, pitch:float, fech:float, nbf:int) -> np.ndarray:
    """Computes (once and for all) the delays associated to beamforming
    at a position (j) [so for each position of t_us] considering a sound
    velocity c, a distance between transducer elements pitch, a sampling frequency 
    fech and a number of beamforming channels nbf"""

    # XXX : SOMETHING FISHY SINCE x0 = c t0 / 2

    x = np.atleast_2d(t_us) * c / 2 
    di = np.atleast_2d(np.arange(-nbf,nbf+1)).T
    dj = fech * x/c * ((1 + (pitch * di)**2/x**2) ** 0.5 - 1)  # For each j0 (initial), computes the dj((i-i0, j0))
    return dj


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

###########################################
########## Beamforming : CPU ##############
###########################################

def beamform_cpu(file_strs:list[str], config:dict, ref=None, recompute=False, batch_size=100) -> list[str]:
    """Beamforms all ultrasound files on the CPU. You can trim the files in x (time_us axis)
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
        print('beamform : Found .dbf files, not recomputing')
        return bf_file_strs

    # Extracting relevant info from the config dict
    n_pts = config['nx']
    n_batches = np.ceil(config['npulses_total']/batch_size).astype(np.int_)
    c = config['c'] 
    t_us = np.array(config['time_us'])
    left = config['left']
    right = config['right']
    if left is None: left = 0
    if right is None: right = n_pts
    if right > n_pts : right = n_pts
    
    # Beamforming loop
    orig_files = open_all(file_strs)
    bf_files = open_all(bf_file_strs, mode='w')
    dj = compute_delays(t_us, c, config['dz'], config['fsample'], config['nchan_bf'])
    
    for _ in tqdm(range(n_batches), 
                  desc='beamform' + Fore.YELLOW + ' using CPU' + Style.RESET_ALL):

        bf_batch = []
        us_batch = read_map_batch(orig_files, n_pts=n_pts, ref=ref, batch_size=batch_size)

        for us in us_batch:
            bf = bf_cpu_single(us, dj, right=right)
            bf_batch.append(bf)

        # When batch is processed
        bf_3d = np.stack(bf_batch)
        bf_batch = []
        write_map(bf_files, bf_3d)
    

    close_all(bf_files)
    close_all(orig_files)
    return bf_file_strs

@njit
def bf_cpu_single(us:np.ndarray, dj:np.ndarray, right:int):        
    """ Beamforms one ultrasound map
    ARGS
    ----
    * us : a (2d) ultrasound map of the form us = us[i0, j0]
    * dj : the delay matrix in the form dj = dj[i-i0, j0]
    
    RETURNS
    ----
    * bf (same size as US) : the beamformed map
        
    NOTE 1 : You can produce the delay matrix using `compute_delays`
    """


    ni0, nj0 = us.shape
    ni, _ = dj.shape            # the indices 0 ... ni [i call them rows_retards] correspond to a shift (i - i0) of - nbf to nbf
    nbf = (ni - 1) // 2         # so 1 + 2*nbf = ni
    bf = np.zeros_like(us)
    us_flat = us.ravel()
    
    # Some things can be computed out of the loops. Let's do it
    di = np.arange(-nbf, nbf+1)[:, np.newaxis]
    j = np.arange(nj0) + dj
    j_int = np.floor(j).astype(np.int_)
    j_frac = j - j_int

    valid_j = (j_int + 1 < right) | ((dj == 0) & (j_int < right)) # Case 1 (non int j) we spill over next point / otherwise we don't
    valid_i = np.zeros_like(valid_j).astype(np.bool_)

    for i0 in range(ni0):

        i = i0 + np.zeros_like(dj).astype(np.int_) + di
        valid_i = (i >= 0) & (i < ni0) 
        valid = valid_i & valid_j
        idx_flat = (j_int + i * nj0)

        for j0 in range(right): # New implementation cannot deal with beam forming on the rightmost point, so I am skipping it

            is_valid = valid[:,j0]
            n_valid = is_valid.astype(np.int_).sum()
            left_idx = idx_flat[:,j0][is_valid]
            right_idx = left_idx + 1
            right_coef = j_frac[:,j0][is_valid]
            left_coef = 1 - right_coef
            left_part = us_flat[left_idx] * left_coef
            right_part = us_flat[right_idx] * right_coef
            bf[i0,j0] = (left_part.sum() + right_part.sum())/n_valid

    return bf


def read_map_single(file_handles:list[BufferedIOBase], n_pts:int, ref=None):

    """ Reads a single map of **RAW US** """ 

    us = np.zeros((len(file_handles), n_pts), dtype=np.int16)
    for chno, file in enumerate(file_handles): # NOTE  : files need to be numerically sorted
        dat = file.read(n_pts * 2 ) # np.int16 : two bytes per point
        us[chno,:] = np.frombuffer(dat, dtype=np.int16)
    us = us - (us // 2048) * 4096
    us = us.astype(float)

    if ref is not None: return us - ref
    return us


###########################################
########## Beamforming : _G_PU ############
###########################################

@cuda.jit
def bf_cuda_batch(us:np.ndarray, 
              bf:np.ndarray, 
              dj:np.ndarray):
    
    # us is [nt [blockIdx], ni0 [threadIdx], nj0 [we for loop over them]]
    # retards is [ni [we sum over them], nj0 [we for loop over them]]
    # i0 : no voie du speckle beamformé
    # j0 : no échantillon du speckle beamformé
    # i : no voie du speckle original
    # j : no échantillon du speckle original

    i0 = cuda.threadIdx.x
    ni0 = cuda.blockDim.x
    t = cuda.blockIdx.x
    max_size = us.size # bf has the same size
    bf_sum_on, nj0 = dj.shape

    for j0 in range(nj0-1):

        idx_bf = (t * ni0 + i0) * nj0 + j0
        rows_retards = range(0, bf_sum_on)
        valid_idx_bf = idx_bf < max_size
        n_sum = 0
       
        for row in rows_retards:

            i = i0 + row - (bf_sum_on - 1)//2
            j = j0 + dj[row, j0]
            j_int = np.floor(j).astype(np.int_)
            frac_left = (1 - j) % 1
            frac_right = (j) % 1

            idx_us_left  = (t * ni0 + i) * nj0 + j_int
            idx_us_right = (t * ni0 + i) * nj0 + j_int + 1

            valid_i = (i >= 0) & (i < ni0)
            valid_j = (j_int + 1 < nj0)
            valid_us_idx = (idx_us_right < max_size)

            if valid_j & valid_i & valid_us_idx & valid_idx_bf:
                n_sum += 1
                bf[idx_bf] += frac_left * us[idx_us_left] + frac_right * us[idx_us_right]            

        if n_sum == 0:
            print(i0, j0)
        bf[idx_bf] = bf[idx_bf] / n_sum

    # Fix for what happens at coordinates (t, i0, nj0-1)
    bf[(t * ni0 + i0) * nj0 + (nj0 - 1)] = us[(t * ni0 + i0) * nj0 + (nj0 - 1)]


def beamform_gpu(file_strs:list[str], config:dict, ref=None, recompute=False, blocks_per_grid=1) -> list[str]:

    """Beamforms all ultrasound files on the _G_PU. You can trim the files in x (time_us axis)
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
        print('beamform_gpu: Found .dbf files, not recomputing')
        return bf_file_strs

    # Extracting relevant info from the config dict
    t_us = np.array(config['Time_us'])*1e-3
    n_pts = len(t_us)
    n_waveforms = config['Nsequence'] * config['Nb_tir']
    n_chan = config['nbvoie']
    n_chan_bf = config['n_chan_bf']
    c = config['C_p'] 
    pitch = config['pitch'] * 1e-3 # convert to USI
    fech = config['rsf'] * 1e6 # convert to USI
    
    # Trimming signal
    left = config['left']
    right = config['right']
    
    # CUDA parameters
    threadsperblock = n_chan
    n_iters = np.ceil(n_waveforms/blocks_per_grid).astype(np.int_)

    # Beamforming loop
    orig_files = open_all(file_strs)
    bf_files = open_all(bf_file_strs, mode='w')
    dj = compute_delays(t_us, c, pitch, fech, n_chan_bf)

    for _ in tqdm(range(n_iters), 
                     desc='beamform' + Fore.GREEN + ' using GPU' + Style.RESET_ALL):

        us = read_map_batch(orig_files, n_pts=n_pts, batch_size=blocks_per_grid, ref=ref)
        us_flat = us.ravel()
        bf_flat = np.zeros_like(us_flat)

        bf_cuda_batch[blocks_per_grid, threadsperblock](us_flat, bf_flat, dj) # NOTE: add 'right' in kernel
        write_map(bf_files, bf_flat.reshape(us.shape))

    close_all(bf_files)
    close_all(orig_files)
    return bf_file_strs


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
