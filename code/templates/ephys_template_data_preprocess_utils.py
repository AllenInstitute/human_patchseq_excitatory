import os
import json
import pandas as pd
import numpy as np
import pg8000
from ipfx.dataset.create import create_ephys_data_set, get_nwb_version
from ipfx.spike_detector import detect_putative_spikes

def limspath_from_cellname(cellname):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                          password="limsro", port=5432)
    cur = conn.cursor()
    cur.execute(
    """SELECT err.storage_directory AS path 
    FROM specimens cell 
    JOIN ephys_roi_results err ON err.id = cell.ephys_roi_result_id 
    WHERE cell.name LIKE '{}'
    """.format(cellname))

    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result != None:
        return result
    
def cellname_from_cellid(cell_id):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                          password="limsro", port=5432)
    cur = conn.cursor()
    cur.execute(
    """SELECT cell.name AS cellname 
    FROM specimens cell 
    WHERE cell.id = {}
    """.format(cell_id))

    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result != None:
        return result[0]

def sweep_qc_state(cellname, sweep_num):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                        password="limsro", port=5432)
    cur = conn.cursor()
    cur.execute(
    """SELECT sw.workflow_state AS wfl_state
    FROM specimens cell
    JOIN ephys_sweeps sw ON sw.specimen_id = cell.id
    WHERE cell.name LIKE '{}' 
    AND sw.sweep_number::text LIKE '{}'
    """.format(cellname, sweep_num))

    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result != None:
        return result

def find_nwb_v2(path):
    nwb2_file = None
    for root, dirs, files in os.walk(path):
        for fil in files:
            if fil.endswith(".nwb"):
                test_nwb_name = os.path.join(path, fil)
                try:
                    test_nwb_version = get_nwb_version(test_nwb_name)
                    nwb_version = test_nwb_version['major']
                    if nwb_version == 2:
                        nwb2_file = path + fil
                except OSError as e:
                    pass
        return nwb2_file 

def make_dataset(nwb_path):
    try:
        dataset = create_ephys_data_set(nwb_file=nwb_path)
    except (ValueError, OSError, TypeError):
        dataset = None
        print("can't make dataset")

    return dataset

def stimset_sweep_nums(dataset, stimset_name):
    try:
        sweep_table = dataset.sweep_table
    except (TypeError, AttributeError):
        sweep_table = pd.DataFrame()
        print("can't make sweep table")
    
    if len(sweep_table) > 0:
        sweep_nums = sweep_table[sweep_table.stimulus_code.str.startswith(stimset_name)]
        sweep_nums_lst = list(sweep_nums.sweep_number)
    else:
        sweep_nums_lst = []
    return sweep_nums_lst

def passing_sweepnums(dataset, sweep_nums_lst, cellname):
    passed_lst = ['auto_passed', 'manual_passed']
    passed_sweep_nums = []
    for sweep_num in sweep_nums_lst:
        qc_state = sweep_qc_state(cellname, sweep_num)
        if qc_state is not None:
            if qc_state[0] in passed_lst:
                sweep = dataset.sweep(sweep_num)
                samp_rate = sweep.sampling_rate
                # checking if recording is longer than 2 seconds b/c of sweep qc issue in LIMS for some cells
                if sweep.epochs['recording'][1] >= int(2.0 * samp_rate):
                    passed_sweep_nums.append(sweep_num)
    return passed_sweep_nums

def get_square_pulse_idx(v):
    """
    Get up and down indices of the square pulse(s).
    Skipping the very first pulse (test pulse)

    Parameters
    ----------
    v: float
        pulse trace

    Returns
    -------
    up_idx, down_idx: list, list
        up, down indices
    """
    dv = np.diff(v)

    up_idx = np.flatnonzero(dv > 0)[1:] # skip the very first pulse (test pulse)
    down_idx = np.flatnonzero(dv < 0)[1:]

    assert len(up_idx) == len(down_idx), "Truncated square pulse"

    return up_idx, down_idx

def nonspiking_sweeps(dataset, sweep_nums_lst):
    nonspiking_sweeps = []
    for sweep_num in sweep_nums_lst:
        print(sweep_num)
        sweep = dataset.sweep(sweep_num)
        up_idx, down_idx = get_square_pulse_idx(sweep.i)
        stim_end = int(up_idx)

        if len(sweep.v) > 0:
            put_spikes = detect_putative_spikes(sweep.v[:stim_end], sweep.t[:stim_end])
            if put_spikes.size == 0:
                nonspiking_sweeps.append(sweep_num)

    return nonspiking_sweeps

def rheobase_sweep_num(cellname):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                          password="limsro", port=5432)
    cur = conn.cursor()
    cur.execute(
    """SELECT epf.threshold_i_long_square AS rheo_sweep 
    FROM specimens cell 
    JOIN ephys_features epf ON epf.specimen_id = cell.id 
    WHERE cell.name LIKE '{}'
    """.format(cellname))

    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result != None:
        return result

def feature_extraction_output(path):
    for root, dirs, files in os.walk(path):
        for fil in files:
            if fil.startswith('EPHYS_FEATURE_EXTRACTION_V3_QUEUE_') and fil.endswith('output.json'):
                with open(path + fil) as f:
                    data = json.load(f)
                    rheobase_sweep_num = data['cell_features']['long_squares']['rheobase_sweep']['sweep_number']
                    
    if rheobase_sweep_num:
        return rheobase_sweep_num
    
def ephys_features_from_LIMS(cellname):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                          password="limsro", port=5432)

    cur = conn.cursor()

    cur.execute(
    """SELECT eff.tau AS time_const, eff.ri AS input_res, eff.sag AS sag
    FROM specimens cell
    JOIN ephys_features eff ON eff.specimen_id = cell.id
    WHERE cell.name LIKE '{}'    
    """.format(cellname))

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result:
        return result
    

def get_features_from_LIMS(cellnames):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", 
                        password="limsro", port=5432)
    cur = conn.cursor()

    query = """
    SELECT epf.*, cell.name AS cellname
    FROM specimens cell
    LEFT JOIN ephys_features epf ON epf.specimen_id = cell.id
    WHERE cell.name IN {}
    """.format(cellnames)

    result = cur.fetchone()
    cur.close()
    conn.close()

    if result != None:
        return result
    
    
def process_ephys_rheobase_data(cells):  
    rheobase_sweep_nums = {}
    cells_with_issues = []

    for cell in cells:
        result = limspath_from_cellname(cell)
        if result is None:
            cells_with_issues.append(cell)
            continue  # Skip this cell and move to the next one

        path = '\\' + result[0].replace('/','\\')

        if path:
            for root, dirs, files in os.walk(path, followlinks=False):
                for fil in files:
                    fx_str = 'EPHYS_FEATURE_EXTRACTION_V3_QUEUE_'
                    if (fil.startswith(fx_str) or fil[13:].startswith(fx_str)) and fil.endswith('output.json'):
                        try:
                            with open(os.path.join(path, fil)) as f:
                                data = json.load(f)
                            if not data['cell_state']['failed_fx']:
                                try:
                                    rheobase_sweep_num = data['cell_features']['long_squares']['rheobase_sweep']['sweep_number']
                                    rheobase_sweep_nums[cell] = rheobase_sweep_num
                                except (TypeError, KeyError):
                                    cells_with_issues.append(cell)
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"Error reading {fil}: {e}")
                            cells_with_issues.append(cell)

    known_cells = list(rheobase_sweep_nums.keys()) + cells_with_issues


    # Initialize storage
    rheobase_sweep_data = {}
    rheobase_sweep_list = []
    rheobase_issues = []

    # Loop through all cells with identified rheobase sweep numbers
    for cell_name, sweep_num in rheobase_sweep_nums.items():
        spike_dict = {}
        
        # Get the initial path from cell name
        result = limspath_from_cellname(cell_name)
        if result is None:
            print(f"Path not found for {cell_name}")
            rheobase_issues.append(cell_name)
            continue

        path = '\\' + result[0].replace('/','\\')

        try:
            # Load NWB file and dataset
            nwb = find_nwb_v2(path)
            dataset = make_dataset(nwb)

            if dataset is None:
                print(f"make_dataset returned None for {cell_name}")
                rheobase_issues.append(cell_name)
                continue

            # Get the sweep
            rheo_sweep = dataset.sweep(sweep_num)
            sampling_rate = rheo_sweep.sampling_rate

            # Get square pulse region
            up_idx, down_idx = get_square_pulse_idx(rheo_sweep.i)

            # Detect spikes
            put_spikes = detect_putative_spikes(rheo_sweep.v, rheo_sweep.t)

            if len(put_spikes) > 0:
                spike_idx = put_spikes[0]
                start = spike_idx
                end = min(spike_idx + int(0.003 / (1 / sampling_rate)), len(rheo_sweep.v))

                spike_data = np.array(rheo_sweep.v[start:end])
                time_data = rheo_sweep.t[start:end]

                # Save spike info
                spike_dict['spike_data'] = spike_data
                spike_dict['time_data'] = time_data
                spike_dict['sampling_rate'] = sampling_rate

                rheobase_sweep_data[cell_name] = spike_dict
                rheobase_sweep_list.append(spike_data)

        except (IndexError, RuntimeError, FileNotFoundError, OSError, KeyError, AttributeError) as e:
            print(f"Error processing {cell_name}: {e}")
            rheobase_issues.append(cell_name)
            continue

    return rheobase_sweep_data, rheobase_issues

def process_ephys_subthresh_data(data_df):
    subthresh_stimuli = ["C1LSCOARSEMICRO", "X1PS_SubThresh", "subthreshold", "PS_SubThresh"]
    cells_dict = {}
    count = len(data_df)

    for idx, row in data_df.iterrows():
        # print(row['cell_specimen_name'])

        path = limspath_from_cellname(row.cell_specimen_name)
        if path:
            path = '\\' + path[0].replace('/','\\')

            nwb = find_nwb_v2(path)
            dataset = make_dataset(nwb)

            if dataset is None:
                #print(f"Warning: dataset is None for cell {row.cell_specimen_name}, skipping")
                count -= 1
                continue

            stim_sweepnums = []
            for stim in subthresh_stimuli:
                try:
                    test_stim_sweepnums = stimset_sweep_nums(dataset, stim)
                    if len(test_stim_sweepnums) > 0:
                        stim_sweepnums = test_stim_sweepnums
                        #print(f"Found stimset: {stim}")
                        break  # stop searching once we find a valid stimset
                except ValueError as e:
                    #print(f"Error for stimset '{stim}': {e}")
                    continue

            if len(stim_sweepnums) > 0:
                sweeps_dict = {}
                passing_sweeps = passing_sweepnums(dataset, stim_sweepnums, row.cell_specimen_name)

                if len(passing_sweeps) > 0:
                    passing_nospike_sweeps = nonspiking_sweeps(dataset, passing_sweeps)
                    #print(len(passing_nospike_sweeps))

                    if len(passing_nospike_sweeps) > 0:
                        for sweepnum in passing_nospike_sweeps:
                            try:
                                sweep_dict = {}
                                sweep = dataset.sweep(sweepnum)
                                up_idx, down_idx = get_square_pulse_idx(sweep.i)

                                if up_idx > down_idx:
                                    pad = 0.1 * sweep.sampling_rate
                                    if min(sweep.i) == -110:
                                        start = int(down_idx - pad)
                                        end = int(up_idx + pad)
                                        sweep_dict['time'] = sweep.t[start:end]
                                        sweep_dict['stim'] = sweep.i[start:end]
                                        sweep_dict['response'] = sweep.v[start:end]
                                        sweep_dict['samp_rate'] = sweep.sampling_rate
                                        sweeps_dict[sweepnum] = sweep_dict
                            except (AssertionError, IndexError, ValueError) as e:
                                #print(f"Error processing sweep {sweepnum}: {e}")
                                continue

                        if len(sweeps_dict) > 0:
                            cells_dict[row.cell_specimen_name] = sweeps_dict

        count -= 1

    return cells_dict