import mat73
import numpy as np
import pandas as pd

def sample_data(data_dict, sample_rate = 128):
    sample_factor = data_dict['spindle_det']["Fs"] // sample_rate
    num_points = len(data_dict['dataConcat'])
    data_dict['dataConcat'] = [data_dict['dataConcat'][i] for i in range(num_points) if i % sample_factor == 0]
    data_dict['spindle_det']["Fs"] = data_dict['spindle_det']["Fs"] // sample_factor
    data_dict['spindle_det']["endSample"] = [x // sample_factor for x in data_dict['spindle_det']["endSample"]]
    data_dict['spindle_det']["startSample"] = [x // sample_factor for x in data_dict['spindle_det']["startSample"]]

    num_points = len(data_dict['dataConcat'])
    sample_rate = int(data_dict['spindle_det']["Fs"])

    return data_dict, sample_rate, sample_factor

def train_test_split(x, t, ratio = 0.95):
    train_size = int(x.shape[0] * ratio)
    train_x = x[:train_size]
    train_t = t[:train_size]
    test_x = x[train_size:]
    test_t = t[train_size:]

    return train_x, train_t, test_x, test_t

def shuffle_data(x, t):
    idx = np.arange(t.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]

    return x, t

def fix_data_distribution(x, t, distribution_coeff = 2):
    true_x = x[t == 1]
    true_t = t[t == 1]
    false_x = x[t == 0]
    false_t = t[t == 0]
    false_x = false_x[:true_x.shape[0] * distribution_coeff]
    false_t = false_t[:true_t.shape[0] * distribution_coeff]

    x = np.concatenate((true_x, false_x), axis=0)
    t = np.concatenate((true_t, false_t), axis=0)

    return x, t

def normalize(data):
    data -= data.mean()
    data /= data.std()
    
    return data

def load_data(file_name, sample_rate = 128):
    data_dict = mat73.loadmat(file_name)
    data_dict, sample_rate, sample_factor = sample_data(data_dict, sample_rate=sample_rate)
    num_points = len(data_dict['dataConcat'])
    target_frame_count = sample_rate * 5

    df = pd.DataFrame([
            [
                data_dict['spindle_det']["startSample"][i], 
                data_dict['spindle_det']["endSample"][i]
            ] 
            for i in range(len(data_dict['spindle_det']["startSample"]))
        ],
        columns=['start', 'end']
    )

    raw_labels = np.zeros((num_points))

    for i in range(df.shape[0]):
        raw_labels[int(df.iloc[i]['start']):int(df.iloc[i]['end'])] = 1

    raw_x = []
    raw_t = []

    for i in range(0, num_points - ((5 * 60 * sample_rate) + target_frame_count), sample_rate):
        frame_start = i
        frame_end = i + (5 * 60 * sample_rate)
        label_start =  i + (5 * 60 * sample_rate)
        label_end =  i + (5 * 60 * sample_rate) + target_frame_count

        raw_x.append(data_dict['dataConcat'][frame_start:frame_end])
        
        if raw_labels[label_start:label_end].max() == 1:
            raw_t.append(1)
        else:
            raw_t.append(0)

    x = np.array(raw_x)
    t = np.array(raw_t)
    x = normalize(x)

    x, t = fix_data_distribution(x, t)
    x, t = shuffle_data(x, t)
    
    return x, t
