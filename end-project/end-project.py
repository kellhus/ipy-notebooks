from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL


def load_data(files):
    data_dir = 'data/'
    data = pd.concat([pd.DataFrame(np.load(data_dir + filename)[()])
                     for filename in files], ignore_index=True)
    df = pd.DataFrame()
    for column in ['stimon', 'subj', 'stim_names', 'spk_times']:
        df[column] = data[column]
    return df


def add_response_rates(df):
    start, stop = 100, 200
    df['rates'] = df[['spk_times', 'stimon']].apply(lambda x: get_rate(x[0], x[1] + start, x[1] + stop), axis=1)
    return df


def get_rate(spk_times, start, stop):
    spk_count = np.count_nonzero((spk_times < stop) & (spk_times >= start))
    rate = 1000 * spk_count / float(stop - start)
    return rate


def plot_response_rates(df):
    stim_rates = df.groupby(['stim_names'])['rates'].mean()
    stim_rates.plot(kind='bar')
    plt.title('Average Response Rates for Different Stimuli (subject s)')
    plt.ylabel('Average Response Rate (spikes/s)')
    plt.xlabel('Stimulus Name')
    return stim_rates


def plot_averaged_image(stim_rates, low, high, title):
    stimuli = [k for k in stim_rates.keys() if stim_rates.between(low, high)[k]]
    imlist = ['data/stimuli/' + f + '.png' for f in stimuli]
    width, height = PIL.Image.open(imlist[0]).size
    N = len(imlist)
    avg_image = np.zeros((height, width, 3), dtype=np.float)
    for im in imlist:
        imarr = np.array(PIL.Image.open(im), dtype=np.float)
        avg_image = avg_image + imarr[:,:,0:3] / N
    avg_image = np.array(np.round(avg_image), dtype=np.uint8)

    fig = plt.figure(figsize=(10, 3)) 
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 

    ax0 = plt.subplot(gs[0])
    ax0.imshow(avg_image)
    ax0.set_title(title)
    plt.axis('off')

    ax1 = plt.subplot(gs[1])
    ax1.hist(avg_image[...,0].flatten(), 256, range=(0, 254), fc='b', histtype='step')
    ax1.hist(avg_image[...,1].flatten(), 256, range=(0, 254), fc='g', histtype='step')
    ax1.hist(avg_image[...,2].flatten(), 256, range=(0, 254), fc='r', histtype='step')
    ax1.set_title('Histogram of Image Channels')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('Pixel count')
    ax1.set_xlim(100, 254)

    plt.tight_layout()

    # return stimuli
