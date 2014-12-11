from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import colorsys
import scipy
import scipy.misc
import scipy.cluster


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

def add_dominant_HSV_values(df):
    stim_names = df['stim_names']
    H, S, V = [], [], []
    for stim in stim_names:
        im = 'data/stimuli/' + stim + '.png'
        image = PIL.Image.open(im)
        ar = scipy.misc.fromimage(image)[:,:,0:3]
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2])
        codes, dist = scipy.cluster.vq.kmeans(ar, 5)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)
        counts, bins = scipy.histogram(vecs, len(codes))
        index_max = np.argpartition(counts, 1)[1]
        peak = codes[index_max]
        hsv = colorsys.rgb_to_hsv(*[v / 255 for v in peak])
        H.append(hsv[0])
        S.append(hsv[1])
        V.append(hsv[2])

    df['H'] = H
    df['S'] = S
    df['V'] = V

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
    ax1.set_xlabel('Bins, R/G/B values')
    ax1.set_ylabel('Pixel count, n')
    ax1.set_xlim(100, 254)

    plt.tight_layout()

    # return stimuli

def plot_HSV_scatterplots(df):
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(df.H, df.rates)
    plt.title('Hue vs Neural Response Rate')
    plt.xlabel('Hue (unitless)')
    plt.ylabel('Rate (spikes/s)')

    plt.subplot(3, 1, 2)
    plt.scatter(df.S, df.rates)
    plt.title('Saturation vs Neural Response Rate')
    plt.xlabel('Saturation (unitless)')
    plt.ylabel('Rate (spikes/s)')

    plt.subplot(3, 1, 3)
    plt.scatter(df.V, df.rates)
    plt.title('Value vs Neural Response Rate')
    plt.xlabel('Value (unitless)')
    plt.ylabel('Rate (spikes/s)')

    plt.tight_layout()
