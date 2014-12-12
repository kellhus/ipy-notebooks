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
    f = lambda x: get_rate(x[0], x[1] + start, x[1] + stop)
    df['rates'] = df[['spk_times', 'stimon']].apply(f, axis=1)
    return df

def get_dominant_HSL_values(stim_rates):
    stim_hsl, stim_rgb = {}, {}
    df = pd.DataFrame()
    stim_names = list(stim_rates.keys())
    stim_rates = stim_rates.values
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
        hsl = colorsys.rgb_to_hls(*[v / 255 for v in peak])
        stim_hsl[stim] = hsl
        stim_rgb[stim] = tuple([v / 255 for v in peak])

    df['Hue'] = [stim_hsl[s][0] for s in stim_names]
    df['Lightness'] = [stim_hsl[s][1] for s in stim_names]
    df['Saturation'] = [stim_hsl[s][2] for s in stim_names]
    df['RGB'] = [stim_rgb[s] for s in stim_names]
    df['stim_rates'] = stim_rates

    return df

def get_rate(spk_times, start, stop):
    spk_count = np.count_nonzero((spk_times < stop) & (spk_times >= start))
    rate = 1000 * spk_count / float(stop - start)
    return rate


def plot_response_rates(df):
    stim_rates = df.groupby(['stim_names'])['rates'].mean()
    stim_rates.plot(kind='bar')
    plt.title('Average Response Rates for Different Stimuli')
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
    ax1.set_xlabel('Bins, RGB values')
    ax1.set_ylabel('Pixel count, n')
    ax1.set_xlim(100, 254)

    plt.tight_layout()

    # return stimuli

def plot_HSL_scatterplots(df):
    plt.style.use('ggplot')
    plt.figure(figsize=(13, 13))
    rmax = df['stim_rates'].max() + 10
    a = 1 / rmax
    for i, color_coord in enumerate(['Hue', 'Saturation', 'Lightness']):
        plt.subplot(1, 3, i + 1, aspect=a)
        plt.scatter(df[color_coord], df['stim_rates'],
                    c=df['RGB'].tolist(), s=50, edgecolor='black')
        plt.title(color_coord + ' vs Neural Response Rate')
        plt.xlabel(color_coord + ' (HSL value)')
        plt.ylabel('Rate (spikes/s)')
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, rmax)

    plt.tight_layout()
