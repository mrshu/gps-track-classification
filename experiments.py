import numpy
import sys
import xmltodict
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from pomegranate import *


def distance(stop1, stop2):
    """Return distance between two stops in km."""

    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        Courtesy of http://stackoverflow.com/a/15737218
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6367 * c
        return km * 1000.0

    d = haversine(stop1['lat'], stop1['lon'], stop2['lat'], stop2['lon'])
    # return np.sqrt(d**2 + (stop1['ele'] - stop2['ele'])**2)
    return d


def load_gpx(file):
    out = []
    print(file)
    with open(file) as f:
        d = xmltodict.parse(f.read())
        for trk in d['gpx']['trk']:
            if type(trk) is unicode and trk != 'trkseg':
                continue
            elif type(trk) is unicode:
                trk = d['gpx']['trk']

            if type(trk['trkseg']['trkpt']) is not list:
                point = trk['trkseg']['trkpt']
                out.append([float(point['@lat']),
                            float(point['@lon']),
                            float(point['ele']),
                            pd.to_datetime(point['time'])])
                continue
            for point in trk['trkseg']['trkpt']:
                out.append([float(point['@lat']),
                            float(point['@lon']),
                            float(point['ele']),
                            pd.to_datetime(point['time'])])
    return pd.DataFrame(out, columns=['lat', 'lon', 'ele', 'time'])


def gpx_to_diffs(dpx):
    diffs = []
    time_diffs = []
    lonlats = []
    prev_row = None
    for i, row in dpx.iterrows():
        if prev_row is None:
            prev_row = row
            continue
        dt = row['time'] - prev_row['time']
        diffs.append(distance(row, prev_row) / dt.seconds)
        time_diffs.append(divmod(dt.days * 86400 + dt.seconds, 60))
        lonlats.append((row['lat'], row['lon'], row['ele']))
        prev_row = row
    return diffs, time_diffs, lonlats


def gpx_to_2d_diffs(dpx):
    diffs = []
    time_diffs = []
    prev_row = None
    for i, row in dpx.iterrows():
        if prev_row is None:
            prev_row = row
            continue
        diffs.append([prev_row['lon'] - row['lon'],
                      prev_row['lat'] - row['lat'],
                      prev_row['ele'] - row['ele']])
        dt = row['time'] - prev_row['time']
        time_diffs.append(divmod(dt.days * 86400 + dt.seconds, 60))
        prev_row = row
    return diffs, time_diffs


def get_random_trans_mat(trans_type, n_states):
    trans_mat = np.random.uniform(0.0, 1.0, size=(n_states, n_states))
    if trans_type == 'b':
        trans_mat = np.triu(trans_mat)
    elif trans_type == 'c':
        trans_mat = np.triu(trans_mat)
        trans_mat[-1][0] = np.random.uniform()
    elif trans_type == 'a':
        trans_mat = np.triu(trans_mat)
        trans_mat = np.tril(trans_mat, 1)

    trans_mat /= trans_mat.sum(axis=1)
    return trans_mat


def visualize_prediction(pred, dist, time_diff, lonlats, filename):
    colors = ['red', 'green', 'blue', 'yellow', 'pink']
    with open("{}.csv".format(filename), 'w') as f:
        f.write('name,color,lat,lon,elevation\n')
        for p, d, t, ll in zip(pred, dist, time_diff, lonlats):
            f.write("W{},{},{},{},{}\n".format(p, colors[p],
                                               ll[0], ll[1], ll[2]))

if __name__ == '__main__':
    files = []
    time_diffs = []
    vars = []
    means = []
    lonlats = []
    filenames = []
    for f in sys.argv[1:]:
        file, time_diff, lonlat = gpx_to_diffs(load_gpx(f))
        filenames.append(f)
        files.append(file)
        time_diffs.append(time_diff)
        lonlats.append(lonlat)

        vars.append(np.std(files[-1]))
        means.append(np.mean(files[-1]))

    seconds = [y[0] for x in time_diffs for y in x]
    print(min(seconds), max(seconds))

    print(vars)
    print(means)

    dists = [NormalDistribution(10.0, 2.0),
             NormalDistribution(30.0, 10.0),
             NormalDistribution(100.0, 50.0),
             NormalDistribution(150.0, 50.0),
             NormalDistribution(250.0, 50.0)]
    L = len(dists)
    starts = np.random.uniform(size=(L,))
    ends = np.random.uniform(size=(L,))
    trans_mat = get_random_trans_mat(trans_type='d', n_states=len(dists))
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
    model.bake()
    model.fit(files, verbose=1, n_jobs=8)

    print(model.dense_transition_matrix())
    print(model)

   ## Model drawing
   #import matplotlib.pyplot as plt
   #model.plot()
   #plt.show()

    for file, filename, time_diff, lonlat in zip(files, filenames,
                                                 time_diffs, lonlats):
        print('Generating {} output'.format(filename))
        visualize_prediction(model.predict(file), file, time_diff, lonlat,
                             filename)
