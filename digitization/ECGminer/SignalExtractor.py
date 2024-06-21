# Standard library imports
from math import ceil
from itertools import groupby
from operator import itemgetter
from typing import Iterable

# Third-party imports
import numpy as np
from scipy.signal import find_peaks

# Application-specific imports
#from Rectangle import Rectangle
#from Point import Point
#from Image import Image
from digitization.ECGminer.assets.Image import Image
from digitization.ECGminer.assets.Point import Point
from digitization.ECGminer.assets.Rectangle import Rectangle
from digitization.ECGminer.assets.DigitizationError import DigitizationError

class SignalExtractor:
    """
    Signal extractor of an ECG image.
    """

    def __init__(self, n: int) -> None:
        """
        Initialization of the signal extractor.

        Args:
            n (int): Number of signals to extract.
        """
        self.__n = n
        
    """
    This replaces the default ecg-miner extract signals function
    Args:
        ecg (Image): ECG image output from u-net, from which to extract the signals.

    Raises:
        DigitizationError: The indicated number of ROI could not be detected.

    Returns:
        Iterable[Iterable[Point]]: List with the list of points of each signal
    """
    def get_signals_dw(self, ecg: Image) -> Iterable[Iterable[Point]]:
        thresh = 50
        test_im = ecg.data
        # test_im = abs(image -1)*255 # invert colours - not required if colors already inverted
        rois = self.__get_roi(ecg)
        s = []
        
        for row in rois:
            x, y, endcol = self.__find_start_coords(test_im, rois, row)
            signal = []
            signal_col = []
            signal_col.append([x,y])
         
            while y < endcol:
                # search up
                if x > 0:
                    x_idx = x-1
                    while test_im[x_idx, y] == 0:
                        signal_col.append([x_idx,y])
                        if x_idx > 0:
                            x_idx = x_idx-1
                        else:
                            break
                # search down
                if x < (test_im.shape[0]-2):
                    x_idx = x+1
                    while test_im[x_idx, y] == 0:
                        signal_col.append([x_idx,y])
                        if x_idx < (test_im.shape[0]-2):
                            x_idx = x_idx+1
                        else:
                            break
         
                signal_col = sorted(signal_col, key=lambda x: x[0], reverse=False)
                signal.append(signal_col)
                
                # go to next column and find nearest black pixels
                top_line = signal_col[0][0] # y-coordinate of top of line in previous column 
                bottom_line = signal_col[-1][0] # y-coordinate of bottom of line in previous column 
                y = y+1
                match = 0
                while match == 0 and y < endcol:
                    candidates = np.where(test_im[:,y] == 0)[0]
                    dists = np.concatenate((candidates - top_line, candidates - bottom_line))
                    candidates = np.concatenate((candidates, candidates))
                    if len(candidates) > 0: # TODO HACK FIXME logic breaks if no candidates
                        if np.min(abs(dists))<thresh:                    
                            #bit of voodoo here to try to make sure that we favour returning towards baseline when there is a choice
                            dist_idx = np.where(abs(dists) == min(abs(dists)))[0]
                            if len(dist_idx)>1:
                                #if there is more than one minimum, select the one that takes us closer to the baseline
                                x = candidates[dist_idx]
                                i = np.argmin(abs(x-row))
                                x = x[i]
                            else:
                                x = candidates[dist_idx][0]     
                            signal_col = []
                            signal_col.append([x,y])
                            match = 1
                        elif y == endcol:
                            match = 1
                        else: #skip a column and try again
                            y = y+1
                    else: #skip a column and try again
                        y = y+1
                     
        #sanitise the list
            signal_flat = [x for xs in signal for x in xs]
            s.append(signal_flat)
        point_signal = self.__signal_to_miner(s, rois)
        return point_signal
            
    
    def extract_signals(self, ecg: Image) -> Iterable[Iterable[Point]]:
        """
        Extract the signals of the ECG image.

        Args:
            ecg (Image): ECG image from which to extract the signals.

        Raises:
            DigitizationError: The indicated number of ROI could not be detected.

        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        N = ecg.width
        LEN, SCORE = (2, 3)  # Cache values
        rois = self.__get_roi(ecg)
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        cache = {}

        for col in range(1, N):
            prev_clusters = self.__get_clusters(ecg, col - 1)
            if not len(prev_clusters):
                continue
            clusters = self.__get_clusters(ecg, col)
            for c in clusters:
                # For each row get best cluster center based on minimizing the score
                cache[col, c] = [None] * self.__n
                for roi_i in range(self.__n):
                    costs = {}
                    for pc in prev_clusters:
                        node = (col - 1, pc)
                        ctr = ceil(mean(pc))
                        if node not in cache.keys():
                            val = [ctr, None, 1, 0]
                            cache[node] = [val] * self.__n
                        ps = cache[node][roi_i][SCORE]  # Previous score
                        d = abs(ctr - rois[roi_i])  # Vertical distance to roi
                        g = self.__gap(pc, c)  # Disconnection level
                        costs[pc] = ps + d + N / 10 * g

                    best = min(costs, key=costs.get)
                    y = ceil(mean(best))
                    p = (col - 1, best)
                    l = cache[p][roi_i][LEN] + 1
                    s = costs[best]
                    cache[col, c][roi_i] = (y, p, l, s)

        # Backtracking
        raw_signals = self.__backtracking(cache, rois)
        print(raw_signals)
        return raw_signals

    def __get_roi(self, ecg: Image) -> Iterable[int]:
        """
        Get the coordinates of the ROI of the ECG image.

        Args:
            ecg (Image): ECG image from which to extract the ROI.

        Raises:
            DigitizationError: The indicated number of ROI could not be detected.

        Returns:
            Iterable[int]: List of row coordinates of the ROI.
        """
        WINDOW = 10
        SHIFT = (WINDOW - 1) // 2
        stds = np.zeros(ecg.height)
        for i in range(ecg.height - WINDOW + 1):
            x0, x1 = (0, ecg.width)
            y0, y1 = (i, i + WINDOW - 1)
            std = ecg[y0:y1, x0:x1].reshape(-1).std()
            stds[i + SHIFT] = std
        # Find peaks
        min_distance = int(ecg.height * 0.1)
        peaks, _ = find_peaks(stds, distance=min_distance)
        rois = sorted(peaks, key=lambda x: stds[x], reverse=True)
        if len(rois) < self.__n:
            raise DigitizationError("The indicated number of rois could not be detected.")
        rois = rois[0: self.__n]
        rois = sorted(rois)
        return rois
    
    """Find the start position of the ECG signal for each row.
    N.B. This will fail if signal from an adjacent row gets close to the baseline for this row."""
    def __find_start_coords(self, image, rois, row):
        # assume that the starting point is within 0.75 * gap between lines. This may not be true if signal starts on a particularly high/low QRS complex
        roi_gap = int(np.max(np.diff(rois)) *0.75)
        height = image.shape[0]
        
        top = row - roi_gap
        if top < 0:
            top = 0
        bottom = row + roi_gap
        if bottom > height:
            bottom = height
        
        # find the starting and end column - columns with black pixels within the active region
        colsums = np.min(image[top:bottom,:],0)
        idxs = np.where(colsums == 0)[0]
        startcol = idxs[0]
        endcol = idxs[-1]
        
        # find the start row - the black pixel that is nearest to the baseline. This also doesn't work if adjacent row bleeds into this row
        rowvals = image[:,startcol]
        idxs = np.where(rowvals == 0)[0]
        idx = np.argmin(abs(idxs - row))
        startrow = idxs[idx]
        return startrow, startcol, endcol
    
    """convert ecg sparse matrices into ecg miner format"""
    def __signal_to_miner(self, signals, rois):
        all_sigs = []
        for i, signal in enumerate(signals):
            start = signal[0][1]
            fin = signal[-1][1]
            raw_s = [None] * (fin-start + 1)
            sig = np.array(signal)
            roi = rois[i]
            for j, col in enumerate(range(start, fin+1)):
                idxs = np.where(sig[:,1]==col)[0]
                yvals = sig[idxs][:,0]
                if len(yvals)>0:
                    max_idx = np.argmax(np.abs(yvals - roi))
                    raw_s[j] = Point(col, yvals[max_idx])
                    y = yvals[max_idx]
                else:
                    # if we skipped a column, then use y val from previous column
                    raw_s[j] = Point(col, y)
            all_sigs.append(raw_s)
        return all_sigs
    
    def __get_clusters(
            self, ecg: Image, col: Iterable[int]
    ) -> Iterable[Iterable[int]]:
        """
        Get the clusters of a certain column of an ECG. The clusters are
        regions of consecutive black pixels.

        Args:
            ecg (Image): ECG image.
            col (Iterable[int]): Column of the ECG from which to extract the clusters.

        Returns:
            Iterable[Iterable[int]]: List of the row coordinates of the clusters.
        """
        BLACK = 0
        clusters = []
        black_p = np.where(ecg[:, col] == BLACK)[0]
        for _, g in groupby(
                enumerate(black_p), lambda idx_val: idx_val[0] - idx_val[1]
        ):
            clu = tuple(map(itemgetter(1), g))
            clusters.append(clu)

        return clusters

    def __gap(
            self,
            pc: Iterable[int],
            c: Iterable[int],
    ) -> int:
        """
        Compute the gap between two clusters. It is the vertical white space between
        them. This gap will be 0 if they are in direct contact with each other.

        Args:
            pc (Iterable[int]): Cluster of the previous column.
            c (Iterable[int]): Cluster of the main column.

        Returns:
            int: Gap between the clusters.
        """
        pc_min, pc_max = (pc[0], pc[-1])
        c_min, c_max = (c[0], c[-1])
        d = 0
        if pc_min <= c_min and pc_max <= c_max:
            d = len(range(pc_max + 1, c_min))
        elif pc_min >= c_min and pc_max >= c_max:
            d = len(range(c_max + 1, pc_min))
        # Otherwise clusters are adjacent
        return d

    def __backtracking(
            self, cache: dict, rois: Iterable[int]
    ) -> Iterable[Iterable[Point]]:
        """
        Performs a backtracking process over the cache of links between clusters
        to extract the signals.

        Args:
            cache (dict): Cache with the links between clusters.
            rois (Iterable[int]): List with the row coordinates of the rois.

        Returns:
            Iterable[Iterable[Point]]: List with the list of points of each signal.
        """
        X_COORD, CLUSTER = (0, 1)  # Cache keys
        Y_COORD, PREV, LEN = (0, 1, 2)  # Cache values
        mean = lambda cluster: (cluster[0] + cluster[-1]) / 2
        raw_signals = [None] * self.__n
        for roi_i in range(self.__n):
            # Get candidate points (max signal length)
            roi = rois[roi_i]
            max_len = max([v[roi_i][LEN] for v in cache.values()])
            cand_nodes = [
                node
                for node, stats in cache.items()
                if stats[roi_i][LEN] == max_len
            ]
            # Best last point is the one more closer to ROI
            best = min(
                cand_nodes,
                key=lambda node: abs(ceil(mean(node[CLUSTER])) - roi),
            )
            raw_s = []
            clusters = []
            while best is not None:
                y = cache[best][roi_i][Y_COORD]
                raw_s.append(Point(best[X_COORD], y))
                clusters.append(best[CLUSTER])
                best = cache[best][roi_i][PREV]
            raw_s = list(reversed(raw_s))
            clusters = list(reversed(clusters))
            # Peak delineation
            roi_dist = [abs(p.y - roi) for p in raw_s]
            peaks, _ = find_peaks(roi_dist)
            for p in peaks:
                cluster = clusters[p - 1]
                farthest = max(cluster, key=lambda x: abs(x - roi))
                raw_s[p] = Point(raw_s[p].x, farthest)
            raw_signals[roi_i] = raw_s
        return raw_signals
