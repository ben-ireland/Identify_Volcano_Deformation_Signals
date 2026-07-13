#!/usr/bin/env python

# Collection of functions to analyse LiCSBAS timeseries datacubes

# Import
from cmcrameri import cm
from skimage import filters
from skimage.feature import peak_local_max
from skimage import measure
from skimage import morphology
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks, peak_prominences, convolve2d
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.spatial.distance import squareform
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.colorizer as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import h5py
import numpy as np

from Pedro_Scripts import get_volcano_info

# DATA PREPROCESSING
def MaskNearbyVolcanoes(GVP,VolcFile,Lon,Lat,VolcName,LOS,TargBufferKm,OtherBufferKm):
    # Get volc name
    #matches = GVP['Volcano Name'].str.contains(VolcName, case=False, na=False)
    # matches = GVP[GVP.iloc[:,1] == VolcName]
    # found = (GVP.iloc[:, 1].astype(str) == VolcName).any()

    # if found:
    #     print("Target volcano location found: " + VolcName)
    # else:
    #     print("No volcano name matching " + VolcName + " found in GVP table.")

    # # Get volc lat and lon
    # volcLat = matches.iloc[0, 8]
    # volcLon = matches.iloc[0, 9]

    Name, volcLon, volcLat, distance = get_volcano_info(VolcName,VolcFile)
    # Convert lat lon to km for buffering
    bufferTargetVolc = TargBufferKm # in km
    bufferOtherVolcs = OtherBufferKm # in km
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(volcLat))

    volc_xy = np.column_stack([
        volcLon * km_per_deg_lon,
        volcLat * km_per_deg_lat
    ])

    # Convert image pixels to km coordinates
    LL_xy = np.column_stack([
        (Lon.ravel() * km_per_deg_lon),
        (Lat.ravel() * km_per_deg_lat)
    ])

    # Use KDTree to find pixels within buffer distance of volcano
    LL_tree = cKDTree(LL_xy)
    neighbors = LL_tree.query_ball_point(volc_xy, r=bufferTargetVolc)
    mask = np.zeros(LOS.shape, dtype=bool)
    for idx in neighbors:
        Row, Col = np.unravel_index(idx, (LOS.shape[0], LOS.shape[1]))
        mask[Row, Col, :] = True

    LOS_ThisVolc = np.copy(LOS)
    LOS_ThisVolc[mask] = np.nan

    # Find other volcanoes within the LOS extent + Buffer distance
    MaxLon = (np.max(LL_xy[:, 0]) + bufferOtherVolcs)/km_per_deg_lon
    MinLon = (np.min(LL_xy[:, 0]) - bufferOtherVolcs)/km_per_deg_lon
    MaxLat = (np.max(LL_xy[:, 1]) + bufferOtherVolcs)/km_per_deg_lat
    MinLat = (np.min(LL_xy[:, 1]) - bufferOtherVolcs)/km_per_deg_lat

    GVP.iloc[:, 8] = pd.to_numeric(GVP.iloc[:, 8], errors="coerce")
    GVP.iloc[:, 9] = pd.to_numeric(GVP.iloc[:, 9], errors="coerce")

    # Filter to nearby volcanoes except target
    Frame_Volcs = GVP[
        (GVP.iloc[:, 8] >= MinLat) &
        (GVP.iloc[:, 8] <= MaxLat) &
        (GVP.iloc[:, 9] >= MinLon) &
        (GVP.iloc[:, 9] <= MaxLon) &
        (GVP.iloc[:, 1] != VolcName)
    ]

    mask2 = np.zeros(LOS.shape, dtype=bool)
    if not Frame_Volcs.empty:
        print("Nearby volcanoes found:")
        print(Frame_Volcs.iloc[:, 1])
        # Create mask based on nearby volcanoes
        nearVolc_xy = np.column_stack([
            Frame_Volcs.iloc[:, 9] * km_per_deg_lon,
            Frame_Volcs.iloc[:, 8] * km_per_deg_lat
        ])

        # Don't mask other volcano if within X km of the target volcano (in case name is wrong or other volc is too close)
        Distances = np.sqrt((nearVolc_xy[:,0].astype(np.float64) - volc_xy [:,0])**2 + (nearVolc_xy[:,1].astype(np.float64) - volc_xy [:,1])**2)
        FarAway = Distances > 5
        nearVolc_xy = nearVolc_xy[FarAway,:]

        # Create mask of points
        neighbors2 = LL_tree.query_ball_point(nearVolc_xy, r=bufferOtherVolcs)

        for idx in neighbors2:
            Row, Col = np.unravel_index(idx, (LOS.shape[0], LOS.shape[1]))
            mask2[Row, Col, :] = True

        # LOS_NeabyVolcs = np.copy(LOS)
        # LOS_NeabyVolcs[mask2] = np.nan

    return mask, mask2

## Signal-noise-ratio
def get_SNR(fullData, maskData, binWidth):
    maxD = np.nanmax(np.abs(fullData))
    # Plot histogram of temporal noise across each step outside the target volcano mask
    temporal_noise = np.nanstd(maskData, axis=2)
    temporal_noise = temporal_noise.flatten()
    temporal_noise = temporal_noise[~np.isnan(temporal_noise)]

    # Plot histogram of temporal noise
    binEdges = np.arange(-1,1,binWidth)
    counts, edges = np.histogram(temporal_noise, bins=binEdges)

    # Get bin with maximum count
    edge_max1 = edges[np.argmax(counts)]
    edge_max2 = edge_max1 + binWidth

    # Get mean of those in max bin
    Noise = np.mean(temporal_noise[(temporal_noise >= edge_max1) & (temporal_noise < edge_max2)])
    SNR = maxD/Noise
    return SNR, maxD, Noise, counts, edges, temporal_noise

### FOR K SHAPE CLUSTERING
def normalize_timeseries(X):
    """
    X: (n_series, n_timesteps)
    Returns z-normalized version required for KShape.
    """
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    return scaler.fit_transform(X)  # returns shape (n, m, 1)


def compute_silhouette(X, labels):
    """
    Silhouette score for time series clustering.
    We flatten because silhouette expects 2D input.
    """
    X_flat = X.squeeze()  # (n, m)
    return silhouette_score(X_flat, labels, metric="euclidean")


def find_best_kshape(X, k_min=2, k_max=10, random_state=0):
    """
    Searches for best number of clusters using silhouette score.
    """
    X_scaled = normalize_timeseries(X)

    best_k = None
    best_score = -1
    best_model = None

    scores = {}

    for k in range(k_min, k_max + 1):
        model = KShape(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(X_scaled)

        score = compute_silhouette(X_scaled, labels)
        scores[k] = score

        print(f"k={k}, silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    print(f"\nBest k: {best_k} (silhouette={best_score:.4f})")

    return best_model, best_k, scores

def GetDispSteps(days_,T_winsize,LOS,ForegroundMask,binW):
    # Divide datacube into bins of specific time and get indexes to query
    yrs = days_ / 365.25
    TS_Len = yrs[-1]
    yrsQ = np.arange(T_winsize,np.ceil(TS_Len),T_winsize)

    # Mask out foreground
    LOS_ThisVolc = np.copy(LOS)
    LOS_ThisVolc[ForegroundMask] = np.nan

    idxYrs = np.empty((len(yrsQ),1))
    idxYrs.fill(np.nan)
    for i , j in enumerate(yrsQ):
        idxYrs[i] = np.argmin(abs(yrs-j))
    idxYrs = idxYrs.astype(int).flatten()

    # Pre-allocated variables
    Set_SNR_Inc = np.empty((len(yrsQ),1))
    Set_SNR_Inc.fill(np.nan)
    Set_maxD_Inc = Set_SNR_Inc.copy()
    Set_Noise_Inc = Set_SNR_Inc.copy()
    RowIdxInc = Set_SNR_Inc.copy()
    ColIdxInc = Set_SNR_Inc.copy()
    TimeIdxInc = Set_SNR_Inc.copy()
    StartIdx = Set_SNR_Inc.copy()
    StartYr = Set_SNR_Inc.copy()
    EndYr = Set_SNR_Inc.copy()
    Disp_Steps = np.empty([LOS.shape[0],LOS.shape[1],len(yrsQ)])

    for i in range(len(yrsQ)):
        if i==0:
            # Extract cumulative disp. during timeperiod X-Y
            Set_IncDisp = LOS[:,:,0:idxYrs[i]]
            Set_IncDispMask = LOS_ThisVolc[:,:,0:idxYrs[i]]

            StartIdx[i] = 0
        else:
            # Extract stepwise disp. during timeperiod X-Y
            Set_IncDisp = LOS[:,:,idxYrs[i-1]:idxYrs[i]] - LOS[:,:,idxYrs[i-1]][:, :, None]
            Set_IncDispMask = LOS_ThisVolc[:,:,idxYrs[i-1]:idxYrs[i]] - LOS_ThisVolc[:,:,idxYrs[i-1]][:, :, None]

            StartIdx[i] = idxYrs[i-1]

        # Get spatial and temporal location of highest displacement during the timeperiod
        RowIdxInc[i], ColIdxInc[i], TimeIdxInc[i] = np.unravel_index(np.nanargmax(np.abs(Set_IncDisp)), Set_IncDisp.shape)
        TimeIdxInc[i] = TimeIdxInc[i] + StartIdx[i]
        StartYr[i] = i-1
        EndYr[i] = i
        # Get SNR
        Set_SNR_Inc[i], Set_maxD_Inc[i], Set_Noise_Inc[i], _, _, _ = get_SNR(Set_IncDisp,Set_IncDispMask,binW)

        # Extract step-wise dislacement
        TimeIdxInc[i] = TimeIdxInc[i].astype(int)
        StartIdx[i] = StartIdx[i].astype(int)
        Disp_Steps[:,:,i]  = LOS[:, :, int(TimeIdxInc[i].item())] - LOS[:,:, int(StartIdx[i].item())]
    return Disp_Steps, Set_Noise_Inc, Set_SNR_Inc, StartYr, EndYr, yrsQ, idxYrs

def GetSignalPeaks(Disp_Steps,spatialRes_m,min_radius_km,SNRThresh,Noise,StartYr,EndYr):
    # Minimum radius for signal peaks
    min_dist = round((min_radius_km*1000)/spatialRes_m)

    coordinates_all = []
    coords_snr = []
    YrsFound = []
    nRows = Disp_Steps.shape[2]
    for k in range(nRows):
        # Filter incremental displacement
        CumDispInc = Disp_Steps[:, :, k]
        CumDispInc = ndi.median_filter(CumDispInc, size=min_dist)
        snr = CumDispInc / Noise[k]

        # Mask NaNs
        MaskNew = np.isfinite(CumDispInc)

        # Extract coordinates of spatial peaks
        coordinates = peak_local_max(
            np.abs(CumDispInc),
            min_distance=min_dist,
            threshold_abs=SNRThresh * Noise[k],
            labels=MaskNew.astype(int),
            num_peaks=3,
            exclude_border=True
        )

        coordinates_all.append(coordinates)
        coords_snr.extend(snr[coordinates[:,0],coordinates[:,1]])

        # Find which years or timeperiods the deformation is found in
        for i in range(coordinates.shape[0]):
            YrsFound.append([StartYr[k]+1, EndYr[k]+1])

    # Filter points to only those in areas with SNR > Threshold value by removing rows
    coordinates_all = np.vstack(coordinates_all)
    YrsFound = np.array(YrsFound)
    coords_snr = np.array(coords_snr)

    np.delete(coordinates_all,coords_snr<SNRThresh,0)
    np.delete(YrsFound,coords_snr<SNRThresh,0)

    return coordinates_all, YrsFound

def ClusterSignalPeaks(coordinates_all,eps_dist_km,spatialRes_m,WS,ClusterBufferKM,YrsFound,LOS,yrsQ,idxYrs,DatesDT,days_):
    # Spatially cluster spatial peaks using DBSCAN to detect inliers
    eps_dist = round((eps_dist_km*1000)/spatialRes_m)
    
    DBSCAN_Clusts = DBSCAN(eps=eps_dist,min_samples=2,metric='euclidean').fit(coordinates_all)
    labels = DBSCAN_Clusts.labels_
    # Get unique clusters
    unique_labels = np.unique(labels)

    # Find which clusters showed deformation in which years
    Clust_StartYr = np.empty(shape=unique_labels.shape[0]-1)
    Clust_EndYr = Clust_StartYr.copy()
    ChosenTS = []
    ChosenTS_Days = []
    ChosenTS_Yrs = []
    numPix_ClustBuff = round((ClusterBufferKM*1000) / spatialRes_m)

    Rect1Bounds = []
    Rect2Bounds = []
    StartEndIdx = []
    # For each cluster
    for label in unique_labels:
        mask = labels == label
        Clust_StartYr[label] = np.min(YrsFound[mask,:])
        Clust_EndYr[label] = np.max(YrsFound[mask,:])

        if Clust_StartYr[label] == 0:
            TS_StartIdx = 0
        elif Clust_StartYr[label] == yrsQ[-1]:
            TS_StartIdx = LOS.shape[2]-1
        else:
            TS_StartIdx = idxYrs[yrsQ==Clust_StartYr[label]].item() # .item() to convert from one-element array to scalar

        if Clust_EndYr[label] == 0:
            TS_EndIdx = 0
        elif Clust_EndYr[label] == yrsQ[-1]:
            TS_EndIdx = LOS.shape[2]-1
        else:
            TS_EndIdx = idxYrs[yrsQ==Clust_EndYr[label]].item()

        # Get bounding rectangle around each cluster of points to search for 
        Xmin = np.min(coordinates_all[mask, 1]) - round(numPix_ClustBuff / 2)
        Xmax = np.max(coordinates_all[mask, 1]) + round(numPix_ClustBuff / 2)
        Ymin = np.min(coordinates_all[mask, 0]) - round(numPix_ClustBuff / 2)
        Ymax = np.max(coordinates_all[mask, 0]) + round(numPix_ClustBuff / 2)

        # Get appropriate subset of the LOS for each cluster
        LOS_sub = LOS[Ymin:Ymax+1, Xmin:Xmax+1, :]
        dlos = np.squeeze(LOS_sub[:, :, TS_EndIdx]) - np.squeeze(LOS_sub[:, :, TS_StartIdx])
        kernel = np.ones((WS, WS)) / WS**2
        window_mean = convolve2d(dlos, kernel, mode='same', fillvalue=np.nan)

        # Window with largest absolute mean displacement in LOS subset
        iy, ix = np.unravel_index(
            np.nanargmax(np.abs(window_mean)),
            window_mean.shape
        )

        # Extract mean timeseries for this region
        best_xmin = Xmin + ix
        best_xmax = best_xmin + (WS-1)
        best_ymin = Ymin + iy
        best_ymax = best_ymin + (WS-1)
        Region = LOS[best_ymin:best_ymax+1, best_xmin:best_xmax+1, TS_StartIdx:TS_EndIdx+1]

        mean_ts = np.nanmean(Region, axis=(0, 1))
        # make mean_ts relative
        mean_ts = mean_ts - mean_ts[0]

        ChosenTS_Yrs.append(DatesDT[TS_StartIdx:TS_EndIdx+1])
        ChosenTS_Days.append(days_[TS_StartIdx:TS_EndIdx+1])
        ChosenTS.append(mean_ts)

        Rect1Bounds.append([Xmin, Xmax, Ymin, Ymax])
        Rect2Bounds.append([best_xmin, best_xmax, best_ymin, best_ymax])
        StartEndIdx.append([TS_StartIdx, TS_EndIdx+1])

    Rect1Bounds = np.array(Rect1Bounds)
    Rect2Bounds = np.array(Rect2Bounds)
    StartEndIdx = np.array(StartEndIdx)

    return ChosenTS, ChosenTS_Days, ChosenTS_Yrs, DBSCAN_Clusts, Rect1Bounds, Rect2Bounds, StartEndIdx

def FindChangePoints(ChosenTS, FiltSize,SNRThresh):
    Changes = []
    # For each identified signal
    for k in range(len(ChosenTS)):
        AllIdx = []
        # Low-pass filter
        mean_kernel = np.full((FiltSize), 1/FiltSize)
        smooth_signal = np.convolve(ChosenTS[k], mean_kernel,
                            mode='valid')
        
        # Calculate derivatives
        FirstOrderDiff = np.diff(smooth_signal)
        SecondOrderDiff = np.diff(smooth_signal,n=2)

        # Define temporal noise
        SDev = np.std(np.abs(FirstOrderDiff))
        SDev2 = np.std(np.abs(SecondOrderDiff))

        # Find anomalies in the derivatives
        Idx_Anomalies = np.where(np.abs(FirstOrderDiff)>np.median(np.abs(FirstOrderDiff))+SNRThresh*SDev)[0] # Calculate anomalies
        Idx_Anomalies2 = np.where(np.abs(SecondOrderDiff)>np.median(np.abs(SecondOrderDiff))+SNRThresh*SDev2)[0] 

        # Filter anomalous indicies to get different episodes
        if len(Idx_Anomalies) == 0:
            starts = np.array([], dtype=int)
            ends = np.array([], dtype=int)
        else:
            breaks = np.where(np.diff(Idx_Anomalies) > 2)[0]

            starts = np.r_[Idx_Anomalies[0], Idx_Anomalies[breaks + 1]]
            ends   = np.r_[Idx_Anomalies[breaks], Idx_Anomalies[-1]]

            # Remove isolated anomalies (groups of length 1)
            keep = starts != ends

            starts = starts[keep]
            ends = ends[keep]

        if len(Idx_Anomalies2) == 0:
            starts2 = np.array([], dtype=int)
            ends2 = np.array([], dtype=int)
        else:
            breaks2 = np.where(np.diff(Idx_Anomalies2) > 1)[0]

            starts2 = np.r_[Idx_Anomalies2[0], Idx_Anomalies2[breaks2 + 1]]
            ends2   = np.r_[Idx_Anomalies2[breaks2], Idx_Anomalies[-1]]

            # Remove isolated anomalies (groups of length 1)
            keep2 = starts2 != ends2

            starts2 = starts2[keep2]
            ends2 = ends2[keep2]
        
        # Find corrected index (after mean filtering of TS)
        starts = np.round(starts + FiltSize/2).astype(int)
        ends = np.round(ends + FiltSize/2).astype(int)
        starts2 = np.round(starts2 + (FiltSize+1)/2).astype(int)
        ends2 = np.round(ends2 + (FiltSize+1)/2).astype(int)

        # Combine all filtered break points
        AllIdx.extend(starts)
        AllIdx.extend(ends)
        AllIdx.extend(starts2)
        AllIdx.extend(ends2)

        # Get IDXs and append to other timeseries 
        Change_Idx = np.unique(AllIdx).astype(int)
        Changes.append(Change_Idx)

    # Remove change points within 3(?) data points of each other
    for i in range(len(Changes)):
        BigDiffs = np.diff(Changes[i]) > 3
        BigDiffs = np.insert(BigDiffs,0,True)

        Changes[i] = Changes[i][BigDiffs]

    return Changes  

def LoadLicsbasTS(TS_File):
    # Open timeseries file
    with h5py.File(TS_File, "r") as f:
        ImDates = f["/imdates"][:]
        LOS = f["/cum"][:]
        cLat = f["/corner_lat"][()]
        cLon = f["/corner_lon"][()]
        postLat = f["/post_lat"][()]
        postLon = f["/post_lon"][()]

    # Convert dates
    DatesDT = pd.to_datetime(ImDates.astype(str), format="%Y%m%d")
    days_ = (DatesDT - DatesDT[0]).days.to_numpy()

    LOS = LOS.transpose((1, 2, 0))  # Transpose to (lat, lon, time)

    # Find extent
    endLat = (LOS.shape[0] - 1) * postLat + cLat
    endLon = (LOS.shape[1] - 1) * postLon + cLon

    # Create lat and lon postings and grids
    lat = np.arange(cLat, endLat + postLat/2, postLat)
    lon = np.arange(cLon, endLon + postLon/2, postLon)
    Lon, Lat = np.meshgrid(lon, lat)

    # Covert from mm to m
    LOS = LOS / 1000.0

    return LOS, Lon, Lat, days_, DatesDT