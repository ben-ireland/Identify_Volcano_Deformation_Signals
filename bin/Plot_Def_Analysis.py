#!/usr/bin/env python

# Collection of functions to plot LiCSBAS timeseries datacubes analyses from Def_Analysis.py
from cmcrameri import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np

def PlotCumulativeDisplacement(Lat,Lon,LOS,DatesDT):
    # Find maximum absolute displacement
    CumDisp = LOS[:, :, -1]
    MaxDisp = np.nanmax(abs(CumDisp))
    MaxIdx = np.nanargmax(abs(CumDisp))
    Row, Col = np.unravel_index(MaxIdx,CumDisp.shape)

    # Plot Cumulative displacement
    fig, axs = plt.subplots(1,2, figsize=(20,10),layout="constrained")
    pcm = axs[0].pcolormesh(
        Lon,
        Lat,
        CumDisp,
        shading="auto",
        cmap=cm.vik,
        vmax=MaxDisp,
        vmin=-MaxDisp
    )
    axs[0].axis('image')
    cbar = plt.colorbar(pcm)
    cbar.set_label("LOS Displacement (m)")

    axs[0].scatter(Lon[Row,Col],Lat[Row,Col])
    axs[1].plot(DatesDT,LOS[Row,Col,:])
    fig.suptitle("Cumulative displacement")
    axs[0].set_title("Spatial pattern")
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('LOS displacement (m)')
    axs[1].set_title("Max. Disp. timeseries at " + str(np.round(Lat[Row,Col],decimals=3)) + u"\u00b0" + "N" + "," + str(np.round(Lon[Row,Col],decimals=3)) + u"\u00b0" + "E")

    return fig

def PlotMaskingResults(Lat,Lon,LOS,ForegroundMask,OtherVolcMask,TargBufferKm,OtherBufferKm):
    # Make figure
    CumDisp = LOS[:, :, -1]
    ForegroundMask = ForegroundMask[:,:,0]
    OtherVolcMask = OtherVolcMask[:,:,0]

    MaxDisp = np.nanmax(abs(CumDisp))

    fig, axs = plt.subplots(1,3, figsize=(20,10),layout="constrained")

    for i in range(3):
        if i==0:
            Img = CumDisp
            axs[i].set_title("No mask")
        elif i==1:
            Img = np.where(ForegroundMask,np.nan,CumDisp)
            axs[i].set_title("Target volcano masked (for SNR)")
        elif i==2:
            Img= np.where(OtherVolcMask,np.nan,CumDisp)
            axs[i].set_title("Other volcanoes in frame masked")
        pcm = axs[i].pcolormesh(
            Lon,
            Lat,
            Img,
            shading="auto",
            cmap=cm.vik,
            vmax=MaxDisp,
            vmin=-MaxDisp
        )
        axs[i].axis('image')
        if i==2:
            cbar = plt.colorbar(pcm)
            cbar.set_label("LOS Displacement (m)")

        fig.suptitle("Masking: Target volcano buffer = " + str(TargBufferKm) + "km, Other volc buffer = " + str(OtherBufferKm) + "km")
    return fig

def PlotDispTimeSteps(Lat,Lon,Disp_Steps,LOS,Noise_Steps,StartYr,EndYr,coordinates,YrsFound,DBSCAN_Clusts,Rect1Bounds,Rect2Bounds):
    nRows = Disp_Steps.shape[2]

    fig, axs = plt.subplots(
        nRows+1,
        2,
        figsize=(12, 4 * nRows),
        layout="constrained"
    )

    # Handle the case nRows == 1
    if nRows == 1:
        axs = axs[np.newaxis, :]

    for i in range(nRows):
        CumDispInc = Disp_Steps[:, :, i]
        MaxDispInc = np.nanmax(np.abs(CumDispInc))
        snr = np.abs(CumDispInc) / Noise_Steps[i]

        # -------------------------
        # Column 1: Incremental displacements
        # -------------------------
        ax = axs[i, 0]
        
        pcm1 = ax.pcolormesh(
            Lon,
            Lat,
            CumDispInc,
            shading="auto",
            cmap=cm.vik,
            vmax=MaxDispInc,
            vmin=-MaxDispInc
        )

        mask = np.all(np.squeeze(YrsFound) == [StartYr[i][0]+1, EndYr[i][0]+1], axis=1)
        coords = coordinates[mask]
        ax.plot(
            Lon[coords[:, 0], coords[:, 1]],
            Lat[coords[:, 0], coords[:, 1]],
            "kx",
            markersize=10
        )

        ax.set_aspect("equal")

        # Remove axis labels/tick labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Row title
        ax.set_title(f"Timeperiod {StartYr[i]+1} - {EndYr[i]+1}")

        cbar1 = fig.colorbar(pcm1, ax=ax)
        cbar1.set_label("")  # remove colourbar label

        # -------------------------
        # Column 2: SNR
        # -------------------------
        ax = axs[i, 1]

        pcm2 = ax.pcolormesh(
            Lon,
            Lat,
            snr,
            shading="auto",
            cmap=cm.lipari,
            vmax=5,
            vmin=0
        )

        ax.set_title('SNR')

        ax.set_aspect("equal")

        # Remove axis labels/tick labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        cbar2 = fig.colorbar(pcm2, ax=ax)
        cbar2.set_label("")  # remove colourbar label

    # Plot Cumulative displacement and clustering results
    #Clustering results
    labels = DBSCAN_Clusts.labels_

    CumDisp = LOS[:,:,-1]
    MaxDisp = np.nanmax(abs(CumDisp))
    pcm1 = axs[nRows,0].pcolormesh(
            Lon,
            Lat,
            CumDisp,
            shading="auto",
            cmap=cm.vik,
            vmax=MaxDisp,
            vmin=-MaxDisp
        )
    
    cbar1 = fig.colorbar(pcm1, ax=axs[nRows,0])
    cbar1.set_label("")  
    axs[nRows,0].set_aspect("equal")
    
    # Plot clusters
    for lab in np.unique(labels):
        mask = labels == lab

        if lab==-1:
            axs[nRows,0].scatter(
                    Lon[coordinates[mask, 0], coordinates[mask, 1]],
                    Lat[coordinates[mask, 0], coordinates[mask, 1]],
                    c='k',
                    label="Outlier"
                )
        else:
            axs[nRows,0].scatter(
                    Lon[coordinates[mask, 0], coordinates[mask, 1]],
                    Lat[coordinates[mask, 0], coordinates[mask, 1]],
                    label=f'Cluster {lab+1}'
                )

            # Plot rectangles bounding clusters
            x1 = Lon[0,Rect1Bounds[lab+1,0]]
            x2 = Lon[0,Rect1Bounds[lab+1,1]]
            y1 = Lat[Rect1Bounds[lab+1,2],0]
            y2 = Lat[Rect1Bounds[lab+1,3],0] 
            Wid = abs(x2-x1)
            Hei = abs(y2-y1)
            Rect1 = mpatches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                Wid,
                Hei,
                linewidth=1,
                edgecolor='k',
                facecolor='none'
            )
            axs[nRows,0].add_patch(Rect1)

            x1 = Lon[0,Rect2Bounds[lab+1,0]]
            x2 = Lon[0,Rect2Bounds[lab+1,1]]
            y1 = Lat[Rect2Bounds[lab+1,2],0]
            y2 = Lat[Rect2Bounds[lab+1,3],0] 
            Wid = abs(x2-x1)
            Hei = abs(y2-y1)
            Rect2 = mpatches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                Wid,
                Hei,
                linewidth=1,
                edgecolor='k',
                facecolor='none'
            )
            axs[nRows,0].add_patch(Rect2)
                
    axs[nRows,0].set_xlabel("")
    axs[nRows,0].set_ylabel("")
    axs[nRows,0].set_xticklabels([])
    axs[nRows,0].set_yticklabels([])
    axs[nRows,0].set_title('Clustering results')
    axs[nRows,0].legend(loc='best', fontsize=8)

    # Hide last subplot axis
    fig.delaxes(axs[nRows,1])
    return fig

def PlotDetectedChanges(Lat,Lon,LOS,ChosenTS,ChosenTS_Yrs,Changes,Rect1Bounds,StartEndIdx):

    figs = []
    for k in range(len(ChosenTS)-1):
        #fig, axs = plt.subplots(len(Changes)-1,2, figsize=(20,15))
        fig = plt.figure(figsize=(20, 30))
        gs = fig.add_gridspec(len(Changes[k+1])+2, 2)
        # Plot full timeseries with changepoints
        fig.suptitle('Deforming region ' + str(k+1) + ' out of ' + str(len(ChosenTS)-1))
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.plot(ChosenTS_Yrs[k+1], ChosenTS[k+1], "-")
        ax_top.xaxis.set_major_locator(mdates.YearLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_top.tick_params(axis='both', which='major', labelsize=10)
        # Add detected change points
        Bot, Top = ax_top.get_ylim()
        ax_top.vlines(ChosenTS_Yrs[k+1][Changes[k+1]],Bot,Top,colors='r')
        ax_top.vlines(ChosenTS_Yrs[k+1][Changes[k+1]],Bot,Top,colors='r')
        ax_top.set_title('Full timeseries with change points')

        # Subset timeseries from each deforming region
        SubsetTS = LOS[:,:,StartEndIdx[k+1,0]:StartEndIdx[k+1,1]-1]

        # Generate remaining axes
        axs = []
        for r in range(1, len(Changes[k+1])+2): # Extras to add change:end section of TS
            axs.append([
                fig.add_subplot(gs[r, 0]),
                fig.add_subplot(gs[r, 1])
            ])
        # Plot corresponding image and timecourse for each Period
        for i in range(len(Changes[k+1])+1):
            if i==0:
                StartIdx=0
                EndIdx=Changes[k+1][i]
            elif i==len(Changes[k+1]):
                StartIdx=Changes[k+1][-1]
                EndIdx=SubsetTS.shape[2]-1
            else:
                StartIdx=Changes[k+1][i-1]
                EndIdx=Changes[k+1][i]

            # Get displacement from each deformation episode in this region
            CumDisp = SubsetTS[:,:,EndIdx] - SubsetTS[:,:,StartIdx]
            MaxDisp = np.nanmax(abs(CumDisp))
            pcm1 = axs[i][0].pcolormesh(
                Lon,
                Lat,
                CumDisp,
                shading="auto",
                cmap=cm.vik,
                vmax=MaxDisp,
                vmin=-MaxDisp
            )

            cbar1 = fig.colorbar(pcm1, ax=axs[i][0])
            cbar1.set_label("LOS Displacement (m)") 

            # Plot rectangles bounding deformation region
            x1 = Lon[0,Rect1Bounds[k+1,0]]
            x2 = Lon[0,Rect1Bounds[k+1,1]]
            y1 = Lat[Rect1Bounds[k+1,2],0]
            y2 = Lat[Rect1Bounds[k+1,3],0] 
            Wid = abs(x2-x1)
            Hei = abs(y2-y1)
            Rect1 = mpatches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                Wid,
                Hei,
                linewidth=1,
                edgecolor='k',
                facecolor='none'
            )
            axs[i][0].add_patch(Rect1)
            axs[i][0].set_aspect("equal")
            axs[i][0].set_title('Deformation episode ' + str(i+1) + ' in region ' + str(k+1))

            # Timeseries for each deformation epsiode
            axs[i][1].plot(ChosenTS_Yrs[k+1][StartIdx:EndIdx+1], ChosenTS[k+1][StartIdx:EndIdx+1], "-")
            axs[i][1].set_title('Timecourse')

        figs.append(fig)

    return figs
    