# Subjective Opinion Scores

There are two files containing subjective opinion scores. Each files is arranged with video names associated with their mean subjective opinion scores and their respective standard deviations. 
The video names are provided in *sourceContentName_spatialResolution_frameRate_quantizationParameter* format. (e.g. Beauty_1080p_120hz_q_27)

>__1. ETRI-LIVE_DMOS.csv__   
>- contains Difference Mean Opinion Scores (DMOS) and their respective standard deviation.   
>- can be used to evaluate prediction performances of reference (full/reduced) image/video quality models.  


>__2. ETRI-LIVE_MOS.csv__  
>- contains Mean Opinion Scores (MOS) and their respective standard deviation.  
>- can be used to evaluate prediction performances of no-reference image/video quality models.  

We also provide video information matrix containing space-time resolution and bit rate information of each distorted video.


>__3. ETRI-LIVE_videoInfo.csv__  
>- __videoName__: sourceContentName_spatialResolution_frameRate_quantizationParameter.  
>- __contentIndex__: videos generated from the same source content have the index.  
>- __sourceSpatialResolution__: spatial resolution of the original source content.  
>- __sourceFrameRate__: frame rate of the original source content.  
>- __processedSpatialResolution__: spatial resolution of each distorted video.  
>- __processedFrameRateRatio__: frame rate ratio (distorted/source) of each distorted video.  
>- __bitrate__: bitrate of each distorted videos in Mbps.  
>- __bitrateLevel__: bitrate level of each distorted video where higher number refers to heavier compression.  

