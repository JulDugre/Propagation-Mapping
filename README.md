# Propagation-Mapping
## To Run
To run *Propagation Mapping Toolbox* on the Cloud, click on [this link](https://propagation-mapping.streamlit.app/)  !!


***IMPORTANT:** Propagation Mapping works for all parcellations, but **visualisations** currently only work for Schaefer-400 7 Networks (will update soon)

## INFOS
Propagation Mapping is a novel precision framework for reconstructing the neural circuits underlying the spatial organization of human brain maps. It combines the magnitude of regional measures with their underlying connectivity to model how changes in one region propagate across the brain. This user-friendly toolbox offers researchers in neuroscience and psychiatry a versatile and powerful alternative to traditional regional analyses, opening new avenues for discovery in both neurological and psychiatric neuroimaging. The method assumes that the spatial organization of a brain map can be predicted from a general and stable brain architecture. Propagation Mapping relies on group-level functional connectivity and structural covariance estimates from a sample of 1,000 healthy subjects (GSP1000, [Yeo et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21653723/);[Holmes et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26175908/))

![Alt text describing the image](https://github.com/JulDugre/Propagation-Mapping/blob/main/Propagation_Mapping/miscellaneous/Framework.png)


*Please reach out for any questions, suggestions, collaborations (jules [dot] dugre [at] umich [dot] edu)

## Key Features
The core idea behind propagation mapping is that if the spatial organization of brain maps can be accurately predicted by a weighted sum, this indicates that the weighted connectome—prior to summation—faithfully captures the connectivity patterns underlying a given spatial pattern. This reconstructed connectome, referred to as a propagation map, can then be used to generate new hypotheses through graph metrics and connectome-based predictive modeling. Propagation Mapping is an extension of [Cole and colleagues, 2016, Nature Neurosci](https://pubmed.ncbi.nlm.nih.gov/27723746/) and should therefore be cited when using the toolbox. 

The toolbox currently allows researchers to calculate and save:
- **Predicted Regional Map** – a 1-dimensional feature vector representing predicted regional values.  
- **Propagation Map** – a region-by-region matrix capturing the propagation patterns underlying the spatial map.  
- **Residual Map** – a 1-dimensional feature vector representing regional deviations (z-scores) from what is typically expected from a normative connectome (i.e., A higher score means that the magnitude of one's region is higher than typically expected based on the normative reference, and vice-versa).

Users can select their preferred atlas to enable mapping across different modalities (e.g., resting-state, diffusion), including:
- **Schaefer-400 7Networks Atlas** ([Schaefer et al., 2018](https://pubmed.ncbi.nlm.nih.gov/28981612/))  
- **Glasser HCP-MMP1 Atlas** ([Glasser et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC4990127/))  
- **Gordon Atlas** ([Gordon et al., 2016](https://pubmed.ncbi.nlm.nih.gov/25316338/))  
- **Desikan-Killiany Atlas** ([Desikan et al., 2006](https://pubmed.ncbi.nlm.nih.gov/16530430/))  

All atlases are supplemented by 14 additional subcortical regions ([Fischl et al., 2002](https://pubmed.ncbi.nlm.nih.gov/11832223/)) and 7 cerebellar regions ([Buckner et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3214121/)).

## To Cite
To cite the method & toolbox, please use: 
- Dugré, J.R. (2025). Propagation Mapping: A Precision Framework for Reconstructing the Neural Circuitry of Brain Maps. bioRxiv, [DOI: 10.1101/2025.09.27.678975](https://doi.org/10.1101/2025.09.27.678975)

<img width="186" height="243" alt="image" src="https://github.com/user-attachments/assets/3b6554b0-ceb8-4a06-a54b-d9110f804825">

## HOW TO

<video src='https://github.com/user-attachments/assets/e2654c54-bb30-445f-ad6d-d40f9bcbeed1' width=180/>

