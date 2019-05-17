## SOM-SNN-Spoken-Digit-Classification

## Background
Humans perform remarkably well for speech recognition using sparse and asynchronous events carried by electrical impulses. 
Motivated by the observations that human brains primarily learn features from environmental stimuli in an unsupervised manner 
and consume extremely low power for complex cognitive tasks, we propose a biologically plausible speech recognition mechanism 
using unsupervised self-organizing map (SOM) for feature representation and event-driven spiking neural network (SNN) for 
spatiotemporal pattern classification. Moreover, we improve the biological realism of the proposed framework by using mel-scaled 
filter bank as the front-end, so as to mimic the human auditory system. The experiments on the TIDIGITS dataset achieve speech
recognition accuracy surpassing those of other bio-inspired systems. 

## How to run the code
+ training script: trainTIDIGITs.m
+ testing script: testTIDIGITs.m

## Citation
If you use code in your research, please cite with:
```
@INPROCEEDINGS{8489535, 
author={J. {Wu} and Y. {Chua} and H. {Li}}, 
booktitle={2018 International Joint Conference on Neural Networks (IJCNN)}, 
title={A Biologically Plausible Speech Recognition Framework Based on Spiking Neural Networks}, 
year={2018}, 
volume={}, 
number={}, 
pages={1-8}, 
keywords={cognition;pattern classification;self-organising feature maps;signal classification;speech recognition;unsupervised learning;biologically plausible speech recognition framework;unsupervised self-organizing map;feature representation;spatiotemporal pattern classification;event-based speech recognition system;biologically plausible speech recognition;event-driven spiking neural network;Neurons;Speech recognition;Feature extraction;Self-organizing feature maps;Spatiotemporal phenomena;Biological neural networks;Task analysis}, 
doi={10.1109/IJCNN.2018.8489535}, 
ISSN={2161-4407}, 
month={July},
}
```


