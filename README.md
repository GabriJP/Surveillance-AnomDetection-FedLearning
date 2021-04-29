# Real-Word anomaly detection in Surveillance through Semi-supervised  Federated Active Learning

This project shows the deployment and research on Semi-supervised deep learning models for the anomaly detection in Surveillance videos deployed on a synchronous **Federated Learning** architecture, for which training is being distributed over many nodes.

Federated Learning is a machine learning paradigm for the distributed learning from multiple datasets avoiding the sharing of data beyond the location is being collected. Federated Learning sets a collaborative architecture between multiple training nodes, which performs local fitting on their local data and aggregates the local models to a global model, which collects all the knowledge covered by the individual nodes. As local data is never shared, this contributes to keep privacity, and legal constrains inherent to data.

In another vein, this research is accompanied with the deployment of an **Active Learning** framework for the continuous learning of the model from continuous video recording streams.

To conduct these investigation, unsupervised spatio-temporal learner models proposed by [1,2,3] are took as precedence to propose the base model besides the training and evaluation methods.

This project is being developped as part of a Master's thesis with the supporting of the *Center of Research in the Information technologies and telecommunication* of the University of Granada (CITIC-UGR). For further references follow this [link](https://github.com/cvr-lab).

This repo collects all the scripts, experiment descriptions and tools being developed for the training and evaluation of the models within the different architectures and paradigms.

See the project website at this [link](https://nico-cubero.github.io/Surveillance-AnomDetection-FedLearning/).

## Prerequisites

Following dependencies are required:

* Tensorflow 1.15 / Tensorflow 2.1.0
* OpenCV
* Numpy >=1.18
* scikit-learn

Download the *UCSD Ped 1* and *UCSD Ped 2* [datasets](http://svcl.ucsd.edu/projects/anomaly/dataset.htm), remove the `Test0xx_gt` video folders from Test set and use the label files on `labels` folder.

## Repository structure

This repository is organised into the following folder structure:

* `experiment description`: Experiment configuration JSON documents used for defining and executing all the experiments ran through the training scripts.
* `labels`: Label files for the *UCSD Ped 1* and *UCSD Ped 2* datasets test sets required for the evaluation of the models.
* `scripts`: Contains all the scripts written for the development of experiments, models evaluation and experiment analysis. More usage documentation is provided into this folder and in the preamble of each script.

## References

[1] - Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6479–6488, 2018.

[2] - Rashmika Nawaratne, Damminda Alahakoon, Daswin Silva, and Xing-huo Yu. Spatiotemporal anomaly detection using deep learning for real-time video surveillance. IEEE Transactions on Industrial Informatics,
PP:1–1, 08 2019.

[3] - Yong Shean Chong and Yong Haur Tay. Abnormal event detection in videos using spatiotemporal autoencoder. In International Symposium on Neural Networks, pages 189–196. Springer, 2017.
