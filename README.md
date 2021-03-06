# KGE-TSPF
Predicting tissue-specific protein functions using KGE models

## Overview
This code contain scripts and data for experiments on the prediction of tissue-specific protein functions using KGE models used in the paper:
```
Predicting tissue-specific protein functions using multi-part tensor decomposition.
Sameh K. Mohamed  
Information Sciences 508 (2020): 343-357.
```

## Requirements
This repository only require the installation of `tensorflow` (standard or gpu) and the [LibKGE library](https://github.com/samehkamaleldin/libkge).

# Usage
First run the preprocessing script as follows:

``` bash
cd preprocessing
sh preprocessing.sh
```

Then, you can use the experiment script at the `src` dir as follows:
``` bash
cd src
python kge_pipeline_rnk.py
```
The output should be as follows:
```
2020-12-12 22:29:45,894 - TriModel - DEBUG - Logging model parameters ...
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] _predict_pipeline_on: False
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] batch_size          : 4000
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] em_size             : 30
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] initialiser         : xavier_uniform
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] log_interval        : 5
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] loss                : pt_log
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] lr                  : 0.01
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] nb_ents             : 4718
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] nb_epochs           : 200
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] nb_negs             : 2
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] nb_rels             : 48
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] optimiser           : amsgrad
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] predict_batch_size  : 40000
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] reg_wt              : 0.01
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] seed                : 1234
2020-12-12 22:29:45,894 - TriModel - DEBUG - [Parameter] verbose             : 2
2020-12-12 22:29:45,894 - TriModel - DEBUG - Model training started ...
2020-12-12 22:29:45,894 - TriModel - DEBUG - Training model [ 18538 #Instances - 4718 #Entities - 48 #Relations ]
2020-12-12 22:29:46,580 - TriModel - DEBUG - Initialising tensorflow session
2020-12-12 22:29:46,581 - TriModel - DEBUG - Executing tensorflow global variable initialiser
2020-12-12 22:29:46,809 - TriModel - DEBUG - [Training] Epoch # 1    - Speed: 93.096 (k. record/sec) - Loss: 10279.2539 - Avg(Loss): 10279.2539 - Std(Loss): 1621.8347
2020-12-12 22:29:47,319 - TriModel - DEBUG - [Training] Epoch # 5    - Speed: 144.922 (k. record/sec) - Loss: 9432.2129 - Avg(Loss): 10014.3066 - Std(Loss): 1648.9341
2020-12-12 22:29:47,963 - TriModel - DEBUG - [Training] Epoch # 10   - Speed: 144.578 (k. record/sec) - Loss: 5451.1045 - Avg(Loss): 8418.2920 - Std(Loss): 2302.9685
2020-12-12 22:29:48,608 - TriModel - DEBUG - [Training] Epoch # 15   - Speed: 146.101 (k. record/sec) - Loss: 3813.6226 - Avg(Loss): 7109.9175 - Std(Loss): 2688.8430
2020-12-12 22:29:49,256 - TriModel - DEBUG - [Training] Epoch # 20   - Speed: 142.058 (k. record/sec) - Loss: 2290.6965 - Avg(Loss): 6044.5283 - Std(Loss): 2988.8286
2020-12-12 22:29:49,908 - TriModel - DEBUG - [Training] Epoch # 25   - Speed: 139.975 (k. record/sec) - Loss: 1440.6641 - Avg(Loss): 5179.4380 - Std(Loss): 3188.9341
2020-12-12 22:29:50,558 - TriModel - DEBUG - [Training] Epoch # 30   - Speed: 142.860 (k. record/sec) - Loss: 1053.4835 - Avg(Loss): 4513.5640 - Std(Loss): 3271.0330
2020-12-12 22:29:51,203 - TriModel - DEBUG - [Training] Epoch # 35   - Speed: 144.914 (k. record/sec) - Loss: 858.1961 - Avg(Loss): 4001.2144 - Std(Loss): 3278.6777
2020-12-12 22:29:51,843 - TriModel - DEBUG - [Training] Epoch # 40   - Speed: 145.317 (k. record/sec) - Loss: 730.7717 - Avg(Loss): 3600.8101 - Std(Loss): 3245.1162
2020-12-12 22:29:52,482 - TriModel - DEBUG - [Training] Epoch # 45   - Speed: 148.196 (k. record/sec) - Loss: 687.3986 - Avg(Loss): 3278.2424 - Std(Loss): 3192.9150
2020-12-12 22:29:53,124 - TriModel - DEBUG - [Training] Epoch # 50   - Speed: 141.704 (k. record/sec) - Loss: 633.7406 - Avg(Loss): 3014.2393 - Std(Loss): 3131.0708
2020-12-12 22:29:53,768 - TriModel - DEBUG - [Training] Epoch # 55   - Speed: 143.362 (k. record/sec) - Loss: 587.0096 - Avg(Loss): 2794.6682 - Std(Loss): 3065.2126
2020-12-12 22:29:54,416 - TriModel - DEBUG - [Training] Epoch # 60   - Speed: 142.960 (k. record/sec) - Loss: 561.2377 - Avg(Loss): 2608.7590 - Std(Loss): 2998.8982
2020-12-12 22:29:55,058 - TriModel - DEBUG - [Training] Epoch # 65   - Speed: 146.985 (k. record/sec) - Loss: 530.6442 - Avg(Loss): 2450.2261 - Std(Loss): 2933.2285
2020-12-12 22:29:55,693 - TriModel - DEBUG - [Training] Epoch # 70   - Speed: 145.728 (k. record/sec) - Loss: 541.3423 - Avg(Loss): 2313.4285 - Std(Loss): 2869.3472
2020-12-12 22:29:56,327 - TriModel - DEBUG - [Training] Epoch # 75   - Speed: 144.822 (k. record/sec) - Loss: 513.4589 - Avg(Loss): 2194.5322 - Std(Loss): 2807.6262
2020-12-12 22:29:56,958 - TriModel - DEBUG - [Training] Epoch # 80   - Speed: 146.637 (k. record/sec) - Loss: 502.4940 - Avg(Loss): 2088.9521 - Std(Loss): 2749.1333
2020-12-12 22:29:57,592 - TriModel - DEBUG - [Training] Epoch # 85   - Speed: 147.268 (k. record/sec) - Loss: 512.2926 - Avg(Loss): 1994.0753 - Std(Loss): 2694.0151
2020-12-12 22:29:58,224 - TriModel - DEBUG - [Training] Epoch # 90   - Speed: 148.670 (k. record/sec) - Loss: 491.3923 - Avg(Loss): 1910.0503 - Std(Loss): 2641.0212
2020-12-12 22:29:58,852 - TriModel - DEBUG - [Training] Epoch # 95   - Speed: 147.949 (k. record/sec) - Loss: 498.1935 - Avg(Loss): 1835.1074 - Std(Loss): 2590.2683
2020-12-12 22:29:59,474 - TriModel - DEBUG - [Training] Epoch # 100  - Speed: 150.064 (k. record/sec) - Loss: 466.1434 - Avg(Loss): 1767.0248 - Std(Loss): 2542.1221
2020-12-12 22:30:00,103 - TriModel - DEBUG - [Training] Epoch # 105  - Speed: 144.362 (k. record/sec) - Loss: 464.2138 - Avg(Loss): 1705.7006 - Std(Loss): 2496.0408
2020-12-12 22:30:00,746 - TriModel - DEBUG - [Training] Epoch # 110  - Speed: 145.181 (k. record/sec) - Loss: 424.3342 - Avg(Loss): 1648.5186 - Std(Loss): 2452.7400
2020-12-12 22:30:01,381 - TriModel - DEBUG - [Training] Epoch # 115  - Speed: 148.012 (k. record/sec) - Loss: 442.0772 - Avg(Loss): 1597.1370 - Std(Loss): 2410.9685
2020-12-12 22:30:02,016 - TriModel - DEBUG - [Training] Epoch # 120  - Speed: 145.081 (k. record/sec) - Loss: 487.0794 - Avg(Loss): 1549.7090 - Std(Loss): 2371.2051
2020-12-12 22:30:02,651 - TriModel - DEBUG - [Training] Epoch # 125  - Speed: 146.206 (k. record/sec) - Loss: 459.7608 - Avg(Loss): 1505.9036 - Std(Loss): 2333.2324
2020-12-12 22:30:03,293 - TriModel - DEBUG - [Training] Epoch # 130  - Speed: 138.842 (k. record/sec) - Loss: 478.6169 - Avg(Loss): 1465.7327 - Std(Loss): 2296.7742
2020-12-12 22:30:03,949 - TriModel - DEBUG - [Training] Epoch # 135  - Speed: 136.887 (k. record/sec) - Loss: 410.6315 - Avg(Loss): 1427.6140 - Std(Loss): 2262.2725
2020-12-12 22:30:04,626 - TriModel - DEBUG - [Training] Epoch # 140  - Speed: 123.143 (k. record/sec) - Loss: 421.4360 - Avg(Loss): 1392.0830 - Std(Loss): 2229.2231
2020-12-12 22:30:05,348 - TriModel - DEBUG - [Training] Epoch # 145  - Speed: 137.138 (k. record/sec) - Loss: 447.6519 - Avg(Loss): 1358.8444 - Std(Loss): 2197.5491
2020-12-12 22:30:06,028 - TriModel - DEBUG - [Training] Epoch # 150  - Speed: 135.861 (k. record/sec) - Loss: 431.1640 - Avg(Loss): 1328.0914 - Std(Loss): 2166.9785
2020-12-12 22:30:06,710 - TriModel - DEBUG - [Training] Epoch # 155  - Speed: 133.207 (k. record/sec) - Loss: 454.6159 - Avg(Loss): 1299.5120 - Std(Loss): 2137.5286
2020-12-12 22:30:07,393 - TriModel - DEBUG - [Training] Epoch # 160  - Speed: 130.764 (k. record/sec) - Loss: 460.9178 - Avg(Loss): 1272.9913 - Std(Loss): 2109.0789
2020-12-12 22:30:08,102 - TriModel - DEBUG - [Training] Epoch # 165  - Speed: 139.229 (k. record/sec) - Loss: 427.2212 - Avg(Loss): 1247.3417 - Std(Loss): 2081.9795
2020-12-12 22:30:08,788 - TriModel - DEBUG - [Training] Epoch # 170  - Speed: 140.737 (k. record/sec) - Loss: 425.1556 - Avg(Loss): 1223.5251 - Std(Loss): 2055.7397
2020-12-12 22:30:09,465 - TriModel - DEBUG - [Training] Epoch # 175  - Speed: 131.873 (k. record/sec) - Loss: 422.8812 - Avg(Loss): 1201.0320 - Std(Loss): 2030.4368
2020-12-12 22:30:10,151 - TriModel - DEBUG - [Training] Epoch # 180  - Speed: 132.593 (k. record/sec) - Loss: 457.5215 - Avg(Loss): 1179.5514 - Std(Loss): 2006.1135
2020-12-12 22:30:10,834 - TriModel - DEBUG - [Training] Epoch # 185  - Speed: 130.050 (k. record/sec) - Loss: 463.5538 - Avg(Loss): 1159.3783 - Std(Loss): 1982.5544
2020-12-12 22:30:11,578 - TriModel - DEBUG - [Training] Epoch # 190  - Speed: 119.540 (k. record/sec) - Loss: 405.6391 - Avg(Loss): 1140.0607 - Std(Loss): 1959.8625
2020-12-12 22:30:12,288 - TriModel - DEBUG - [Training] Epoch # 195  - Speed: 136.615 (k. record/sec) - Loss: 436.8052 - Avg(Loss): 1122.1792 - Std(Loss): 1937.7560
2020-12-12 22:30:12,961 - TriModel - DEBUG - [Training] Epoch # 200  - Speed: 142.121 (k. record/sec) - Loss: 446.2113 - Avg(Loss): 1104.7755 - Std(Loss): 1916.5063
2020-12-12 22:30:12,961 - TriModel - DEBUG - [Reporting] Finished (200 Epochs) - Avg(Speed): 141.302 (k. record/sec) - Avg(Loss): 1104.7755 - Std(Loss): 1916.5063
============================================================
= Tissue-specific evaluation                               =
============================================================
= AUC-ROC: 0.9907 - AUC-PR: 0.4167 > granulocyte
= AUC-ROC: 0.8014 - AUC-PR: 0.0756 > chondrocyte
= AUC-ROC: 0.9908 - AUC-PR: 0.6667 > artery
= AUC-ROC: 0.7654 - AUC-PR: 0.1042 > adipose_tissue
= AUC-ROC: 0.9604 - AUC-PR: 0.5435 > forebrain
= AUC-ROC: 0.9969 - AUC-PR: 0.8748 > muscle
= AUC-ROC: 0.6490 - AUC-PR: 0.1049 > spleen
= AUC-ROC: 0.4599 - AUC-PR: 0.0344 > testis
= AUC-ROC: 0.9473 - AUC-PR: 0.6191 > embryo
= AUC-ROC: 0.9871 - AUC-PR: 0.8073 > leukocyte
= AUC-ROC: 0.9670 - AUC-PR: 0.5081 > heart
= AUC-ROC: 0.9124 - AUC-PR: 0.3044 > cartilage
= AUC-ROC: 0.9444 - AUC-PR: 0.5871 > natural_killer_cell
= AUC-ROC: 0.9964 - AUC-PR: 0.8667 > retina
= AUC-ROC: 0.9765 - AUC-PR: 0.7162 > nervous_system
= AUC-ROC: 0.9970 - AUC-PR: 0.8635 > lymphocyte
= AUC-ROC: 1.0000 - AUC-PR: 1.0000 > blood_plasma
= AUC-ROC: 0.9290 - AUC-PR: 0.4420 > kidney
= AUC-ROC: 0.9811 - AUC-PR: 0.8339 > blood_vessel
= AUC-ROC: 0.9805 - AUC-PR: 0.6353 > osteoblast
= AUC-ROC: 0.9913 - AUC-PR: 0.3333 > fetus
= AUC-ROC: 0.9286 - AUC-PR: 0.1750 > hair_follicle
= AUC-ROC: 0.8371 - AUC-PR: 0.3209 > blood_platelet
= AUC-ROC: 0.9987 - AUC-PR: 0.8875 > megakaryocyte
= AUC-ROC: 0.9166 - AUC-PR: 0.3570 > bone
= AUC-ROC: 0.4750 - AUC-PR: 0.0125 > mammary_gland
= AUC-ROC: 0.9408 - AUC-PR: 0.4107 > epidermis
= AUC-ROC: 0.5025 - AUC-PR: 0.0156 > macrophage
= AUC-ROC: 0.9959 - AUC-PR: 0.8586 > skeletal_muscle
= AUC-ROC: 0.7731 - AUC-PR: 0.0371 > cerebral_cortex
= AUC-ROC: 0.9647 - AUC-PR: 0.6195 > blood
= AUC-ROC: 0.9995 - AUC-PR: 0.9777 > smooth_muscle
= AUC-ROC: 0.9670 - AUC-PR: 0.7932 > neuron
= AUC-ROC: 0.9857 - AUC-PR: 0.4206 > keratinocyte
= AUC-ROC: 0.9904 - AUC-PR: 0.6930 > eye
= AUC-ROC: 0.9127 - AUC-PR: 0.3970 > uterine_endometrium
= AUC-ROC: 0.9961 - AUC-PR: 0.9101 > cardiac_muscle
= AUC-ROC: 0.8517 - AUC-PR: 0.6946 > glia
= AUC-ROC: 0.5531 - AUC-PR: 0.0224 > thyroid_gland
= AUC-ROC: 0.8000 - AUC-PR: 0.0408 > substantia_nigra
= AUC-ROC: 0.9523 - AUC-PR: 0.5085 > placenta
= AUC-ROC: 0.9076 - AUC-PR: 0.3819 > liver
= AUC-ROC: 0.9997 - AUC-PR: 0.9682 > central_nervous_system
= AUC-ROC: 0.8992 - AUC-PR: 0.3463 > tooth
= AUC-ROC: 0.7897 - AUC-PR: 0.3418 > brain
= AUC-ROC: 0.9222 - AUC-PR: 0.1042 > lung
= AUC-ROC: 0.8519 - AUC-PR: 0.1049 > hematopoietic_stem_cell
= AUC-ROC: 0.5820 - AUC-PR: 0.2548 > hippocampus
============================================================
= AUC-ROC: 0.8858 - AUC-PR: 0.4790 > [AVERAGE]
============================================================
```

# Citation
If use this code for you publication, please cite the following paper:
```
@article{mohamed2020predicting,
  title={Predicting tissue-specific protein functions using multi-part tensor decomposition},
  author={Mohamed, Sameh K},
  journal={Information Sciences},
  volume={508},
  pages={343--357},
  year={2020},
  publisher={Elsevier}
}
```