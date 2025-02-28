# TimeCHEAT
The implementation of "TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis"

![framework](https://github.com/user-attachments/assets/ea643b15-8ef6-42ca-92c6-ceacb537f5bf)

Paper: [Arxiv](https://arxiv.org/pdf/2412.12886)

Appendix: 

# Sample

```bash
# classification
python -u main.py -da phy12  -rf 96 -lr 1e-3 -ns 0 -e 60 -bs 64 -pa 8 -l 2 -s 42 -ns 1

# forecasting
python -u main.py -da mimiciii -rf 96 -lr 1e-3 -ns 0 -e 60 -bs 64 -pa 8 -l 2 -ds forecast -s 42 -fi 0

# imputation
python -u main.py -da phy -rf 96 -lr 1e-3 -ns 0 -e 60 -bs 64 -pa 8 -l 2 -ds impute -s 42 -stp 0.5
```


# Citation

```
@inproceedings{liu2024timesurl,
  title={Timesurl: Self-supervised contrastive learning for universal time series representation learning},
  author={Liu, Jiexi and Chen, Songcan},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={39},
  year={2025}
}
```

# Acknowledgement

[GrapFITi](https://github.com/yalavarthivk/GraFITi)
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)
[PatchTST](https://github.com/yuqinie98/PatchTST)


# Email
```
liujiexi@nuaa.edu.cn
alrash@nuaa.edu.cn
```
