# SOTA: Stabilizing Optimal Transport Alignment for Domain Generalization via a Dynamic Feature Reservoir

## Requirements

- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0

![arch](figs/arch.PNG)
Figure 1: Federated Domain Generalization Architecture. Each client (Domains A, B, C) processes local data through a frozen ViT-Base model followed by  a trainable MoE Adapter. The resulting parameters (µ, σ) are sent to the  federated server for aggregation.

![tSNE](figs/tSNE.png)
Figure 2: Federated Domain Generalization Inference Architecture. The server stores adapter parameters (μ, σ) from multiple domains. For an unseen domain, the client extracts features, calculates similarity to known domains, and creates weighted adapter parameters for final prediction while maintaining frozen ViT-Base backbones.

## Dataset

Firstly create directory for log files and change the dataset path (`pacs_path`, `officehome_path` and `vlcs_path`) and log path (`log_count_path`) in configs/default.py.

Please download the datasets from the official links:

- [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
- [officehome_path](https://hemanthdv.github.io/TheDataset-dataset)
- [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)

## Training / Testing

To train, you can run the command:
`
# Run your program
python3 -m domainbed.scripts.sweep launch\
       --output_dir=PathToOutFolder \
       --command_launcher multi_gpu\
       --algorithms OT_EMAPivotAlign \
       --datasets TheDataset \
       --n_hparams 30 \
       --n_trials 3 \
       --skip_confirmation \
       --data_dir=./domainbed/data/ \
       --single_test_envs \

python -m domainbed.scripts.sweep delete_incomplete\
       --output_dir=PathToOutFolder \
       --command_launcher multi_gpu\
       --data_dir=./domainbed/data/ \
       --skip_confirmation \

python -m domainbed.scripts.collect_results\
       --input_dir=PathToOutFolder
`

## Acknowledgement

Part of our code is borrowed from the following repositories.
- DomainBed [https://github.com/facebookresearch/DomainBed]

We thank the authors for releasing their codes. Please also consider citing their works.
