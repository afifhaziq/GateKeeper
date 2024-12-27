# GateKeeper
The repository of GateKeeper, code for our *Computer Networks* Journal paper: [**GateKeeper: An UltraLite malicious traffic identification method with dual-aspect optimization strategies on IoT gateways**](https://www.sciencedirect.com/science/article/abs/pii/S1389128624003888)

**Note:**
- ⭐ **Please leave a <font color='orange'>STAR</font> if you like this work!** ⭐

## Dependencies
Ensure the following dependencies are installed:
- Python 3.7+
- PyTorch 1.4.0+

## Dataset
- [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets)
- [IoT-23](https://www.stratosphereips.org/datasets-iot23)
- [CIC-IoT-2022](https://www.unb.ca/cic/datasets/iotdataset-2022.html)
  

## File Structure

- `run_Base.py`: Script to run the Base model.
- `train.py`: Contains functions for training and testing the model.
- `Base.py`: Defines the structure of the Base model.
- `GateKeeper.py`: Defines the structure of the GateKeeper model.
- `KBS_score.py`: Script for calculating and evaluating attention scores.
- `utils_base.py`: Utility functions for dataset construction and iterators.
- `utils_GateKeeper.py`: Utility functions specific to the GateKeeper model.

  
## Usage 
> Adjust **./dataset** to your data
### (1) Train the Base Model
> python run_Base.py --test False
### (2) Get the importance score of each byte 
> python KBS_score.py
### (3) Train the GateKeeper
Copy the results of importance score **(KBS_score.py print)** and assign them to the pos variable in utils_GateKeeper.py.
> python run_GateKeeper.py --test False


## Please quote if it helps you
```bibtex
@article{cao2024gatekeeper,
  title={GateKeeper: An UltraLite malicious traffic identification method with dual-aspect optimization strategies on IoT gateways},
  author={Cao, Jie and Xu, Yuwei and Yu, Enze and Xiang, Qiao and Song, Kehui and He, Liang and Cheng, Guang},
  journal={Computer Networks},
  pages={110556},
  year={2024},
  publisher={Elsevier}
}
@inproceedings{cao2023mathcal,
  title={$$\backslash$mathcal $\{$L$\}$$\{$-$\}$ $ ETC: A Lightweight Model Based on Key Bytes Selection for Encrypted Traffic Classification},
  author={Cao, Jie and Xu, Yuwei and Xiang, Qiao},
  booktitle={ICC 2023-IEEE International Conference on Communications},
  pages={2370--2375},
  year={2023},
  organization={IEEE}
}



