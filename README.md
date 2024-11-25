# DeepGEM-egf: Generalized Expectation Maximization for Empirical Green's Functions approach

<!--- ![overview image](https://github.com/angelafgao/DeepGEM/blob/main/teaser.jpg) -->

<!-- ## Citation
```
@inproceedings{gao2021deepgem,
 author = {Gao, Angela F and Castillo, Jorge C and Yue, Yisong and Ross, Zachary E and Bouman, Katherine L},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {DeepGEM: Generalized Expectation-Maximization for Blind Inversion},
 volume = {35},
 year = {2021}
}
``` -->


## Requirements and environment setup
General requirements for PyTorch release:
* [pytorch](https://pytorch.org/)

Please check ``` DeepGEM.def``` to build the singularity container.

### A) On your machine, without access to GPU
#### Create new conda env from yml (preferred)


#### Create new conda env from scratch
 ```
 # Set anaconda path
  export PATH=/opt/conda/bin:$PATH

  # Update conda
  conda update -y -n base conda

  # Create environment
  conda create -c conda-forge -n gem python=3.9
    
  conda activate gem 
    
  # Install conda packages; -y is used to silently install
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  conda install -c anaconda xarray
  conda install -y -c conda-forge cartopy numpy ffmpeg matplotlib scipy obspy pandas pillow pyproj pyqt5 seaborn shapely pip 

  conda clean --tarballs

  pip install gpustat
  pip install nvidia-ml-py3

%environment
  export PYTHONPATH=/opt/conda/lib/python3.9/site-packages:$PYTHONPATH
  
```


### B) On an OAR cluster with access to GPU

#### Create new conda env from yml (preferred)


#### Create new conda env from scratch
```
# use oarsub because otherwise uses too much ressources
oarsub -I -t devel -l /nodes=1/gpu=1/migdevice=1
source /applis/environments/conda.sh
source /applis/environments/cuda_env.sh 12.1

conda create -c conda-forge -n gem python=3.8
conda activate gem
conda clean --tarballs
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge matplotlib scipy obspy 
```
And then `exit` to quit the job.

### C) On a machine or cluster with access to GPU

```
  # Set anaconda path
  export PATH=/opt/conda/bin:$PATH

  # Update conda
  conda update -y -n base conda

  # Create environment
  conda create -c conda-forge -n gem python=3.9

  # Install conda packages; -y is used to silently install
  conda config --add channels conda-forge

  conda install pytorch pytorch-cuda torchvision cudatoolkit=11.8 -c pytorch
  conda install -c anaconda xarray
  conda install -y -c conda-forge cartopy numpy ffmpeg matplotlib scipy obspy pandas pillow pyproj pyqt5 seaborn shapely  pip

  conda clean --tarballs

  pip install gpustat
  pip install nvidia-ml-py3

%environment
  export PYTHONPATH=/opt/conda/lib/python3.9/site-packages:$PYTHONPATH
```

  
## Clone git repository

```
mkdir /home/your-path-to-deepgem/
cd /home/your-path-to-deepgem/
git clone https://github.com/thearagon/DeepGEM_EGF.git
```

## Test run
There are two examples. One a very simple toy model example, the other runs with data from the Cahuilla swarm (see next section). For a test run, you can use the toy model example.

### Run on CPU
```
cd /home/your-path-to-deepgem/docs/examples/toy_model/
conda activate gem
./EGF_ex_cpu.sh /home/your-path-to-deepgem/
```

### Run on GPU
```
cd /home/your-path-to-deepgem/docs/examples/toy_model/
# source necessary modules
conda activate gem
./EGF_ex_gpu.sh /home/your-path-to-deepgem/  # this will run on GPU cuda:0
```

## Run other example