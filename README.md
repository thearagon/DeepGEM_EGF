# DeepGEM-egf: Generalized Expectation Maximization for Empirical Green's Functions approach

Please refer to this publication when using this code:
```
Théa Ragon, Angela Gao, Zachary Ross (xxx). xxxxx
```
A permanent repository is available at:
```
zenodo xxx
```

DeepGEM-egf is a Bayesian inversion framework that aims at providing reliable and probabilistic estimates of source time functions, and their posterior uncertainty, while jointly solving for the best Empirical Green's functions (EGF) using one or a few events as prior EGFs. Our approach is based on DeepGEM, an unsupervised generalized expectation-maximization framework for blind inversion (Gao et al., 2021). 

*Angela Gao, Jorge Castellanos, Yisong Yue, Zachary Ross, Katherine Bouman (2021). DeepGEM: Generalized Expectation-Maximization for Blind Inversion. Part of [Advances in Neural Information Processing Systems 34 (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021)*


<!--- ![overview image](https://github.com/angelafgao/DeepGEM/blob/main/teaser.jpg) -->



## Requirements and environment setup
General requirements for PyTorch release:
* [pytorch](https://pytorch.org/)

### A) To run on CPU
#### Create  conda environment from yml
```
# Set anaconda path
export PATH=/opt/conda/bin:$PATH

# Update conda
conda update -y -n base conda

# Create conda env
conda env create -f environment_cpu.yml

conda activate gem
```


#### Create conda environment from scratch
 ```
# Set anaconda path
export PATH=/opt/conda/bin:$PATH

# Update conda
conda update -y -n base conda

# Create environment, check pytorch requirements for your system
conda create -c conda-forge -n gem python=3.9 pytorch torchvision torchaudio xarray cartopy numpy ffmpeg matplotlib scipy obspy pandas pillow pyproj pyqt5 shapely
conda activate gem 
conda clean --tarballs
```


### B) On a cluster with access to GPU

#### Create conda environment from scratch
```
# Set anaconda path
export PATH=/opt/conda/bin:$PATH

# Update conda
conda update -y -n base conda

# Create environment, check pytorch requirements for your system
conda create -c conda-forge -n gem python=3.9
conda activate gem
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda xarray
conda install -y -c conda-forge cartopy numpy ffmpeg matplotlib scipy obspy pandas pillow pyproj pyqt shapely pip
conda clean --tarballs
pip install nvidia-ml-py3
```

#### Example on an OAR cluster
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

  
## Clone git repository
```
mkdir /home/your-path-to-deepgem/
cd /home/your-path-to-deepgem/
git clone https://github.com/thearagon/DeepGEM_EGF.git
```

## Test run
There are two examples. One is a simple toy model with a perfectly known forward model. The other runs with data from the Cahuilla swarm (see next section). For a test run, you can use the toy model example.

### Run on CPU
```
cd /home/your-path-to-deepgem/examples/toy_model/
conda activate gem
./EGF_ex_cpu.sh /home/your-path-to-deepgem/
```

### Run on GPU
```bash
cd /home/your-path-to-deepgem/examples/toy_model/
# set up environment
conda activate gem
# run deepGEM-egf
./EGF_ex_gpu.sh /home/your-path-to-deepgem/  # this will run on GPU cuda:0
```

## Cahuilla swarm example
To run the default version of the example:
```bash
cd /home/your-path-to-deepgem/examples/cahuilla_swarm/
# set up environment
conda activate gem
# Prepare the input files
python prepdata.py
# Run deepGEM-egf
./EGF_ex.sh /home/your-path-to-deepgem/
```
You can modify the following options in `prepdata.py`:
```python
stpc = False # if True, downloads events and waveforms from SCEDC, else loads locally
phasenet = 'api' # Use phasenet 'api' to detect phase arrivals. if False, use basic STA/LTA
read_arrivals = True # if True, reads arrivals in json dict
Swave = False # if False, use P wave arrivals
use_gc_cat = True # use Zach Ross's catalog. if False use regular SCEDC cat
use_cc = False # use cross-correlation to select EGFs, else distance to main event only.
dist_egf = 0.8 #km, maximum distance from mainshock to EGFs
nbr_cc = 4 ## maximum number of EGFs to select
```
and in l. 214, you can change the stations:
```
stations = ['BOR', 'CTW', 'BLA2', 'PSD']
```

If using phasenet API, you also need to do: `conda install conda-forge::gradio-client`