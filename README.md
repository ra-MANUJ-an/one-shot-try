## To set up VS Code Remote SSH for Google Cloud TPU VM

Run the following commands

### Set Environment Variables

export PROJECT_ID=project-id (your project id)

export TPU_NAME=my-v5e-1chip-tpu

export ZONE=europe-west4-b

export ACCELERATOR_TYPE=v5litepod-4

export RUNTIME_VERSION=v2-alpha-tpuv5-lite

### Create TPU VM

gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$RUNTIME_VERSION \
  --preemptible

### Connect via gcloud

gcloud compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE

### Once connected to TPU VM, run these commands:

pkill -f "vscode-server"

pkill -f "code-server"

rm -rf ~/.vscode-server

rm -rf /tmp/vscode-*

### Exit back to your local machine
exit

### Install VS Code Extensions

Search for "Remote - SSH"
Install "Remote - SSH" by Microsoft
Also install "Remote - SSH: Editing Configuration Files" by Microsoft

### Create .ssh Directory (if it doesn't exist)

Open Terminal and type echo $HOME

cd ~
mkdir -p ~/.ssh

### Create SSH Config File

Type "Remote-SSH: Open SSH Configuration File" and select the config file, Add TPU Configuration

### Connect with VS Code

Ctrl+Shift+P → "Remote-SSH: Connect to Host"

### Install Python 3.11 directly via apt

sudo apt update

sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 --version

python3.11 -m venv ~/one-shot-try/venv

source ~/one-shot-try/venv/bin/activate

pip install --upgrade pip

### Install dependencies

export HF_TOKEN=""

export KAGGLE_USERNAME=""

export KAGGLE_KEY=""

pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

python3 -c "import jax; print(jax.devices())"

pip install -U kagglehub huggingface_hub ipywidgets hydra-core tensorflow tensorflow_datasets tensorboardX transformers jinja2 grain datasets sympy pylatexenc

pip install git+https://github.com/google/tunix.git

pip install git+https://github.com/google/qwix.git

pip uninstall -y flax

pip install git+https://github.com/google/flax.git

then bash script.sh, 

bash run_evaluation.sh for baseline model evaluation
