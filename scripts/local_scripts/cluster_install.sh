# Check if dp_matfac_venv already exists
module load gcc/14.2.0

# Install Bazel
if [ ! -f "/tmp/install/bazel-3.7.2-installer-linux-x86_64.sh" ]; then
    echo "Downloading Bazel installer..."
    wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh -P /tmp/install/
    chmod +x /tmp/install/bazel-3.7.2-installer-linux-x86_64.sh
    /tmp/install/bazel-3.7.2-installer-linux-x86_64.sh --user
    export PATH="$PATH:$HOME/bin"
fi

if [ -d "/tmp/dp_matfac_venv" ]; then
    echo "Virtual environment 'dp_matfac_venv' already exists in /tmp."
else
    echo "Creating virtual environment 'dp_matfac_venv' in /tmp..."
    python3.8 -m venv /tmp/dp_matfac_venv
fi

source /tmp/dp_matfac_venv/bin/activate

pip install --upgrade pip

# Create and navigate to installation directory
if [ ! -d "/tmp/install" ]; then
    mkdir -p /tmp/install
fi
cd /tmp/install

# Download and install pre-built jaxlib wheel
# https://storage.googleapis.com/jax-releases/jax_releases.html
if [ ! -f "jaxlib-0.1.75-cp38-none-manylinux2010_x86_64.whl" ]; then
    echo "Downloading jaxlib wheel..."
    wget https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.1.75-cp38-none-manylinux2010_x86_64.whl
fi
pip install jaxlib-0.1.75-cp38-none-manylinux2010_x86_64.whl

# Install tensorflow and its dependencies
pip install tensorflow==2.7.0

# Install tensorflow-federated and its dependencies
# pip install tensorflow-federated==0.19.0
pip install jax==0.2.27

# Install remaining requirements
pip install tensorflow-privacy==0.8.0

pip install ipykernel


# Register the kernel
python -m ipykernel install --user --name=dp-matfac-env --display-name "Python (dp-matfac-env)"



# Create the custom kernel spec directory
KERNEL_DIR=~/.local/share/jupyter/kernels/dp-matfac-env
mkdir -p "$KERNEL_DIR"

# Write the kernel.json
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "/tmp/dp_matfac_venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (dp-matfac-env)",
  "language": "python"
}
EOL

