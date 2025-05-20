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

pip install jax jaxlib numpy matplotlib seaborn

pip install jupyterlab ipykernel

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

