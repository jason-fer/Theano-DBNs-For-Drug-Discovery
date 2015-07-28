# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
export PIP_CONFIG_FILE=/home/jferiante/.config/pip/pip.conf

# the pip.conf file contains:
# [global]
# target = /usr/local/lib/python2.7/site-packages

export CUDA_ROOT=/usr/local/cuda/7.0.28/cuda
export THEANO_FLAGS='cuda.root=/usr/local/cuda/7.0.28/cuda,device=gpu,floatX=float32'
alias la="ls -la"
