# .bash_profile

alias ll="ls -l"
alias la="ls -la"

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi

# User specific environment and startup programs

# Shared executables
# Need to put Anaconda on your path
# GitterLab is a symbolic link to '/mnt/ws/virology/shared/lab folders/GitterLab'
export PATH=~/GitterLab/progs/anaconda/bin:$PATH
export PATH=~/GitterLab/progs/bin:$PATH
export PATH=$PATH:$HOME/bin

# Python matplotlib
# Initial workaround fails for pyplot
#export MATPLOTLIBRC=~/.config/matplotlib
# New workaround from Ben Huebner at WID IT
export MPLCONFIGDIR=/tmp/agitter/.matplotlib

# R libraries
export R_LIBS=~/GitterLab/progs/R/library
