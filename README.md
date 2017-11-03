# my\_reinforcement\_learning\_tutorial
This repositpry created for "Reinforcement Learning Tutorial". For example, 'Value Iteration(state values)', 'Q-Learing', 'Actor-Critic', 'Inverse Reinforcement Learning', 'Deep Q-Learing', and 'Deep Actor-Critic'.

'Deep Q-Learning' implementations are influenced by [EnsekiTT's blog](http://ensekitt.hatenablog.com/entry/2016/11/28/035827) and [trtd56's blog](https://qiita.com/trtd56/items/3a09d37788d8d13ff131).

'Inverse Reinforcement Learning' implementations are forked from [Yiren Lu's implementation](https://github.com/stormmax/irl-imitation).



## Hardware spec
- PC
	- OS : Ubuntu 14.04
	- Memory : 32GB
	- CPU : Intel® Core™ i7-4790K CPU @ 4.00GHz × 8 
	- GPU : GeForce GTX TITAN X
	- Strage : 2TB

## Requirements
- CUDA 7.5+
- Python 2.7.6+
- [Chainer](https://github.com/pfnet/chainer) 2.0.0+
- NumPy 1.13
- [OpenAI Gym](https://github.com/openai/gym) 0.9.2

## Install of dependencies
### Install chainer
```
pip install numpy
pip install cupy
pip install cython
pip install chainer
```
**NOTE: Please see the details [chainer install guide](https://docs.chainer.org/en/v2.0.0/install.html) and [cupy install guide](https://docs-cupy.chainer.org/en/stable/install.html).**

### Install OpenAI Gym
```
$ apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
$ pip install -e '.[all]'
```
**NOTE: Please see the details [openai.com](https://gym.openai.com/)**

## How to run
### Enviroments
