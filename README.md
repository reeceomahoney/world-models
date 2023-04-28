# World Models

A project implementing the dreamer world model in PyTorch. This repository can run experiments in both OpenAI gym and
RaiSim.

Dreamer: https://danijar.com/project/dreamerv3/

OpenAI gym: https://www.gymlibrary.dev/

RaiSim: https://raisim.com/

### Setup

To start using this repository, you must first have raisim installed. To do this, follow the instructions here:
https://raisim.com/sections/Installation.html. It is assumed that your installation directory is called
```$LOCAL_INSTALL```

Once this is complete, clone the repository and run the following:

```bash
git clone --recursive-submodules git@github.com:reeceomahoney/world-model.git
python setup.py develop
```

### Usage

To run an experiment, use the following command:

```bash
python dreamer/run.py --env <env_name>
```

Logging can be done using tensorboard:

```bash
tensorboard --logdir <log_dir>
```