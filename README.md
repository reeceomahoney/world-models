# World Models

A project implementing the dreamer world model in PyTorch. This repository can run experiments in both OpenAI gym and
RaiSim.

Dreamer: https://danijar.com/project/dreamerv3/

OpenAI gym: https://www.gymlibrary.dev/

RaiSim: https://raisim.com/

### Setup

To start using this repository, you must first have raisim installed. To do this, follow the instructions here:
https://raisim.com/sections/Installation.html. It is assumed that your installation directory is called `$LOCAL_INSTALL`

Once this is complete, clone the repository:

```bash
git clone --recurse-submodules git@github.com:reeceomahoney/world-models
```

Build:

```bash
mkdir build
cd build
cmake ..
make -j4
```

Then install python packages:

```bash
cd ..
pip install -r requirements.txt
```

### Usage

To run an experiment, use the following command:

```bash
cd world_models
python train.py 
```

By default this will run the DITTO algorithm in raisim. To run dreamer instead, add the flag `--ditto False`. To change the environment to one from OpenAI gym, add the `--env` flag followed by the name of the environment. To run from a prexisting agent add the `--agent` flag followed by a path to the corrent `.pt` file.

Logging can be done using tensorboard:

```bash
tensorboard --logdir <log_dir>
```

Finally, to test a pre-existing agent, run the following:

```bash
python run.py --agent <path_to_agent>
```

### TODO

- [ ] systematic eval of the shortest effective wm training time
- [ ] add timestep resolution eval logging
