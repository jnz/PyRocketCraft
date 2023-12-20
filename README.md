PyRocketCraft
-------------

![LOGO](img/pyrocketcraft.png)

Run via:

    python3 rocketcraft.py

![MOVIE](img/rocketlanding.gif)

Installation on Linux and macOS
-------------------------------

Install acados (see section below) next to the PyRocketCraft folder.

Create virtual environment:

    python3 -m venv venv_pyrocketcraft
    source venv_pyrocketcraft/bin/activate

Install Python packages:

    pip3 install numpy scipy gym pybullet

For MPC (acados)
----------------

Install acados [(installation instructions)](https://docs.acados.org/installation/),
ideally in a folder next to this repository.

Link: [https://github.com/acados/acados](https://github.com/acados/acados)

Install the acados module (use the virtual Python `venv_pycoptercraft` environment for
this as well):

    pip3 install -e <acados_root>/interfaces/acados_template

Make sure these environment variables are set:

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
    export ACADOS_SOURCE_DIR="<acados_root>"

You can use `env.sh` to setup the virtual environment and the acados variables:

    source env.sh

macOS note: on macOS it is `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`

    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"$(pwd)/../acados/lib"

Program structure
-----------------

    .
    ├── env.sh                    Setting up a virtual environment (recommended)
    ├── geodetic_toolbox.py       Helper functions
    ├── modelrocket.urdf          Pybullet visualization and physics definition of the rocket
    ├── mpc
    │   └── rocket_model.py       NMPC model definition
    ├── rocketcraft.py            main entry point of application
    └── simrocketenv.py           Physics simulation with gym interface, using pybullet for the heavy lifting

Block diagram:
--------------

The main function in rocketcraft.py runs the NMPC code decoupled from the
physics simulation in a thread. The simulation part is in the simrocketenv
file that is using the OpenAI gym interface.

    ┌────────────────────┐
    │  rocketcraft.py    │
    │  --------------    │
    │                    │
    │  main()            │
    │                    │
    │                    │   'step'  ┌────────────────────┐
    │                    ├◄─────────►│  simrocketenv.py   │
    └───────┬────────────┘           │  ---------------   │
            │           ▲            │                    │
    'keymap'│           │            │  OpenAI gym interf.│
    'state' │           │ 'u'        │                    │
            │           │            └────────────────────┘
            ▼           │
    ┌─────────────────────┐
    │                     │
    │ NMPC Thread         │
    │ nmpc_thread_func()  │          ┌────────────────────┐            ┌────────────────────┐
    │                     │◄────────►│  rocketmodel.py    │◄─────────► │  acados            │
    │                     │          │  --------------    │            │  ------            │
    └─────────────────────┘          │                    │            │                    │
                                     │  NMPC model        │            │  Auto generated    │
                                     │                    │            │  C-code            │
                                     └────────────────────┘            └────────────────────┘

Coordinate Frames
-----------------

pybullet is using:

 - World Frame (enu) East/North/Up(ENU): X = East, Y = North, Z = Up
 - Body Frame (rosbody), X = Forward, Y = Left, Z = Up

OpenGL is using:

 - World Frame (gl) X = Right, Y = Up, Z = Pointing out of screen

The OpenGL X axis is aligned with the North axis (X) of the NED frame.

