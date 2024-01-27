PyRocketCraft
-------------

Control and land a rocket via deep learning or non-linear model predictive
control (NMPC) in a 3D physics environment (using the pybullet engine).  The
non-linear MPC controller is using the acados library. The neural network parts
are using pytorch.

![LOGO](img/pyrocketcraft.png)

Run the program with

```sh
(source env.sh; ./src/rocketcraft.py)
```

![MOVIE](img/rocketlanding.gif)

Installation on Linux and macOS
-------------------------------

Run `./setup`

Program structure
-----------------

    .
    ├── env.sh                      Setting up the env if coming back
    ├── torch_nn_mpc-rocket-v1.pth  Trained network to imitate NMPC
    ├── setup                       Setting up the project for first use
    ├── src/geodetic_toolbox.py     Helper functions
    ├── src/expert_train.py         Re-train the neural network
    ├── src/modelrocket.urdf        Pybullet visualization and physics definition of the rocket
    ├── src/mpc
    │   └── rocket_model.py         NMPC model and system dynamics definition
    ├── src/nnpolicy.py             Neural Network Controller
    ├── src/mpcpolicy.py            Model Predictive Control Module
    ├── src/rocketcraft.py          main entry point of application
    └── src/simrocketenv.py         Physics simulation with gym interface, using pybullet

Block diagram:
--------------

The main function in rocketcraft.py runs the NMPC code decoupled from the
physics simulation in a thread. The simulation part is in the simrocketenv file
that is using the OpenAI gym / Gymnasium interface and using pybullet in the
background for the heavy lifting of the physics simulation incl. collision
detection.
The `ctrl_thread_func` will either call the MPCPolicy.py OR the NNPolicy.py.
So either the rocket is controlled by a model predictive control algorithm or
a neural network.


    ┌───────────────────┐
    │  rocketcraft.py   │
    │  --------------   │   'state' ┌─────────────────────┐    ┌─────────────────┐
    │                   │◄──────────│  simrocketenv.py    │    │ pybullet        │
    │  main()           │           │  ---------------    │──► │ --------        │
    │                   │   'u'     │                     │    │                 │
    │                   │──────────►│  OpenAI gym env.    │    │ Physics engine  │
    └───────┬───────────┘           │  Physics simulation │    │ and GUI         │
            │      ▲                └─────────────────────┘    └─────────────────┘
            │      │
    'state' │      │ 'u'
            │      │
            ▼      │
    ┌───────────────────┐
    │                   │
    │ Controller Thread │ 'state' >
    │ ctrl_thread_func()│ < 'u'  ┌─────────────────┐       ┌─────────────────┐
    │                   │◄───┬──►│ MPCPolicy.py    │◄────► │ rocketmodel.py  │
    │                   │    │   │ --------------  │       │ --------------  │
    └───────────────────┘    │   │                 │       │                 │
                           or│   │ NMPC controller │       │ NMPC model and  │
                             │   │ u = next(state) │       │ dynamics        │
                             │   └─────────────────┘       └───┬─────────────┘
                             │   ┌─────────────────┐           │   ┌────────────────┐
                             └──►│ NNPolicy.py     │           └─► │ acados         │
                                 │ --------------  │               │ ------         │
                                 │                 │               │                │
                                 │ Neural network  │               │ Auto generated │
                                 │ u = next(state) │               │ C-code         │
                                 └─────────────────┘               └────────────────┘


Neural Network and Model Predictive Control
-------------------------------------------

Different control policies are available:

    - NNPolicy
    - MPCPolicy

Switch between the policies in rocketcraft.py:

    # policy = MPCPolicy(initial_state)
    policy = NNPolicy()

If MPCPolicy is active you can enable data collection in rocketcraft.py.
This will write a .json file with state vector and control input (u) pairs
(observation and action). If this file is sufficiently large (0.5 - 1.0 GB)
you can re-train the network via:

    python3 src/expert_train.py

This will train the network based on the MPC data.

Model Predictive Control
------------------------

The core "magic" of the model predictive control is located in the
`src/mpc/rocket_model.py` file. Here the system dynamics are being described.
The heavy lifting of solving the MPC problem is performed by the awesome acados
library.

Coordinate Frames
-----------------

pybullet is using:

 - World Frame (enu) East/North/Up(ENU): X = East, Y = North, Z = Up
 - Body Frame (rosbody), X = Forward, Y = Left, Z = Up

Info
----

2023-2024 Jan Zwiener. Free to use for academic research. Contact author for commercial use.
