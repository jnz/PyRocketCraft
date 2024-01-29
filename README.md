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
    ├── torch_nn_mpc-rocket-vX.pth  Trained network to imitate NMPC
    ├── setup                       Setting up the project for first use
    ├── src/expert_collect.py       Generate data for training
    ├── src/expert_train.py         Train the neural network
    ├── src/geodetic_toolbox.py     Helper functions
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

Run:

    python3 src/expert_collect.py

This will write a `expert_data.json` file with training data (state vector, control input
(u) pairs, etc.). Then a new policy can be trained with this data:

    python3 src/expert_train.py

This will train a neural network based on the MPC data and generate a
`torch_nn_mpc-rocket-vX.pth` file that can be used by the NNPolicy class.

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
