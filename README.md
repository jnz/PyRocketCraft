PyRocketCraft
-------------

Control and land a rocket via deep learning or non-linear model predictive
control (NMPC) in a 3D physics environment (using the pybullet engine and the
acados MPC library).

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
    │   └── rocket_model.py         NMPC model definition
    ├── src/nnpolicy.py             Neural Network Controller
    ├── src/mpcpolicy.py            Model Predictive Control Module
    ├── src/rocketcraft.py          main entry point of application
    └── src/simrocketenv.py         Physics simulation with gym interface, using pybullet for the heavy lifting

Block diagram:
--------------

The main function in rocketcraft.py runs the NMPC code decoupled from the
physics simulation in a thread. The simulation part is in the simrocketenv
file that is using the OpenAI gym / Gymnasium interface.

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

Neural Network and Model Predictive Control
-------------------------------------------

Different control policies are available:

    - NNPolicy
    - MPCPolicy

Switch between the policies in rocketcraft.py:

    # policy = MPCPolicy(initial_state)
    policy = NNPolicy()

If MPCPolicy is active you can enable data collection in rocketcraft.py and re-train the network via:

    python3 src/expert_train.py

Coordinate Frames
-----------------

pybullet is using:

 - World Frame (enu) East/North/Up(ENU): X = East, Y = North, Z = Up
 - Body Frame (rosbody), X = Forward, Y = Left, Z = Up

OpenGL is using:

 - World Frame (gl) X = Right, Y = Up, Z = Pointing out of screen

The OpenGL X axis is aligned with the North axis (X) of the NED frame.


Info
----

2023-2024 Jan Zwiener. Free to use for academic research. Contact author for commercial use.
