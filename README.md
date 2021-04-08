# <a href="#TODO">ManipulaTHOR: A Framework for Visual Object Manipulation</a>
#### Kiana Ehsani, Winson Han, Alvaro Herrasti, Eli VanderBilt, Luca Weihs, Eric Kolve, Aniruddha Kembhavi, Roozbeh Mottaghi
#### (Oral Presentation at CVPR 2021)
#### <a href="https://prior.allenai.org/projects/manipulathor">(Project Page)</a>--<a href="http://ai2thor.allenai.org/manipulathor">(Framework)</a>--<a href="#TODO">(Video)</a>--<a href="#TODO">(Slides)</a> 

We present <b>ManipulaTHOR</b>, a framework that facilitates <b>visual manipulation</b> of objects using a robotic arm. Our framework is built upon a <b>physics engine</b> and enables <b>realistic interactions</b> with objects while navigating through scenes and performing tasks. Object manipulation is an established research domain within the robotics community and poses several challenges including <b>avoiding collisions</b>, <b>grasping</b>, and <b>long-horizon planning</b>. Our framework focuses primarily on manipulation in visually rich and <b>complex scenes</b>, <b>joint manipulation and navigation</b> planning, and <b>generalization</b> to unseen environments and objects; challenges that are often overlooked. The framework provides a comprehensive suite of sensory information and motor functions enabling development of robust manipulation agents.

This code base is based on <a href=https://allenact.org/>AllenAct</a> framework and the majority of the core training algorithms and pipelines are borrowed from <a href=https://github.com/allenai/allenact>AllenAct code base</a>. 

### Citation

If you find this project useful in your research, please consider citing:

```
   @inproceedings{ehsani2021manipulathor,
     title={ManipulaTHOR: A Framework for Visual Object Manipulation},
     author={Ehsani, Kiana and Han, Winson and Herrasti, Alvaro and VanderBilt, Eli and Weihs, Luca and Kolve, Eric and Kembhavi, Aniruddha and Mottaghi, Roozbeh},
     booktitle={CVPR},
     year={2021}
   }
```

### Contents
<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-armpointnav-task-description">📝 ArmPointNav Task Description</a></li>
<li><a href="#-dataset">📊 Dataset</a></li>
<li><a href="#-sensory-observations">🖼️ Sensory Observations</a></li>
<li><a href="#-allowed-actions">🏃 Allowed Actions</a></li>
<li><a href="#-defining-a-new-task">✨Defining a New Task</a></li>
<li><a href="#-training-an-agent">🏋 Training an Agent</a></li>
<li><a href="#-evaluating-a-pre-trained-agent">💪 Evaluating a Pre-Trained Agent</a></li>
</ul>
</div>

## 💻 Installation
 
To begin, clone this repository locally
```bash
git clone https://github.com/ehsanik/manipulathor.git
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `utils/*.py` - Helper functions and classes including the visualization helpers.
* `projects/armpointnav_baselines`
    - `experiments/`
        + `ithor/armpointnav_*.py` - Different baselines introduced in the paper. Each files in this folder corresponds to a row of a table in the paper.
        + `*.py` - The base configuration files which define experiment setup and hyperparameters for training.
    - `models/*.py` - A collection of Actor-Critic baseline models.  
* `plugins/ithor_arm_plugin/` - A collection of Environments, Task Samplers and Task Definitions
    - `ithor_arm_environment.py` - The definition of the `ManipulaTHOREnvironment` that wraps the AI2THOR-based framework introduced in this work and enables an easy-to-use API.  
    - `itho_arm_constants.py` - Constants used to define the task and parameters of the environment. These include the step size 
      taken by the agent, the unique id of the the THOR build we use, etc.
    - `ithor_arm_sensors.py` - Sensors which provide observations to our agents during training. E.g. the
      `RGBSensor` obtains RGB images from the environment and returns them for use by the agent. 
    - `ithor_arm_tasks.py` - Definition of the `ArmPointNav` task, the reward definition and the function for calculating the goal achievement. 
    - `ithor_arm_task_samplers.py` - Definition of the `ArmPointNavTaskSampler` samplers. Initializing the sampler, reading the json files from the dataset and randomly choosing a task is defined in this file. 
    - `ithor_arm_viz.py` - Utility functions for visualization and logging the outputs of the models.

</p>
</details>

You can then install requirements by running
```bash
pip install -r requirements.txt
```



**Python 3.6+ 🐍.** Each of the actions supports `typing` within <span class="chillMono">Python</span>.

**AI2-THOR <43f62a0> 🧞.** To ensure reproducible results, please install this version of the AI2THOR.

## 📝 ArmPointNav Task Description

<img src="media/armpointnav_task.png" alt="" width="100%">

ArmPointNav is the goal of addressing the problem of visual object manipulation, where the task is to move an object between two locations in a scene. Operating in visually rich and complex environments, generalizing to unseen environments and objects, avoiding collisions with objects and structures in the scene, and visual planning to reach the destination are among the major challenges of this task. The example illustrates a sequence of actions taken a by a virtual robot within the ManipulaTHOR environment for picking up a vase from the shelf and stack it on a plate on the countertop.
   
## 📊 Dataset

To study the task of ArmPointNav, we present the ArmPointNav Dataset (APND). This consists of 30 kitchen scenes in AI2-THOR that include more than 150 object categories (69 interactable object categories) with a variety of shapes, sizes and textures. We use 12 pickupable categories as our target objects. We use 20 scenes in the training set and the remaining is evenly split into Val and Test. We train with 6 object categories and use the remaining to test our model in a Novel-Obj setting. For more information on dataset, and how to download it refer to <a href="datasets/README.md">Dataset Details</a>.

## 🖼️ Sensory Observations

The types of sensors provided for this paper include:

1. **RGB images** - having shape `224x224x3` and an FOV of 90 degrees.  
2. **Depth maps** - having shape `224x224` and an FOV of 90 degrees.
3. **Perfect egomotion** - We allow for agents to know precisely what the object location is relative to the agent's arm as well as to its goal location.


## 🏃 Allowed Actions

A total of 13 actions are available to our agents, these include:

1. **Moving the agent**

* `MoveAhead` - Results in the agent moving ahead by 0.25m if doing so would not result in the agent colliding with something.

* `Rotate [Right/Left]` - Results in the agent's body rotating 45 degrees by the desired direction.

2. **Moving the arm**

* `Moving the wrist along axis [x, y, z]` - Results in the arm moving along an axis (<span>&#177;</span>x,<span>&#177;</span>y, <span>&#177;</span>z) by 0.05m.

* `Moving the height of the arm base [Up/Down]` - Results in the base of the arm moving along y axis by 0.05m.

3. **Abstract Grasp**

* Picks up a target object. Only succeeds if the object is inside the arm grasper.
  
4. **Done Action**

* This action finishes an episode. The agent must issue a `Done` action when it reaches the goal otherwise the episode considers as a failure.

## ✨ Defining a New Task

In order to define a new task, redefine the rewarding, try a new model, or change the enviornment setup, checkout our tutorial on defining a new task <a href="DefineTask.md">here</a>.

## 🏋 Training An Agent

You can train a model with a specific experiment setup by running one of the experiments below:

```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ <EXPERIMENT-NAME>
```

Where `<EXPERIMENT-NAME>` can be one of the options below:

```
armpointnav_no_vision -- No Vision Baseline
armpointnav_disjoint_depth -- Disjoint Model Ablation
armpointnav_rgb -- Our RGB Experiment
armpointnav_rgbdepth -- Our RGBD Experiment
armpointnav_depth -- Our Depth Experiment
``` 

## 💪 Evaluating A Pre-Trained Agent 

To evaluate a pre-trained model, (for example to reproduce the numbers in the paper), you can add `--mode test -c <WEIGHT-ADDRESS>` to the end of the command you ran for training. 

In order to reproduce the numbers in the paper, you need to download the pretrained models from <a href="https://drive.google.com/file/d/1wZi_IL5d7elXLkAb4jOixfY0M6-ZfkGM/view?usp=sharing">here</a> and extract them to pretrained_models. The full list of experiments and their corresponding trained weights can be found <a href="pretrained_models/EvaluateModels.md">here</a>.

```
python3 main.py -o experiment_output -s 1 -b projects/armpointnav_baselines/experiments/ithor/ <EXPERIMENT-NAME> --mode test -c <WEIGHT-ADDRESS>
```
