## Environment
[CartPole-v1](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) was used.

**state space**

|Num|Observation|Min|Max|
|:----|:----|:----|:----|
|0|Cart Position|-4.8|4.8|
|1|Cart Velocity|-Inf|Inf|
|2|Pole Angle|-0.418 rad (-24 deg)|0.418 rad (24 deg)|
|3|Pole Angular Velocity|-Inf|Inf|

**action space**

|Num|Action|
|:----|:----|
|0|Push cart to the left|
|1|Push cart to the right|

**reward**: Reward is 1 for every step taken, including the termination step

## Result
**The average rewards of last 100 episodes**
![result](result.png)

**Simulation after 10,000 episodes**
> To be posted ASAP
        