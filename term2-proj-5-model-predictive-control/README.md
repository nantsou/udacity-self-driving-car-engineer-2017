# CarND-Controls-MPC

_**Nan-Tsou Liu**_ 2017-06-17

---

## Abstract

_**Controls MPC Project**_ is one of _**Udacity**_ **Self-Driving Car Engineer Nanodegree**. The task is to complete the source codes of MPC controller and run it with the [simulator]((https://github.com/udacity/CarND-PID-Control-Project/releases)) provided by _**Udacity**_. This porject was carried out with the start code provided by _**Udacity**_ which you can check [**HERE**](https://github.com/udacity/CarND-MPC-Project). 

As the final results, I set the following factors of the cost function to make the vehilce can complete the whole track well:

| factor | value |
| --- | --- |
|cost\_cte\_factor|`2.5`|
|cost\_epsi\_factor|`4.0`|
|cost\_v\_factor|`0.1`|
|cost\_delta\_factor|`6000.0`|
|cost\_a\_factor|`1.0`|
|cost\_gap\_delta\_factor|`24000.0`|
|cost\_gap\_a\_factor|`1.0`|

Besides, the assignment requires us to implement an artifical latency which is 100 ms. In order to against the latency, I took the weighted average of first three values of steering and throttle values.

**<sub>click picture for youtube videos</sub>**

[![proj_5_result](assets/proj_5_result.gif)](https://www.youtube.com/watch?v=az_Q-b4menA)

---

## Descriptions

### The Model

Fisrt of all, trajectory is given by a cubic polynomial:

```
f(x) = a0 + a1*x + a2*x^2 + a3*x^3
```

The simple kinematic model, taught in the lessons, was used for this project. It consists vehicle state and actuators.

The vehicle state has following elements:

* px: vehicle's x position.
* py: vehicle's y position.
* psi: vehicle’s orientation or heading direction.
* v: vehicle's speed
* cte: cross track error
* epsi: orientation error


The actuators has following elements:

* delta: steering angle in radians.
* a: acceleration.

The kinematic equations are shown below, which is used to compute the next state based on the current state

```
x(t+1) = x(t) + v(t) * cos(psi(t)) * dt
y(t+1) = y(t) + v(t) * sin(psi(t)) * dt
psi(t+1) = psi(t) + v(t) / Lf * delta * dt
v(t+1) = v(t) + a(t) * dt

etc(t+1) = (f(t) - y(t)) + (v(t) * sin(epsi(t)) * dt)
epsi(t+1) = epsi(t) - psides(t) + v(t) * delta(t) / Lf * dt

where 
1. Lf is defined as the distance between the front of the vehicle and its center of gravity.
2. f(t) is the parameteric form of the idea trajectory at time t.
3. psides(t) is atan of 1st derivative of the trajectory

```

### Timestep Length and Elapsed Duration (N & dt)

At the begnning, I had no idea how to decide timestep lenght (`N`) and elapsed duration (`dt`). So I started the trial with `N` is `7` and `dt` was `0.05` because this `N` was shown in the lesson and I thought `0.05` might be a good start, which the model only predicted the next state for next`350 ms`. However, the reference velocity was set to `40 mph`. Apparently, the settings of `N` and `dt` were not appropriate for that speed. But I wasn't aware of that. Eventually, the vehicle  failed immediately at turning part.

After I was aware of that the predcition was too short for this speed, I increased  `N` to `20` and `dt` to `0.5`, which the the model predicted the next state for next `1` second. Fortunately, the vehicle could complete the whole track with a not bad performance. However, I found that vehicle was getting unstable at turning part as the speed had been increased. And I knew that N affects the loading of the solver so that I have to keep N low. 

Obviously, the performance of the solver was getting better when I reduced `N`. Meanwhile I increased `dt` to make `N*dt` around 1. In the end, I set `N` to `10` and `dt` to `0.1` as the final decision. Also, I realized that the higher seed requires greater `N` because it becomes increasingly important to look further ahead to plan a trajectory with higher speed. But I was quite satisified the performance of this combination of `N` and `dt` while the vehicle ran at the speed `70 mph`.

### Polynomial Fitting and MPC Preprocessing

The viewpoints are transformed into vehicle base coordinates, which is suggested in the **Tips and Tricks**. Then I fit a cubic polynomial (`f(x) = a0 + a1*x + a2*x^2 + a3*x^3`) to the transformed viewpoints.

As the initial state, px, py and psi were set to 0 because the coordinate is a vehicle base coordinate system. And initial `cte` (cross track error) is calculated by the fitted polynomial at `x = 0` and initial `epsi` (orientation error) was calculated by the form of 1st derivative.

### Model Predictive Control with Latency

As the requirements of assignment, I set Latency to `100 ms`. That means the compulated reuslts for the fisrt tiem-step will already be in the past. It results in increasing cross track error and orientation error. Although the solver would try to compensate the resulting increase in cost in the next iteration, the compensation will also be in the past in next iteration.

To fight aganst the effect brought by latency, I used weighted average of first three values of steering and throttle. With `dt` of 100ms, this corresponds a time interval of `200 ms`, which is about twice of latency. I was satisfied the results that the vehicle could complete the track smoothly. And the reason I used wieghted average is because I would like to enhance the effect the second value which is thought as the **"current"** state. I also considered the third value which I wanted to make the output results more predictive.

---

## Reflection

With the quiz in the lesson, I succesfully implemetned the MPC controller. However, the hardest part of this project is tuning the factors I set on the cost function, which I mentioned in **Abstract** section. The cost function looks like follow:

```
cost = cte^2 + epsi^2 + v^2 + delta^2 + a^2 + gap_delta^2 + gap_a^2
```

Honestly, I got some inspiration from the other students in Slack. I firstly set the factors of `delta` and `gap_delta` to 1000 and 1000 respectively. But the result wasn't great that vehicle failed passing the big turning. As the suggestions from other students, I gradually increased the factors of `delta` and `gap_delta` until `10000` and `50000` while the reference velocity was `70 mph`. And the result was getting quite better.

Meanwhile, I noticed that some students added the factors to other term. So, I added other factors too. I spent quite much time on tuning these factors until I satisfied the performance.

One should be noticed is that I set the factor of `v` to 0.1, which constrains the speed quite much. But it make the vehicle run more smoothly. The other factors did not affect the performance much.

All in all, as my mentor said, the fianl project is the beast, which I quite agreed with it. Although the lessons help me to implement the mpc controller, I still spent lots of time on completing `main.cpp` to make the program work. Meanwhile, I had reviewed the lessons many time to find the hints. Of course my mentor gave me some suggestions. But I am glad I could get myself involved in this project, which I did learn much more from the lesson.

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.

---

## Basic Build Instructions

1. Clone this repo.
2. run bulid.sh with command, `./build.sh`
3. Go to the built directory, `cd build` and Run `./mpc`

---

## Call for IDE Profile

Please check [ide profile instruction](ide_profiles/README.md)

### Editor Settings

The following settings are suggested to make the source codes in consistent.

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)
