# CarND-Path-Planning-Project

_**Nan-Tsou Liu**_ 2017-08-25

---

## Update 2017.08.28

Because the previous reviewer pointed out that

>This car comes into contact with other cars often on the highway after a few miles and stops moving.

However, this issue never happened on my laptop.

As what the forum mentor suggested, I attached a 15 miles accident-free driving video here.


## Abstract

_**Path Planning Project**_ if the first project in term 3 of _**Udacity**_ **Self-Driving Car Engineer Nanodegree**.
The task is to implement the source code of a planner that safely navigate the car with other traffic.
This porject was carried out with the start code provided by _**Udacity**_ which you can check [HERE](https://github.com/udacity/CarND-Path-Planning-Project).
And the [simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2) which is used to check your own implementation.

![path_planning_result](images/path_planning.gif)

---

## Descriptions

### Structure

* **main.cpp**: The main source provided by _**Udacity**_ that communicate with the simulator. I modified it to adapt to my planner source.
* **planner.cpp** (planner.h): The main source of the planner. It controls the logic to generate and evaluate trajectories in Frenet space.
* **way_points.cpp** (way_points.h): Load way points in highway_map.csv and provides the functions about coordinates.
* **polynomial.cpp** (polynomial.h): Used for jerk minimized trajectory. It provides the functions to get three orders of derivatives at a certain time step.
* **car.h**: Define the struct of cars (vehicle) of my car and other cars.
* **utils.h**: Provides common functions like rag2deg, deg2rad and distance which are used in all the files.
* **cost_function.h**: Define all the cost functions which are used in planner.
* **spline.h**: Suggested in the lesson that handle polynomial fitting (http://kluge.in-chemnitz.de/opensource/spline/)

<br/>

### Local Space

It is more convenient to handle the planner in local space rather than global space. 
I generated the local space from the closest 30 points (previous 10 points and next 20 points). 
I fitted spline in relations with S for `x`, `y`, `dx` and `dy`.
They are mainly used in converting Frenet coordinates to Cartesian coordinates and enabling my car to follow the curvature of the road properly and jerk-free.

<br/>

### Action & Decision

I defined four actions, `go straight`, `follow lead`, `change left` and `change right`. The possible actions are determined by the states of my car and other cars in each update cycle.
After possible actions have been determined, The end points of each actions were generated with `10` perturbed samples as the variations.
Finally, they are turned into jerk minimized trajectories.

There trajectories were then checked for the feasibility such as collision, exceeding speed limit and so on. 
If a trajectory fail feasibility check, `calculate_cost` would return infinity as the cost and skip cost calculation for this trajectory.

Those trajectories which pass feasibility check would be evaluated and be given a weighted cost.
Finally, the trajectory with lowest cost would be chosen as the final output.

<br/>

### Smooth Path

In order to smooth the running path, I defined `resue_previous_range` and `smooth_range`. 
The former defines the range of using previous path and the latter defines the range of points to be smoothed.   
I also define an equation `smooth_scale_fac = (smooth_range - (i - reuse_prev_range)) / smooth_range` to calculate the values of points in `smooth_range`.

<br/>

### Cost Function

There are two parts of `cost_functions`, one is checking the feasibility and the other is calculating the cost. 
As what I mentioned in [**Action & Desicion**](#Action-&-Decision) section, 
if a trajectory fails the feasibility check then cost calculation will be skipped and return infinity as the cost.

Besides, in order to calculate the cost, **logistic** function is defined as follow:

```
double logistic(double x) {
  return (2.0 / (1 + exp(-x)) - 1.0);
}
```

Thus, the basic format of the function would be like follow:

```

double cost(const pair<Polynomial, Polynomial> &traj, const double time_step, other args...) {
  double cost = 0.0;
  for (int i = 0; i < time_step; i++) {
    cost += abs(traj.first.eval(i, "second"));
  }
  return logistic(cost)*cost_weights["acc_s_cost"];
}

```

_**Feasibility Check**_

**check_exceed_speed_limit**

Scan all the time steps to calculate the speed by **first** order of derivative of jerk minimized trajectories for both s and d.
If the summation of value of s and value of d is greater max velocity set as `49.5*.02/22.4`, then this trajectory fails this check

**check_exceed_accel_limit**

Scan all the time steps to calculate the speed by **second** order of derivative of jerk minimized trajectories for s.
If the calculated result is greater max acceleration set as `10/50`, then this trajectory fails this check.

**check_exceed_jerk_limit**

Scan all the time steps to calculate the speed by **third** order of derivative of jerk minimized trajectories for s.
If the calculated result is greater max velocity set as `10/50`, then this trajectory fails this check.

**check_collision**

Scan all the time steps and all the other cars to check whether there is an other car involve in the **critical area** of my car along with this trajectory.
**critical area** is defined as a area with original point is my car, width is 2.5 times of my car's width and length. 
One should be noticed is that if the other car is behind my car but its velocity is faster than my car,
I increased critical length to 5 times of my car's length to prevent the collision during the changing lane. 
By the way, I define car's width and length as `2.5` and `5.0` respectively.  
Thus, if there is any other car involve into the **critical area** then this trajectory fails this check.

_**Cost Calculation**_

**traffic_distance_cost**

`traffic_distance_cost` considers whether there is any car is too close to my car when my can is running on this trajectory.
In order to define "too close", **safe area** is defined as the area with my car's width and 5 times of my car's length.
If the targeted other car is behind my car and the distance is more than 10 points, then this other car is ignored.
The cost is defined as follow:

```
logistic(1 - (diff_s / car_safe_length)) / time_step;
```

where `diff_s` is the difference between my car and other car in s and car_safe_length is 5 times of my car's length

**total_accel_s_cost**

Scan all the time steps to calculate the summation of acceleration of s. 
Acceleration of s is calculated by the second order derivative of jerk minimized trajectory.

**total_accel_d_cost**

Scan all the time steps to calculate the summation of acceleration of d. 
Acceleration of d is calculated by the second order derivative of jerk minimized trajectory.

**total_jerk_cost**

Scan all the time steps to calculate the summation of jerk of s and d. 
Jerk of s and d are calculated by the third order derivative of jerk minimized trajectory.
This function considers Jerk of s and d together.

**busy_lane_cost**

`busy_lane_cost` considers whether this trajectory goes into a busy lane when the action has been determined as **change left** and **change right**.
**busy lane** is mainly defined by the difference of position of s and velocity of s between my car and the targeted other car.
The targeted other car is closest other car from my car but on the lane of end points.

I firstly checked the current lane and future lane are the same lane or not. If they are the same lane, then just return `0.0` as the cost.
If not on the same lane, difference of s between my car and targeted other car would be calculated. 
And then the function would check whether there is a closest other car on the current lane.

If there is and the difference of s between the targeted other car from my car is below 100 points, 
then the velocity of targeted other car is faster or slower than that of the other car on current lane.

If velocity of the targeted other car is slower, then the function will return a big value as the cost to prevent changing lane.
Otherwise, `busy_lane_cost` would be calculated.

By the way, in order to calculate `busy_lane_cost`, I set the time steps to `180` rather than `50`. 

<br/>

### Tricks

**Set Intervals**

Because I set the time steps to `180`, I also define `interval_` to avoid updating the planned path every .02 second.
As we know the size of the previous path is decreasing because my car is moving forward,
so I set that if the size of the previous path is smaller than `tiem_steps - interval_`, which is `180` and `40`,
then the planner would update the planned path.

Besides, the last points in previous path could not be used because of setting `interval_`.
Thus, I stored points at `interval_ + 1` in each updating cycle as the start points for next update.
The information I stored not only the positions s and but also the velocity and the acceleration of s and d.

**Slow Down The Speed**

The planner handle the path planning and car states in Frenet space and s-coordinates of Frenet space are based on the center median of he highway.
Thus, my car actually cover more distance in global space than what was planned in Frenet during turing on the outside lane.
And therefore in order to avoid exceeding acceleration limit, I simply set the logic to slow down the speed limit used for deciding the velocity at the end points of the trajectories.  

**Emergency Mode**

As I mentioned above that I define **safe area**, this area also helps my car be away from the collision when there is other car changes into my current lane suddenly.
Once the planner detect there is other car involve in 0.7 times of safe area's length and 0.75 times of safe area's width,
**emergency mode** would be activated, which forces a hard braking maneuver and disallowing lane changing.

---

## Reflection

At the beginning, I had followed the instruction and project walkthrough to implement the planner in Cartesian coordinate system.
However, I got the hard time to implement the cost function to make the planner decide the action. 
At that time, I review the lessons and look around the slack. I notice **Polynomial Playground (making PTG work)**'s example using Frenet coordinate system rather than Cartesian coordinate system.
And therefore I decided to change my program from Cartesian coordinate system into Frenet coordinate system.
And I consulted this example to construct the cost functions, especially for `logistic` function.
The next hard time is to generate the local space for Frenet coordinate system. At this moment, I noticed that there is a student (I am sorry for that I forgot who he/she is) also using Frenet coordinate system.
Then I eventually created `fit_spline_segment` function to help me to generate the local space. 
The final hard time is to decide the constants and the parameters. These constants and parameters do influence the behavior of my car.
Tuning the parameters did cost me much much time, especially for the situation that my car gets stuck by the leading cars and hardly changes the lanes. 

Before doing this project, honestly I can not fully understand the lessons because I hardly imaged the situation very well.
By doing this project, I could observe the situations and my car's behaviors when I was running the program and the simulator.
Of course there are many parts of my planner can be improved. More states and actions can be added to my program to make more suitable decision.
For examples, I noticed that the leading car is sometimes braking to a very slow speed, which results in exceeding acceleration limit as my car need to slow down to a very slow speed in a short term.
So, it may be a good idea to increase the distance from the leading car if the planner recognize it is potentially dangerous.

All in all, this is a very interesting project and it make me learn more from the lessons. 
This project took me most time compared with the other projects I have done.
But it also the first project that make me feel I am building self-driving car.  

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

---

## Basic Build Instructions

1. Clone this repo.
2. run bulid.sh with command, `./build.sh`
3. Go to the built directory, `cd build` and Run `./path_planning`

---

## Call for IDE Profile

Please check [ide profile instruction](https://github.com/ddrsmile/CarND-Extended-Kalman-Filter-Project/blob/master/ide_profiles/README.md)

### Editor Settings

The following settings are suggested to make the source codes in consistent.

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

