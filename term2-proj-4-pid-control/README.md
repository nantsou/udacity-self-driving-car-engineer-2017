# CarND-Controls-PID-Project

_**Nan-Tsou Liu**_ 2017-06-02

## Abstract

_**Controls PID Project**_ is one of _**Udacity**_ **Self-Driving Car Engineer Nanodegree**. The task is to complete the source codes of PID controller and run it with the [simulator]((https://github.com/udacity/CarND-PID-Control-Project/releases)) provided by _**Udacity**_. This porject was carried out with the start code provided by _**Udacity**_ which you can check [**HERE**](https://github.com/udacity/CarND-PID-Control-Project). As the final results, I set `0.16, 0.002, 20.0` to `Kp, Ki, Kd` respectivily.

---

## Reflections

### Describe the effect each of the P, I, D components had in your implementation.

* P component is a simple and straightful way to control the car. It considers the current error, cross track error here. P component multiplies the error and a coefficient, Kp, as the output to control the vehicle. As Kp increased, the output is getting large. But it will cause the control unstable once Kp is too large. On the other hand, the control is getting insensitive when Kp is decreased. And it takes longer time to respond the error. And one should be noticed is that P component can not eliminate the error, which will result in a `steady-state error`. P compnent keeps respond to the error so that it causes the vehicle oscillated.

* I component considers the error from the past to present. To calculate the output, the error would be accumulated from the beginnings and then multiplied by a coefficient, Ki. The purpose of I component is to eliminate the `steady-state error` mentioned in P component. Increasing Ki can help to decrease responding time. But it might cause overshoot if Ki is too large.

* D component considers the error in the "future". It multiplies derivate of the error and a coefficient, Kd, which can respond to the change of the error. And therefore it is also called predictable controller. Similar to other components, the larger Kd, the faster it can respond to the change of error. Our porject is to control the vehicle and there are lots of short term change like left/right turning, D componet is useful to help us to control the vehicle.

### Describe how the final hyperparameters were chosen.

According to the conecpts of each component, I firstly tuned Kp with Ki and Kd were `0` to check out the approrpiate magnitude. The initial value I uesd for Kp was `0.5` and added slight change (plus minus `0.02`) to tune it. As the primary result, the vehicle was driven oscillatedly and it always failed passing the big turning. But oscillation was in a acceptable range. 

Secondly, I added I component before D componet, which I expected it can eliminate the error to make the vehicle can complete the track. However, the change of the error was quite rapid so that I component didn't help too much. So, I decreased Ki to a small value, `0.001`, and began tuning Kp and Kd by adding the small value to Kp and lager value to Kd alternatively. And finally the vehicle could complete the whole track. And I also noticed that vehcile was running centrally without tuning Ki too much.

As the final step, I slightly tuned Kp, Ki and Kd to try to make the vehicle run more smoothly. And the final values I used are `0.16, 0.002, 20.0` for `Kp, Ki, Kd` respectivily.

### Thoughts

I have studied process controls for the unit operations when I was studying chemical engineer in the university. Althohg the knowledge domain is different from self-driving car, the concept of the controllers is similar. Implementing PID controller is quite simple here. But tuning Kp, Ki and Kd took much time from me, espeically for tuning Kp at the beginninng.

By this project, I realized how to apply PID controllers on self-driving car. With this very simple case, it was not so hard as what I thought. But I wonder how to combine the sensor fusion, localization and controller together to run a car on a real road.

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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.13, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.13 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.13.0.zip).
  * If you run OSX and have homebrew installed you can just run the ./install-mac.sh script to install this
* Simulator. You can download these from the [project intro page](https://github.com/udacity/CarND-PID-Control-Project/releases) in the classroom.

---

## Basic Build Instructions

### Controller
1. Clone this repo.
2. run bulid.sh with command, `./build.sh`
3. Go to the built directory, `cd build` and Run `./pid`

### Simulator
1. Download the simulator mentioned above and unzip it.
2. run `PID Project Mac desktop Universal`.
3. select the `screen resolution` and `graphics quality` and click `Play!` button.
4. select the **left** track and click `AUTONOMOUS` button.

---

## Call for IDE Profile

Please check [ide profile instruction](ide_profiles/README.md)

### Editor Settings

The following settings are suggested to make the source codes in consistent.

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)
