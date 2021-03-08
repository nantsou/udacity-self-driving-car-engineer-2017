## CLion IDE profile

### Requirements

* OS: MAC OS
* `cmake` which you can install with the command `brew cmake`

please check [Homebrew](https://brew.sh/) if you haven't install homebrew.

### Required folders for **Clion** IDE

* `.idea`
* `cmake-build-debug`

### Prepare the environment

`.idea` will be generated automatically when you launch **Clion**.

Follows are the steps to set up `cmake-build-debug` folder:

1. press `command` + `,` to open Preference window.
2. go to `Build, Execution, Deployment` > `CMake` setting.
3. set `Generation path`

After set `Generation path`, **Clion** will build `cmake-build-debug` folder to where you set automatically.

## Xcode IDE profile

For the setup of Xcode, please check Udacity's instruction. [**HERE**](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project/tree/master/ide_profiles/xcode)
