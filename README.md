# segmentation_openvino_test

| CPU        | FPS           |
| ------------- |:-------------:|
| i5-9400        | 靠右對齊      |

## CPU: i5-9400 
## FPS: 6~7
## Demo
```
mkdir build && cd build
cmake ..
make -j8

mv ../components_20220427* .
./afford_demo
```

