Add cnpy (https://github.com/rogersce/cnpy) repo to subprojects directory and run:
```
meson setup build
meson compile -C build
./build/my_app -d "images/" -w 9 -h 6 -s 25 -o "test.npz"
```
with images of the 9x6 calibration checkerboard in the "images" directory. The size (in mm) of each square is defined by -s param. Output goes to test.npz.
