# File: meson.build
project('my_app', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

# 1. Find OpenCV (System)
opencv_dep = dependency('opencv4')

# 2. Find cxxopts (System)
# If libcxxopts-dev provides a pkg-config file, this works. 
# If not, we use find_library or just declare it as it is header-only.
cxxopts_dep = dependency('cxxopts', required : false)
if not cxxopts_dep.found()
  # Fallback: just check for the header if pkg-config is missing
  cxxopts_dep = declare_dependency() 
endif

# 3. Find cnpy (Subproject)
cnpy_proj = subproject('cnpy')
cnpy_dep = cnpy_proj.get_variable('cnpy_dep')

executable('my_app',
           'calibrate.cpp',
           dependencies : [opencv_dep, cxxopts_dep, cnpy_dep],
           install : true)
