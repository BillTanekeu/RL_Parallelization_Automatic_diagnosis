# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bill/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bill/code/build

# Include any dependencies generated for this target.
include app/CMakeFiles/trainer.dir/depend.make

# Include the progress variables for this target.
include app/CMakeFiles/trainer.dir/progress.make

# Include the compile flags for this target's objects.
include app/CMakeFiles/trainer.dir/flags.make

app/CMakeFiles/trainer.dir/src/trainer.cpp.o: app/CMakeFiles/trainer.dir/flags.make
app/CMakeFiles/trainer.dir/src/trainer.cpp.o: ../app/src/trainer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object app/CMakeFiles/trainer.dir/src/trainer.cpp.o"
	cd /home/bill/code/build/app && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trainer.dir/src/trainer.cpp.o -c /home/bill/code/app/src/trainer.cpp

app/CMakeFiles/trainer.dir/src/trainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trainer.dir/src/trainer.cpp.i"
	cd /home/bill/code/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/app/src/trainer.cpp > CMakeFiles/trainer.dir/src/trainer.cpp.i

app/CMakeFiles/trainer.dir/src/trainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trainer.dir/src/trainer.cpp.s"
	cd /home/bill/code/build/app && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/app/src/trainer.cpp -o CMakeFiles/trainer.dir/src/trainer.cpp.s

app/CMakeFiles/trainer.dir/src/trainer.cpp.o.requires:

.PHONY : app/CMakeFiles/trainer.dir/src/trainer.cpp.o.requires

app/CMakeFiles/trainer.dir/src/trainer.cpp.o.provides: app/CMakeFiles/trainer.dir/src/trainer.cpp.o.requires
	$(MAKE) -f app/CMakeFiles/trainer.dir/build.make app/CMakeFiles/trainer.dir/src/trainer.cpp.o.provides.build
.PHONY : app/CMakeFiles/trainer.dir/src/trainer.cpp.o.provides

app/CMakeFiles/trainer.dir/src/trainer.cpp.o.provides.build: app/CMakeFiles/trainer.dir/src/trainer.cpp.o


# Object files for target trainer
trainer_OBJECTS = \
"CMakeFiles/trainer.dir/src/trainer.cpp.o"

# External object files for target trainer
trainer_EXTERNAL_OBJECTS =

app/trainer: app/CMakeFiles/trainer.dir/src/trainer.cpp.o
app/trainer: app/CMakeFiles/trainer.dir/build.make
app/trainer: ../lib/libFramework.a
app/trainer: app/CMakeFiles/trainer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable trainer"
	cd /home/bill/code/build/app && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trainer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
app/CMakeFiles/trainer.dir/build: app/trainer

.PHONY : app/CMakeFiles/trainer.dir/build

app/CMakeFiles/trainer.dir/requires: app/CMakeFiles/trainer.dir/src/trainer.cpp.o.requires

.PHONY : app/CMakeFiles/trainer.dir/requires

app/CMakeFiles/trainer.dir/clean:
	cd /home/bill/code/build/app && $(CMAKE_COMMAND) -P CMakeFiles/trainer.dir/cmake_clean.cmake
.PHONY : app/CMakeFiles/trainer.dir/clean

app/CMakeFiles/trainer.dir/depend:
	cd /home/bill/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bill/code /home/bill/code/app /home/bill/code/build /home/bill/code/build/app /home/bill/code/build/app/CMakeFiles/trainer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : app/CMakeFiles/trainer.dir/depend

