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
include Framework/CMakeFiles/Framework.dir/depend.make

# Include the progress variables for this target.
include Framework/CMakeFiles/Framework.dir/progress.make

# Include the compile flags for this target's objects.
include Framework/CMakeFiles/Framework.dir/flags.make

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o: ../Framework/src/ActivationFunctions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o -c /home/bill/code/Framework/src/ActivationFunctions.cpp

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/ActivationFunctions.cpp > CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.i

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/ActivationFunctions.cpp -o CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.s

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o


Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o: ../Framework/src/Dense.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/Dense.cpp.o -c /home/bill/code/Framework/src/Dense.cpp

Framework/CMakeFiles/Framework.dir/src/Dense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/Dense.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/Dense.cpp > CMakeFiles/Framework.dir/src/Dense.cpp.i

Framework/CMakeFiles/Framework.dir/src/Dense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/Dense.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/Dense.cpp -o CMakeFiles/Framework.dir/src/Dense.cpp.s

Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o


Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o: ../Framework/src/Layers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/Layers.cpp.o -c /home/bill/code/Framework/src/Layers.cpp

Framework/CMakeFiles/Framework.dir/src/Layers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/Layers.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/Layers.cpp > CMakeFiles/Framework.dir/src/Layers.cpp.i

Framework/CMakeFiles/Framework.dir/src/Layers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/Layers.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/Layers.cpp -o CMakeFiles/Framework.dir/src/Layers.cpp.s

Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o


Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o: ../Framework/src/MLP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/MLP.cpp.o -c /home/bill/code/Framework/src/MLP.cpp

Framework/CMakeFiles/Framework.dir/src/MLP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/MLP.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/MLP.cpp > CMakeFiles/Framework.dir/src/MLP.cpp.i

Framework/CMakeFiles/Framework.dir/src/MLP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/MLP.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/MLP.cpp -o CMakeFiles/Framework.dir/src/MLP.cpp.s

Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o


Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o: ../Framework/src/Preprocessor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/Preprocessor.cpp.o -c /home/bill/code/Framework/src/Preprocessor.cpp

Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/Preprocessor.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/Preprocessor.cpp > CMakeFiles/Framework.dir/src/Preprocessor.cpp.i

Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/Preprocessor.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/Preprocessor.cpp -o CMakeFiles/Framework.dir/src/Preprocessor.cpp.s

Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o


Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o: ../Framework/src/Reader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/Reader.cpp.o -c /home/bill/code/Framework/src/Reader.cpp

Framework/CMakeFiles/Framework.dir/src/Reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/Reader.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/Reader.cpp > CMakeFiles/Framework.dir/src/Reader.cpp.i

Framework/CMakeFiles/Framework.dir/src/Reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/Reader.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/Reader.cpp -o CMakeFiles/Framework.dir/src/Reader.cpp.s

Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o


Framework/CMakeFiles/Framework.dir/src/agent.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/agent.cpp.o: ../Framework/src/agent.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object Framework/CMakeFiles/Framework.dir/src/agent.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/agent.cpp.o -c /home/bill/code/Framework/src/agent.cpp

Framework/CMakeFiles/Framework.dir/src/agent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/agent.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/agent.cpp > CMakeFiles/Framework.dir/src/agent.cpp.i

Framework/CMakeFiles/Framework.dir/src/agent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/agent.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/agent.cpp -o CMakeFiles/Framework.dir/src/agent.cpp.s

Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/agent.cpp.o


Framework/CMakeFiles/Framework.dir/src/env.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/env.cpp.o: ../Framework/src/env.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object Framework/CMakeFiles/Framework.dir/src/env.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/env.cpp.o -c /home/bill/code/Framework/src/env.cpp

Framework/CMakeFiles/Framework.dir/src/env.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/env.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/env.cpp > CMakeFiles/Framework.dir/src/env.cpp.i

Framework/CMakeFiles/Framework.dir/src/env.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/env.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/env.cpp -o CMakeFiles/Framework.dir/src/env.cpp.s

Framework/CMakeFiles/Framework.dir/src/env.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/env.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/env.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/env.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/env.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/env.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/env.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/env.cpp.o


Framework/CMakeFiles/Framework.dir/src/test.cpp.o: Framework/CMakeFiles/Framework.dir/flags.make
Framework/CMakeFiles/Framework.dir/src/test.cpp.o: ../Framework/src/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object Framework/CMakeFiles/Framework.dir/src/test.cpp.o"
	cd /home/bill/code/build/Framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Framework.dir/src/test.cpp.o -c /home/bill/code/Framework/src/test.cpp

Framework/CMakeFiles/Framework.dir/src/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Framework.dir/src/test.cpp.i"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/code/Framework/src/test.cpp > CMakeFiles/Framework.dir/src/test.cpp.i

Framework/CMakeFiles/Framework.dir/src/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Framework.dir/src/test.cpp.s"
	cd /home/bill/code/build/Framework && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/code/Framework/src/test.cpp -o CMakeFiles/Framework.dir/src/test.cpp.s

Framework/CMakeFiles/Framework.dir/src/test.cpp.o.requires:

.PHONY : Framework/CMakeFiles/Framework.dir/src/test.cpp.o.requires

Framework/CMakeFiles/Framework.dir/src/test.cpp.o.provides: Framework/CMakeFiles/Framework.dir/src/test.cpp.o.requires
	$(MAKE) -f Framework/CMakeFiles/Framework.dir/build.make Framework/CMakeFiles/Framework.dir/src/test.cpp.o.provides.build
.PHONY : Framework/CMakeFiles/Framework.dir/src/test.cpp.o.provides

Framework/CMakeFiles/Framework.dir/src/test.cpp.o.provides.build: Framework/CMakeFiles/Framework.dir/src/test.cpp.o


# Object files for target Framework
Framework_OBJECTS = \
"CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o" \
"CMakeFiles/Framework.dir/src/Dense.cpp.o" \
"CMakeFiles/Framework.dir/src/Layers.cpp.o" \
"CMakeFiles/Framework.dir/src/MLP.cpp.o" \
"CMakeFiles/Framework.dir/src/Preprocessor.cpp.o" \
"CMakeFiles/Framework.dir/src/Reader.cpp.o" \
"CMakeFiles/Framework.dir/src/agent.cpp.o" \
"CMakeFiles/Framework.dir/src/env.cpp.o" \
"CMakeFiles/Framework.dir/src/test.cpp.o"

# External object files for target Framework
Framework_EXTERNAL_OBJECTS =

../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/agent.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/env.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/src/test.cpp.o
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/build.make
../lib/libFramework.a: Framework/CMakeFiles/Framework.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bill/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library ../../lib/libFramework.a"
	cd /home/bill/code/build/Framework && $(CMAKE_COMMAND) -P CMakeFiles/Framework.dir/cmake_clean_target.cmake
	cd /home/bill/code/build/Framework && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Framework.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Framework/CMakeFiles/Framework.dir/build: ../lib/libFramework.a

.PHONY : Framework/CMakeFiles/Framework.dir/build

Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/ActivationFunctions.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/Dense.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/Layers.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/MLP.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/Preprocessor.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/Reader.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/agent.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/env.cpp.o.requires
Framework/CMakeFiles/Framework.dir/requires: Framework/CMakeFiles/Framework.dir/src/test.cpp.o.requires

.PHONY : Framework/CMakeFiles/Framework.dir/requires

Framework/CMakeFiles/Framework.dir/clean:
	cd /home/bill/code/build/Framework && $(CMAKE_COMMAND) -P CMakeFiles/Framework.dir/cmake_clean.cmake
.PHONY : Framework/CMakeFiles/Framework.dir/clean

Framework/CMakeFiles/Framework.dir/depend:
	cd /home/bill/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bill/code /home/bill/code/Framework /home/bill/code/build /home/bill/code/build/Framework /home/bill/code/build/Framework/CMakeFiles/Framework.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Framework/CMakeFiles/Framework.dir/depend
