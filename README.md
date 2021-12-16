![poorly drawn rain cloud icon](https://raw.githubusercontent.com/TheIncgi/Cuda-Weather/master/src/app/storm.png?token=AHG3CNBVK33XY24W2T7DHKK765ZL4 "Poorly drawn rain cloud icon")
# Cuda-Weather
A CUDA powered weather simultation attempt with Spring.

# Status
Currently under development. Simulation is not yet at a useable state.

# Index
[1.0 About this project](#section-1.0)\
  [1.1 Some Requirments](#section-1.1)\
  [1.2 Setup](#section-1.2)\
[2.0 JavaFX Interface](#section-2.0)\
  [2.1 How to use](#section-2.1)\
[3.0 Spring API](#section-3.0)\
[4.0 Development Notes](#section-4.0)\
[5.0 Some fun facts](#section-5.0)\

<div id="section-1.0"></div>

# 1.0 About this project
  This project is a started out as an idea for a larger project. I wanted to create a game where
the world could have a detailed enviroment. I had initialy tried to draw up a map of the world but found it lacking..
I wanted a world procedurally generated world where the weather was influenced by the terrain and the biomes influenced by the weather so I
could acheive something that felt belivable. It also gave me a good opportunity to learn more about GPU computing (in this case with CUDA).

<div id="section-1.1"></div>

## 1.1 Some Requirments

 - A decent amount of RAM, I'm using 16GB of ram to test with medium worlds
 - A decent **NVIDA GPU** since this uses CUDA, I've been testing with a GeForce GTX 1050 Ti. (again, medium sized world)
 - Built on [CUDA V11.1.105](https://developer.nvidia.com/cuda-11.1.1-download-archive)
 - Java 17 or newer (Built on Open JDK 17)
 
 <div id="section-1.2"></div>
 
## 1.2 Setup

 - **The application is not yet finallized, steps may change**
 - Required jars are in JCuda-All-10.1.0
 - Import as **gradle** project
 - Build.bat can be used to quickly rebuild `.ptx` files if developing
 - Launch from FXViewer or SpringAPI class

<div id="section-2.0"></div>

# 2.0 JavaFX Interface

The JavaFX interface was mainly designed to be able to visually check that things were working correctly and provide some manual controls
over the simulation.
At the time of writing this, rendering is currently done with a series of (JavaFX )Rectangles and labels, this method seems to use up a significant amount of memory at
larger scales and I plan to switch to GPU rendering in a future itteration.

<div id="section-2.1"></div>

## 2.1 How to use

1. Launch the application from the `FxViewer` class
2. Choose world size. Hovering over a button will tell you how much memory is needed to store the world (this does not included memory used for rendering currently)
   * Known issue: Worlds bigger than medium tend to crash from `Out of Memory` errors
   * Known issue: After running tests with medium worlds multiple times they may later cause `Out of Memory` errors from the JVM, this seems to be due to memory fragmentation.
                  A workaround for that is a reboot. Memory is reused during the simulation and should remain stable after launching.
3. Wait for the terrain to generate. *Currently the world seed isn't customizable via GUI (planned feature)
4. You can hit the `Step` button to move the simulation forward in time. After the first step the atmospheric conditions will have been initalized.
   **Note:** The slowest part of each timestep is retreiving the data from the GPU, doing multiple timesteps at once will be much faster.
5. Hovering over any tile will show you the current weather conditions there. If the current altitude is below ground level then the state will be based on 
   the surface above (everything below the surface is a copy).
   The time is based on 0° longitude. The first number is the number of **rotations** the planet has done. The second number is the number ofo **revolutions** completed.
   *Note: In this simulation the planet rotate and revolve in the same direction (like the earth).
6. The toolbar on the top can be used to adjust the zoom and enable overlays for things like temperature, pressure, percipitation and others.
7. The 'Regen Terrain' and 'Test' buttons are development tools.
   Regen terrain will reload the terrain generation `.ptx` and replace the current map. (atmospheric conditions are also reset)
   
<div id="section-3.0"></div>

# 3.0 Spring API

To use the Spring API the simulation should instead be launched from the `SpringAPI` class (app.spring.SpringAPI).
The simulation will automaticly start and autosave every hour. Timesteps are completed in batches and are kept as close to real time as possible.

**API is not yet finallized. **

<div id="section-4.0"></div>

# 4.0 Development notes

 - The world is generated based on a modified version of perlin noise. The modification allows the generated noise to wrap east and west correctly while still allowing for offsets.
 - Formulas may not be exact in many cases as I've had to cobble together decent amounts of information from many sources.
 - There are Desmos graphs for a couple of intersesting formulas:
   - [https://www.desmos.com/calculator/5ziykdrgdq](Air density from humidity, pressure, and temp
   - [https://www.desmos.com/calculator/jut6bbsuw7](Temp change based on specific heat, watts, mass, time ellapsed, and current temp)
   - [https://www.desmos.com/calculator/gzqcdksdhs](Max water held at different temperatures), this formula is the best aproximation I could make and is based on a bell curve.
 - Lakes are generated when a body of water of a certain size is surrounded by land. The idea is that some lakes should dry up over time and create vallies where rainfall is limited.
   When the initally generate they are quite large...
 - The effect from infrared cooling still seems to be very out of balance right now. (12/31/2020)
 - Data is only loaded to the GPU on startup, after that the data is kept in the GPU and only has to be sent back to the host after a batch of time steps. This allows a sequence of steps to happen quickly.

<div id="section-5.0"></div>

 # 5.0 Fun Facts

 - At 70F a cubic meter of air can carry about 18 grams of water
 - A cubic meter of dry air weighs about 1.3 kg
 - A cubic km of cloud (at 100% relitive humidity and) at 70F can weight about 18,089,600kg **however** the upper atmosphere tends to be cold (say -15/20F) and instead ends up weighing about
   500,000kg (about 551 tons)
   If the upper atmosphere was warmer we could end up with huge amounts of rain. However sunlight doesn't interact much with the air and will pass through most of it so most of the heat is from the ground.
   The upper atmosphere can then cool off via infrared cooling while the planets surface stays toasty (the surface technically cools too of course).
