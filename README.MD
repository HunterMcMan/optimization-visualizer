# Optimization Visualizer
This tool generates visuals pertaining to the topics of mathematical optimization, analysis, and topology. I created this to better help students like myself grasp difficult concepts where visualization can mean the difference between understanding and confusion. More coming soon...

## Setup
```
pip install -r requirements.txt
```

## Run the app
```
python app.py
```
This opens a landing page with a card for each visualizer. Press **Open** on a card to launch that tool in its own window; you can have several open at once. Each visualizer can also still be run on its own, as shown below.

## Build a Windows executable
```
pip install -r requirements-dev.txt
python build_exe.py
```
This creates `dist/OptimizationVisualizer.exe` (about 80 MB), a single file that runs without Python installed. Because the exe unpacks itself on every launch, it takes a few seconds to start. To check a build without opening any windows, run `OptimizationVisualizer.exe --self-test`; it opens every tool in the background and exits with code 0 when all of them work.

## Features
```
Epsilon-Delta Visualizer: Helps students understand what is meant by the often intimidating Epsilon-Delta definition of a limit, which remains a fundamental challenge one must overcome to firmly understand concepts in analysis.
```
Lagrange Multiplier Visualizer: 
