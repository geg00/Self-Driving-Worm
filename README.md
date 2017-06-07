# Self-Driving-Worm

This project was inspired by https://github.com/Sentdex/pygta5 and by https://github.com/llSourcell/How_to_simulate_a_self_driving_car

Using a convolutional neural network 

# Dependencies

You can install all dependencies by running one of the following commands

You need a anaconda or miniconda to use the environment setting.

```python
conda env create -f environments.yml 
```

Start chrome on the top corner of the screen with a size of 800x600 go to http://slither.io

Start a command prompt with Anaconda

```python
activate cars
python mousemoves.py
```

Start moving the mouse around the screen to direct the worm.
Every 500 moves a new set of data will be saved to the numpy file slither_data.npy
Gather 6000 to 8000 moves.


