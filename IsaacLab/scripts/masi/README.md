# Excavator IK Test Demo

This is a quick test demo I made for figuring out how to use differential IK control with an excavator arm. It simulates different trajectories that the excavator's end effector follows.

## Running the tests

Make sure Isaac Sim is running first, then try these different trajectories:

```bash
# Run with a circular trajectory (default)
isaaclab.bat -p scripts/masi/excavator_example.py --trajectory circle

# Run with a square trajectory
--trajectory square

# Run with a figure-eight trajectory
--trajectory figure_eight

# Run with rotation trajectory
--trajectory rotate
--trajectory rotate2
```


You can try different IK controller methods by changing the `ik_method` value in the code. Currently available options:

- `"pinv"` - Moore-Penrose pseudo-inverse (currently set with k_val=6.0, crazy high)
- `"dls"` - Damped Least Squares 
- `"svd"` - Singular Value Decomposition
- `"trans"` - Jacobian transpose

Feel free to also mess with the trajectory generators.

I'll try to run these same test on real 1:14 scale excavator soon, stay tuned!

