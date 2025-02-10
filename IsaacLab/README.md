# IsaacLab stuff
These are mainly placeholders for future development.
```masiV0.py``` is for configuring the model, and ```excavator_example.py``` is for testing the IsaacLab environment. More to come.
## Usage
Drop these files to your IsaacLab installation folder and change the model0.usd file path (inside masiv0.py) to the one you want to use.
Run the example with (venv active, use ./isaaclab.sh with linux):

```plaintext
isaaclab -p source\masi\excavator_example.py --num_envs 1
```