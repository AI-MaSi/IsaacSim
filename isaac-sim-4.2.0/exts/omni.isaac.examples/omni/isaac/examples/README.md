### Usage
Drop these folders to:
```plaintext
IsaacSim/isaac-sim-4.2.0/exts/omni.isaac.examples/omni/isaac/examples/
```

Then run the demo from the IsaacSim:

![isaac4 2 0 img](https://github.com/user-attachments/assets/2bfbd094-4111-49ad-b276-d80bcd16ed05)


*Keep in mind that you need to have the models loaded as well! Script defaults asset folder to:
```plaintext
userprofile/Documents/GitHub/IsaacSim/IsaacSim Models/
```



### Nice little tip
You can add your own folders to *config/extension.toml*:
```plaintext
...
[[python.module]]
name = "omni.isaac.examples.[folder name]"
```
I find this much easier to handle multiple scripts than just putting everything under *user_examples*.

