# DockingModels

Loading checkpoints for the models
First, authenticate for HuggingFace.
Secondly, if you want to modify the sampling parameters, you can change them through config = CustomConfig(), otherwise, the default config will be used.
```
from DockingModels import EquivariantElucidatedDiffusion, CustomConfig

config = CustomConfig()
model = EquivariantElucidatedDiffusion.from_pretrained('stair-lab/docking_model', subfolder="ckpts")
```
