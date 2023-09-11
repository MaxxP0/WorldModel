import torch
from masked_model import GPT
import torch.nn.functional as F

PATH1 = r'masked_gptMc.pt'

VOCAB_SIZE = 831
gpt = GPT(512,VOCAB_SIZE)
gpt = torch.load(PATH1).cuda().eval()


my_list = torch.tensor([831] * 512)[None,:]
pred = gpt.generate(my_list,temperature=.8,topk_filter_thres=.5).view((8,8,8))

import matplotlib.pyplot as plt
x = pred.transpose(0,2).transpose(0,1)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('auto')
ax.voxels(x, edgecolor="k")
plt.show()

# THIS CODE PLACES THE PREDICTED BLOCKS IN YOUR MINECRAFT WORLD
import mcpi.minecraft as minecraft
mc = minecraft.Minecraft.create()
def place_voxel_blocks_in_minecraft(voxel):
    start_x,start_y,start_z = mc.player.getPos()
    for z in range(8):
        for y in range(8):
            for x in range(8):
                block_info = voxel[y, x, z]
                
                mc.setBlock(start_x + x, start_y + y, start_z + z, int(block_info))


place_voxel_blocks_in_minecraft(pred)

