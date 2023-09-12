![IMG_7621](https://github.com/MaxxP0/WorldModel/assets/95175667/525c9f61-7e29-4cbf-8bbd-30421a10f763)
![IMG_7622](https://github.com/MaxxP0/WorldModel/assets/95175667/58954cb3-78d8-4c32-a0d8-abaad2df0455)


# WorldModel
Note: This was a passion project, crafted in two days. I hope it can be of help to many!

WorldModel is a MaskGIT model trained on 8x8x8 Minecraft voxel volumes. Beyond generating blocks from scratch,
it excels in filling spaces based on neighboring blocks, ensuring seamless integration in Minecraft worlds

The model weights can be found [here](https://drive.google.com/file/d/1--4Z5VQ-mRQz805yDmJPsPKrIrVCaoCv/view?usp=sharing)

## Prerequisites:
Ensure your server has the mcpi package installed. This code relies on it to place blocks.

## Functionality:
This tool uses a special mask token, designated as 831. When you provide a tensor of shape [1,512] containing
block values ranging from 0-831(including the mask token), the generate function will intelligently fill in
the masked areas until none remain.


## Sampling Strategy Adjustments to maskGIT:
In contrast to the original maskGIT, sampling solely from the most confident tokens can lead to mode collapse, often
resulting in generating only stones or air. To address this, I've introduced a random sampling strategy. In the early
sampling phases, the confidence of potential tokens is disregarded, and tokens are selected for unmasking at random.

You can fine-tune this behavior using the "random_steps" variable. It's crucial to strike a balance between "random_steps"
and other parameters such as the total "timesteps", "temperature", and "topk_filter_thres".
