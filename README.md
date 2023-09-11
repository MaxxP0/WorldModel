![IMG_7621](https://github.com/MaxxP0/WorldModel/assets/95175667/a44c37e5-4951-469a-ae6a-9c5e4fff6568)
![IMG_7622](https://github.com/MaxxP0/WorldModel/assets/95175667/22b11fd4-eaf2-4c50-95e3-f7e443947ea7)

# WorldModel
Note: This was a passion project, crafted in two days. I hope it can be of help to many!

WorldModel is a MaskDit model trained on 8x8x8 Minecraft voxel volumes. Beyond generating blocks from scratch,
it excels in filling spaces based on neighboring blocks, ensuring seamless integration in Minecraft worlds

## Prerequisites:
Ensure your server has the mcpi package installed. This code relies on it to place blocks.
The model weights can be found [here](https://drive.google.com/file/d/1O6i-WQ-h6H_evhJifdiqhPbPFTC3UAF1/view?usp=drive_link)

## Functionality:
This tool uses a special mask token, designated as 831. When you provide a tensor of shape [1,512] containing
block values ranging from 0-831(including the mask token), the generate function will intelligently fill in
the masked areas until none remain.


## Sampling Strategy Adjustments in maskGIT:
In contrast to the original maskGIT, sampling solely from the most confident tokens can lead to mode collapse, often
resulting in generating only stones or air. To address this, we've introduced a random sampling strategy. In the early
sampling phases, the confidence of potential tokens is disregarded, and tokens are selected for unmasking at random.

You can fine-tune this behavior using the "random_steps" variable. It's crucial to strike a balance between "random_steps"
and other parameters such as the total "timesteps", "temperature", and "topk_filter_thres".
