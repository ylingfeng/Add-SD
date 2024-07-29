# Stage3. Generating synthetic data for downstream tasks

This stage generates synthetic data for downstream data augmentation. Annotations are update with 
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

## Pipeline
<p align="center">
  <img src="../assets/super-class.svg" alt="/super-class.svg" style="width: 100%;"/>
  <br>
  <span style="display: block; text-align: center; font-size: 14px; color: #555;">We design a super-label-based sampling strategy to restrict the category of the added object, ensuring rationality. Then, we randomly sample a sub-label within the super-label, assigning a higher weight to tail-class labels to alleviate the long-tail problem. After image generation, the annotations are inherited from the vanilla dataset <font color=LawnGreen>green</font> for the original instance and grounded <font color=red>red</font> for the added instance.</span>
</p>


## Results

<p align="center">
  <img src="../assets/stage3.svg" alt="stage3.svg" style="width: 100%;"/>
  <br>
  <span style="display: block; text-align: center; font-size: 14px; color: #555;">Grounding results of synthetic data.</span>
</p>
