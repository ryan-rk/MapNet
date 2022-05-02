# An Adaptive and Scalable ANN-based Model-Order-Reduction Method for Large-Scale TO Designs

The project files are associated with the research project on the development of MapNet for accelerating the large-scale Topology Optimization designs. The corresponding research paper can be found at: https://arxiv.org/abs/2203.10515


# MapNet

As discussed in the scientific paper, a ANN-based model, MapNet is being developed to perform super-resolution on the mechanical field of structure, specifically as demonstrated on strain energy field. In the structural design, iterative optimization process is required to be performed so that given a specific boundary condition, constraints, and the desired objective, the design domain (or structural design) can be optimized such that some desired mechanical properties can be maximized or minimized (for example, maximizing the mechanical strength of the design by utilizing as few amount of materials as possible). However, to perform the optimization process, objective function which is usually based on the calculation of mechanical field by numerical simulation (eg: stress field or strain energy field) is required to be evaluated at each iteration, this process is computationally expensive. Therefore, a novel approach is proposed, in which at each iteration, the numerical simulation is first performed at a coarser mesh (lower resolution). Then a Neural Network known as MapNet, which is built based on the concept of ResNet and U-Net, is used to upscale the simulation result from the coarse mesh back to the original fine mesh (high resolution). The detailed process and implementation however is more complicated as different techniques are utilized to further improve the performance of the model. These processes and methods are discussed in details in the scientific paper mentioned above.

## Architecture

The architecture of MapNet with an illustration indicating the type of inputs and outputs associated with the model are shown in the images below. Again thse are just simplified illustration, for full details on the implementation kindly refer to the scientific paper.

![Image on architecture of MapNet](/images/Fig1.png)
![Image illustrating inputs and outputs of MapNet](/images/Fig2.png)


## Results
The results on the performance and efficiency of the developed model are discussed in detailed in the journal paper.


## Project Files

The provided codes includes:
1. Architecture for MapNet written in Tensorflow 2 library.
2. The Topology Optimization code (BESO) with MapNet integrated (for L-shaped beam case).
3. Various Python modules and utils written to perform some of the procedures detailed in the scientific paper (eg: Fragmentation process).

Due to the upload limitation, the dataset used for the training and testing of the network model are not included. However they can be easily generated by performing Topology Optimization process on the design problem discussed in the paper.

## Future Works
From the discussions in the scientific paper, it is explained that the design methodology is not only limited to the application in structural design. Any design or optimization process which requires numerical simulations or calculations (which is computationally expensive) can benefit from this methodology. Therefore, with further development and improvement on the method, it can possibly be applied to other field such as fluid dynamic simulation (CFD), architectural design (Civil) and even for solving PDE (Math).
