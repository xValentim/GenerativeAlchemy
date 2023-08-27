# Generative Models

In short, generative models belong to a class of statistical models that are capable of generating new instances based on a dataset. Typically, these models are utilized in unsupervised learning.

Formally, we can think of a dataset as a sequence of instances $x_1, x_2, ..., x_n$ from a data probability distribution $p(x)$. In the example image below, the blue region represents the part of the image space that contains real images with a high probability (above some threshold), while the black dots indicate our data points (each representing an image in our dataset).

<p align="center">
    <img src="./images/gm.png" alt="Example Image" style="width:600px;">
</p>

We can define our generative model as a probability distribution with $\theta$ parameters, denoted as $\hat p_{\theta}(x)$. It's important to note that we define this distribution using points from a Gaussian unit. Therefore, we need to find $\theta$ parameters that satisfy our requirements: generating new data $x'$ consistently.

In this context, we will be creating various generative models and exploring both experimentation and theory. Let's begin by discussing the first architecture that we will be constructing.

## Autoencoders

This architecture is very important to us, because autoencoders are a base for some popular generative models. Autoencoders have power to compress data and reconstruct data. Basicaly, this is very interesting because, first application can be compress data and you can spend less memory for any application (yay!). But autoencoder can be much more! Let's check this out and see architecture below:

<p align="center">
    <img src="./images/autoencoder_architecture.png" alt="Example Image" style="width:600px;">
</p>

In here, we can see architecture and loss function that we must to use. This loss is called by "Reconstruction Loss", because when system minimizes this, we can to turn output aproximates by input. 

$$\text{Reconstruction Loss} = \text{MSE}(x, \hat x)$$

Before we in fact construct some generative architecture, lets build an autoencoder and check some theory subjects! Spoiler: You will learn about **latent space** (Latent space is a little bit abstact, think about him like a something in Matrix)

### Visualizing Reconstruction

After training, we can see reconstruction loss and visualize some images. In this case, we will use MNIST dataset. Let's check this out!

<p align="center">
    <img src="./images/reconstruction_ae.png" alt="Example Image" style="width:600px;">
</p>

For now, it's not very interesting. But, we can see some interesting things in latent space. Let's check this out!

### Visualizing latent space

Let's analyze this architecture. If you've noticed, the $z$ variable in the middle of the autoencoder are the input compressed data. Here, we can identify two significant aspects: data compression and dimensionality reduction (similar to PCA). 

<p align="center">
    <img src="./images/generative/mnist_visualization_plot_ae.png" alt="Example Image" style="width:600px;">
</p>

Look at this! In the **latent space** some semantic properties is conserved. Lets think about this using some cases:

- If you analyzes the number 3 cluster, you will note him closer to 8. Its happen because *3* shape is very similar to *8* if you slice him at the vertical axis. 
- The same ideia can be apply to *4* shape and *9* shape. Note both cluster are closer!

Some semantic in this case are very similar to embeddings semantics: $$f(Queen) - f(King) = f(Girl) - f(boy)$$

## Convolutional Autoencoders

We can apply convolutional layers in autoencoders. This is very interesting because we can use this architecture in images. Let's check this out!

<p align="center">
    <img src="./images/conv_ae.png" alt="Example Image" style="width:600px;">
</p>

### Visualizing Reconstruction

<p align="center">
    <img src="./images/reconstruction_conv_ae.png" alt="Example Image" style="width:600px;">
</p>

### Visualizing latent space

<p align="center">
    <img src="./images/generative/mnist_visualization_plot_conv_ae.png" alt="Example Image" style="width:600px;">
</p>

Note the improve of quality! But we need to take care about the size of neural network. In many cases of Neural Network is pretty complex the system learn indentity function, and its not cool! Because we want Neural network learns deep patterns about dataset and not inditity functions!


## Divergence KL

In here, we will discuss about Kullback-Leibler Divergence. This equation works like a regularization and your principal role is aproximates distributions. In this case, we will use the following dicrete form:

$$D_{KL}(P \parallel Q) = \sum_{i} P(i) \cdot \log\left(\frac{P(i)}{Q(i)}\right)$$

After some math, we can compute to variational encoder and use in this form:

$$D_{KL}(Q(z|X) \parallel P(z)) = \frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

## Variational Autoencoders

A Variational Autoencoder (VAE) is a type of neural network architecture used for generative tasks, such as creating new data samples. It combines elements of both autoencoders and probabilistic modeling. In a VAE, the encoder part transforms input data into a compact representation in a lower-dimensional space, while the decoder part converts this representation back into the original data domain. The "variational" aspect comes from the inclusion of probabilistic elements that allow the model to capture the underlying distribution of the data.

<p align="center">
    <img src="./images/vae.png" alt="Example Image" style="width:600px;">
</p>

The loss function of a VAE is composed of two parts: the reconstruction loss and the KL divergence. The reconstruction loss is the same as the one used in the autoencoder, and it measures how well the model is able to reconstruct the input data. The KL divergence was explained before and it just will approximate distribution in latent space. The goal of the VAE is to minimize both of these losses simultaneously.

### Extra - Classification and Reconstruction (Multi-stage VAE)

Until here, you see autoencoders applications with only reconstruction on output. But, we can introduce another decoder and make classification together reconstruction. It will lead us to a new concept application with autoencoder: Multi-stage autoencoder. It will be explicity in a figure below:

<div style="text-align:center;">
    <img src="./images/multi_stage_autoencoder.png" alt="Example Image" style="width:600px;">
</div>

#### Latent space effects of Multi-stage VAE

Let's see effects of multi-stage VAE in latent space. We will use MNIST dataset and check this out!

<div style="text-align:center;">
    <img src="./images/latent_effects.png" alt="Example Image" style="width:600px;">
</div>

Each color represents a different class. We can see that the latent space is more organized and the classes are more separated. However, when we add classification, we can see more separatability between classes. This is very interesting because we can use this separation to make transitions between two classes without pass into another class. 

Finally, when we increase $\beta$, we can see more approximation, beacause we increase the weight of KL divergence in loss function. Let's check this out!

$\mathcal{L} = MSE(x, \hat x) + \beta \cdot D_{KL}(q, p) - y\log(\hat y)-(1 - y)\log(1-\hat y)$


## Transition between two classes

We now have all the interpolated images between $0 \rightarrow 1$, $1 \rightarrow 2$, $2 \rightarrow 3$, and so on. Remember: We calculate centroids for each class and formulate the equation of a straight line,

$$\vec r(d) = \vec o + d \hat t$$

In the first transition, we use $\vec{o} = \vec{c}(0)$ and $\hat{t} = \frac{\vec{c}(1) - \vec{c}(0)}{||\vec{c}(1) - \vec{c}(0)||_2}$, where $\vec{c}(x)$ is the centroid of the number $x$ in the latent space. We simply march along the equation of a straight line, pick up some points on the line, put them into the decoder, and voilà! You can see this technique in the image below:

<div style="text-align:center;">
    <img src="images/z_space_to_frames.png" alt="Example Image" style="width:720px;">
</div>
Then, using the idea explained before, we can construct a gif with connections between each frame. We will witness the smooth transition between classes in our dataset. See the gif below:

<div style="text-align:center;">
    <img src="output_gif_regular.gif" alt="Example Image" style="width:100px;">
</div>
This gif shows us a beautiful animation that interpolates images that are not in the original dataset.

## Quality Transition

Another thing we need to discuss about multi-stage VAE is: There is difference in transition between two architecture?

Let's see and compare!


<div style="text-align:center;">
    <img src="output_gif_regular.gif" alt="Example Image" style="width:100px;">
</div>


<div style="text-align:center;">
    <img src="output_gif_multi_stage.gif" alt="Example Image" style="width:100px;">
</div>

## Conclusions (Comparing two architectures):

- There are many differents between two architetures. We reach the objective of make transitions two classes without pass into another, with multi-stage VAE we can see that. However, the image quality was decrease, probabily because we increase too much $\beta$ in KL divergence.

- We need to take care about $\beta$ in KL divergence. For balance, we can introduce others hyperparameters in loss function in multi-stage case. Suggestions: start with low value hyperparameters to classification term.

- Another observed effect is a little bit less capability of create new instances with diversity under our dataset.

## Future Work

- In the future, we will see more application using multi-stage VAE and explore your power.

- We will see more about GANs and your applications.

- We will see more about VAE-GANs and your applications.

- We will see more about arithmetics in latent space and your applications and input features with face dataset (Deep Fake).
