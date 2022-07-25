# MST-VAE
### Multi-Scale Temporal Variational Autoencoder for Anomaly Detection in Multivariate Time Series

MST-VAE is an unsupervised learning approach for anomaly detection in multivariate time series. Inspired by InterFusion paper, we propose a simple yet effective multi-scale convolution kernels applied in Variational Autoencoder. 
Main techniques in this paper:
- Multi-scale module: short-scale and long-scale module
- We adopted Beta-VAE for training the model
- MCMC is applied to achieve better representations while detecting anomalies

## How to use the repository
### Clone the repository
<pre><code>git clone https://github.com/tuananhphamds/MST-VAE.git && cd MST-VAE
</code></pre>
