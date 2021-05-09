# Predictive Modeling of Belief Revision on the Spatial-Relational Domain
___
### Models implemented/adapted in the course of Johannes Mannhardt's bachelor thesis

The goal of my thesis was to explore belief revision on the spatial-relational domain, the underlying cognitive processes, and methods to predict it. The Dataset (Dataset.csv) results from an experiment conducted by Bucher et al. (2013) as part of their paper *Plausibility and visualizability in relational belief revision*.

## Models

### Baselines

- Random Model
- Transitive Closure
- MFA

### Cognitive Models
##### Simple Models
- LO-relocation
- Relocation of the object last inserted in to the mental model
- Preference for the plausible model
- First/Second Premise rejection
- Relocation of the left/right object
- Preference for the object presented left/right on the screen
##### Sophisticated Models that were adapted
- Verbal Model *(Krumnack, A., Bucher, L., Nejasmic, J., and Knauff, M. (2010). Spatial reasoning as verbal reasoning.)*
- PRISM *(Ragni, M. and Knauff, M. (2013). A theory and a computational model of spatial reasoning with preferred mental models.)*

### Machine Learning Models

- Content-based filtering (CBF)
- User-based collaborative filtering (UBCF)
- Multilayer Perceptron (MLP), implementation from *Riesterer, N., Brand, D., and Ragni, M. (2020). Predictive modeling of individual human cognition: Upper bounds and a new perspective on performance.*
- Ensemble Model

## Prerequisites

Python Libraries : CCobra, Numpy, Pandas, Torch

