# Conditional Flow Matching
This repo explores various applications of Flow Matching. 

### Original state
Originally the repo was meant to give intuition for how Flow Matching models can be implemented in a simple way. The repo has moved on from this, but an implementation of a simple conditional flow matching objective can still be found at the branch `simple_flow_matching`

I also wrote an article explaining how Flow Matching works: https://medium.com/@uisdahl/understanding-flow-matching-de2f706cb09d 
<br>

### Resources 
- Original Flow Matching paper: https://arxiv.org/abs/2210.02747 
- Mini-batch Optimal Transport Flow Matching paper: https://arxiv.org/abs/2302.00482  
- VQ-VAE paper: https://arxiv.org/abs/1711.00937 
<br>

### Sample results for simple Flow Matching objective

Generating flows that take samples from a standard gaussian to the flower-102 distribution

![flowers](/static/Flowers.png)

The learned flows create decent samples after around 20 epochs, but there is some collapse in the distribution of colors
