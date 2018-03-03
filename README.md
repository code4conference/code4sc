## Code for Sentence Compression

We describe the steps in detail. If any points unclear, please contact authors by emailing code4conference@gmail.com .

### Step 1: Training a bi-directional neural language model. 
Take an example, use "BOS This is an anonymous Github EOS" to predict "This is an anonymous Github". (see RNN.py) 
    
### Step 2: Pre-training a policy network
Pre-training a sequence labeling neural network as policy network using labels yielded by the unsupervised method, Integer Linear Programming. 

### Step 3: Reinforcment Learning
Start with the pre-trained policy instead of random policy, and take pre-trained language model as reward to fine tune the policy network.  
