# Is Dale’s law computationally beneficial?
## Michael Seay – Project Proposal – CCNSS 2019

Dale’s law states that each of a neuron’s synaptic terminals releases the same set of neurotransmitters<sup>1</sup>. In most cases, Dale’s law means that each neuron is strictly excitatory or inhibitory, and it cannot have excitatory effects at one terminal and inhibitory effects at another.

However, it remains unknown why Dale’s law has evolved to be a fundamental property of nervous systems. A simple biologically-based argument can be formed based on constraints provided by limited energy resources: considering most axons are extensively branched into terminal arbors, often producing hundreds or thousands of presynaptic terminals, regulating the neurotransmitter type available for specific terminals would presumably incur considerable energetic and organizational costs.

Beyond energy efficiency arguments, a question remains as to whether Dale’s law may be beneficial on a computational level. Previous work has shown that computational models of nervous systems that obey Dale’s law can take advantage of certain dynamically stable modes that are not available to those that disobey Dale’s law, for example supporting contour enhancement in vision<sup>2</sup>. Other work suggests that adhering to Dale’s law decorrelates excitatory population activity, which can be beneficial for certain computations<sup>3, 4</sup>. However, these results are at odds with other findings that trained image recognition networks unconstrained by Dale’s law have higher accuracy than their Dale-constrained counterparts<sup>5</sup>.

Indeed, in many biologically-inspired models of neural networks, Dale’s law is not upheld, and individual units are allowed to possess both positive and negative output weights. For example, consider a recurrent neural network (RNN) described by the equation:

τv ̇= -v+ W_rec ϕ(v)+ W_in i+ e

If the RNN follows Dale’s law, each column of W_rec will contain either exclusively nonnegative or exclusively nonpositive values, corresponding to excitatory and inhibitory units’ output weights, respectively. On the other hand, if the RNN disobeys Dale’s law, the columns of Wrec will contain mixtures of positive, negative, and zero values. We will refer to weight matrices that obey and disobey Dale’s law as W_dale and W_free, respectively.

In the current work, we propose to investigate the computational benefit of Dale’s law by training an RNN whose W_rec matrix is formulated as a competitive sum of constrained and unconstrained weight matrices:

W_rec=αW_dale+(1-α)W_free

By providing inputs i that correspond to task-related input stimuli and evaluating the network’s output as

z=W_out  ϕ(v)

we can train the network in a supervised manner to produce target outputs ztarget by modifying free parameters to minimize a loss function such as:

L= 〖∑▒〖(z〗-z_target)〗^2

Crucially, the parameter α, which represents the competition between the Dale’s law-constrained and unconstrained matrices, will also be included as a trainable parameter in addition to the input, recurrent, and output weight matrices. An advantage of this approach is that the trained solution will provide a theory-free assessment of the computational benefit of Dale’s law for the chosen task. Furthermore, we will have access to the initial, final, and the change in the values of model parameters over the course of training, which may provide additional insight.

Based on previous work<sup>5</sup>, we hypothesize that with the above formulation of the loss function and for a simple image recognition task such as the MNIST dataset, α will likely converge to a small number or zero because the unconstrained solution results in less error. A fundamental challenge of our work will thus involve identifying the conditions or tasks under which α converges to 1, indicating that the Dale’s law-constrained solution prevails through training. If time permits, we hope to explore and gain intuitions about the conditions in which Dale’s law excels by using treatments of the computational benefit of Dale’s law including spiking models<sup>3, 4</sup> and dynamical systems theory<sup>2</sup>.

At this time, further ideas for exploring the conditions under which Dale’s law-constrained solutions become computationally beneficial include initialization of the α parameter, formulation of the loss function, and the enforcement of additional biological constraints, such as differences in the membrane time constants of excitatory and inhibitory units, or type-specific connectivity.

1.	Eccles, J. C., Fatt, P. & Koketsu, K. Cholinergic and inhibitory synapses in a pathway from motor-axon collaterals to motoneurones. J Physiol 524–562 (1954).
2.	Zhaoping, L. & Dayan, P. Computational Differences between Asymmetrical and Symmetrical Networks. NIPS 11 1–9 (1999).
3.	King, P. D., Zylberberg, J. & DeWeese, M. R. Inhibitory Interneurons Decorrelate Excitatory Cells to Drive Sparse Code Formation in a Spiking Model of V1. J. Neurosci. 33, 5475–5485 (2013).
4.	Tetzlaff, T., Helias, M., Einevoll, G. T. & Diesmann, M. Decorrelation of Neural-Network Activity by Inhibitory Feedback. PLoS Comput. Biol. 8, (2012).
5.	Minni, S. et al. Understanding the functional and structural differences across excitatory and inhibitory neurons. bioRxiv (2019).

