# framework-orbit

# Overview
The aim of this architecture is for motor action to arise through the reconciliation of external sensory input with a prior, without a scalar reward signal or a task-specific global loss. Much of it is based on the philosophy of predictive coding (Rao & Ballard, 1999) and active inference (Friston et al., 2010).

# Main architecture
The primary network, which we will call $O_w$, is in the form of a directed graph. Let $d$ be the internal dimension of the network. Each node $X$ of the graph maintains an internal state $X_h$ ($d \times 1$), a force vector $X_f$ ($d \times 1$), and a degree, $X_{\deg}$. Each edge $E$ of the graph maintains a transformation matrix $E_W$ ($d \times d$), an error vector $E_e$ ($d \times 1$), and two vector exponential moving averages $E_u$ and $E_v$ of $e \odot e$ and $e$ respectively. We note that some nodes may be fixed (i.e. $X_h$ is externally set and does not update). Initially, $E_W\approx I$, with small Gaussian noise perturbations.

$O_w$ also has 4 scalar hyperparameters:

- A fast learning rate $\alpha$
- A slow learning rate $\beta$
- An epsilon value $\varepsilon$
- A tanh scaling value $\tau$

There are 4 additional parameters for action exploration:

- A noise magnitude $\phi_{\mathrm{mag}}$
- A noise minimum $\phi_{\min}$
- A noise maximum $\phi_{\max}$
- A decaying noise floor constant $T$

## Forward step
Call the current timestep $t$.
For every time-dependent variable $x$, let $x^*\equiv x_{t+\Delta t}$, where $x\equiv x_t$.
First, for every node $X$ in the graph, $X_f \leftarrow 0$.
Let us now go through each edge of the graph. Consider an edge $E$. Assume $E$ connects node $X$ to node $Y$.
We calculate precision $P$ as a $d \times 1$ vector, where

$$
P=\frac{1}{\max(0,E_u-E_v\odot E_v)+\varepsilon}.
$$

We note that

$$
E_u-E_v\odot E_v\approx\mathbb{E}[e\odot e]-\mathbb{E}[e]\odot\mathbb{E}[e]=\mathrm{Var}[e],
$$

thus, $P\approx 1/(\mathrm{Var}[e]+\varepsilon)$.
Letting $X_h'=\tau\tanh((E_W)(X_h)/\tau)$, we get:

- $E_e=X_h'-Y_h$
- $E_u^*=E_u+\Delta t(E_e\odot E_e-E_u)$
- $E_v^*=E_v+\Delta t(E_e-E_v)$
- Precision-weighted error $\widehat{E_e}=E_e\odot P$

We now utilize the delta rule to update $E_W$.
Define the local error signal as

$$
\delta=\widehat{E_e}\odot\left(1-\frac{X_h'\odot X_h'}{\tau^2}\right),
$$

$$
E_W^*=E_W-\beta\Delta t\,\delta X_h^\top.
$$

Additionally, forces are contributed to both nodes:

$$
X_f \leftarrow X_f-E_W^\top\delta
$$

$$
Y_f \leftarrow Y_f+\widehat{E_e}
$$

Note that all forces and weight updates are local gradient descent on precision-weighted prediction error.

Finally, apply all forces, normalized by total degree.
For every non-fixed node $X$,

$$
X_h^*=X_h+\alpha\Delta t\frac{X_f}{\max(1,X_{\deg})}.
$$

Let us call the above process a “forward step”.

## Actuator architecture
Before action can be generated, we must introduce a secondary network $O_k$, which acts as an actuator that converts environmental/sensory change into motor action. $O_k$ is a standard feedforward network with internal dimension $d$ and a scaled tanh activation function. To avoid redundancy, we will not go into too much detail regarding it.

## Action generation
Denote the environment as $Q$. To generate action, we first fix two nodes, $N_{\mathrm{env}}$ and $N_{\mathrm{prior}}$, to the environment vector and prior vector respectively (both of which have dimensions $d \times 1$). Following that we run a forward step. Interpreting $(N_{\mathrm{env}})_f$ as the change that must be applied to the environment, we calculate an action vector $a_t=O_k((N_{\mathrm{env}})_f)$.
Now we apply exploration noise.
$$
\sigma_t=\min\left(\phi_{\max},\max\left(\phi_{\min}e^{-t/T},\phi_{\mathrm{mag}}\frac{\lVert a_t\rVert_2}{\sqrt{d}}\right)\right).
$$

We sample

$$
\eta_t\sim\mathcal{N}(\mathbf{0},\sigma_t^2I_d)
$$

and set

$$
a_t \leftarrow a_t+\eta_t.
$$

This action vector $a_t$ is the motor output, which is directly applied to the environment. Now, we observe an environmental rate of change $\Delta Q_t=(Q_{t+\Delta t}-Q_t)/\Delta t$, and have an action that produced it, $a_t$. Thus, we will train $O_k(\Delta Q_t)\approx a_t$. For direct-kinematic systems, this can approximate an inverse actuator. Under systems with inertia or other temporal dynamics, it learns a relationship between sensory change and action observed during the controller’s own behaviour. An alternative method could be to train $O_k$ as a forward model (i.e. $O_k(a_t)\approx\Delta Q_t$), and then backpropagae $(N_{\mathrm{env}})_f$ through $O_k$ to obtain an action-space update direction.

## Recommended graph structure
The recommended graphs structure consists of two isomorphic subgraphs $G_{\mathrm{env}}$ and $G_{\mathrm{prior}}$ that share a source node $N_{\mathrm{source}}$. $G_{\mathrm{env}}$ terminates at $N_{\mathrm{env}}$, which is fixed to the environment vector, while $G_{\mathrm{prior}}$ terminates at $N_{\mathrm{prior}}$, which is fixed to the prior vector. The transformation matrices of corresponding edges should be tied together (while retaining separate errors and error statistics). This places the environment and prior in a shared coordinate system, allowing them to be compared at $N_{\mathrm{source}}$.

# Current results:

- In tested graph configurations, local raw and precision-weighted error energy reliably decrease, from approximately $3$ to around $0.01$-$0.02$
- Successful sequential target seeking in up to 4 dimensions in both direct-kinematics and undamped inertial experiments.
  - Specifically, in both cases: On a trial of thirty sequential uniformly random four-dimensional targets with each coordinate in the range $[-1,1]$, all of 100 initializations completed the sequence. Success at each target was measured as settling within $0.01$ of each coordinate for 200 consecutive ticks, within a total timeframe of 100,000 ticks per target. In the undamped inertial experiments, velocity was not supplied in either sensory input or the prior.
- Two-joint arm remains unsolved, although switching to the alternative forward-model action generation method mentioned above may help.
