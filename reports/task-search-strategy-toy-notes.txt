# Unjudged rehearsal as access to an existing strategy

The current arithmetic experiments are consistent with a change in the strategy elicited by context. They do not establish that self-generation creates new knowledge. The constrained follow-up is particularly useful: Qwen can be instructed that two 1s are invalid, yet context can still amplify that pattern. A minimal toy should permit both improvement and this failure, without a correctness filter.

## A fixed learner with a small context state

Let j denote a solution strategy. For each input, strategies are different ways to attempt its answer. The task questions are identical across conditions. Strategy B is difficult for the learner to execute, but an exact teacher can supply correct B solutions. Strategy A is easier for the learner to execute. Both can express valid, input-dependent answers; neither needs to be a constant shortcut.

The learner has fixed initial logits h0 and fixed cue weights w. All initial problem–answer pairs have already been absorbed into h0. Its generated rehearsal styles are independent samples from its own initial distribution:

```
z_i ~ Categorical(softmax(h0))
```

Every generated answer is retained, correct or incorrect. A fixed feature extractor recognizes its strategy/style, without evaluating its correctness. Reading either source's samples applies the identical local update:

```
h_(i+1) = h_i + diag(w) one_hot(z_i)
P(strategy j on a held-out question | h_N) = softmax(h_N)_j
accuracy(h_N) = sum_j success_j * softmax(h_N)_j
```

Here h is an interpretable context state (or fast weights), not a changed pretrained network. The success probabilities are used only to evaluate the model; they never enter generation, feature extraction, or the update. The stochastic strategy readout summarizes variation across queries; it is not a literal model of Qwen's greedy token decoder.

The teacher supplies valid B-style answers to the same inputs. Its exact solutions can be more accurate than the learner's samples while eliciting a strategy the learner executes poorly on new inputs. The deliberately restricted reader uses style cues rather than extracting all the mathematical information in a worked example. This limitation, and the assumed cue weights, are the reason an advantage is possible.

For two strategies ordered (B,A), the illustrative parameters are h0=(0,-1), w=(1,4), and success=(.25,.9). Own generation chooses A with probability .269. With N rehearsals, the A count is binomial, C~Binomial(N,.269), and the logit difference is exactly

```
h_A - h_B = -1 + 5*C - N.
```

We sum the binomial probabilities exactly to obtain the learning curve. The rare A cue has a stronger effect on recognition, so its positive expected drift overcomes B. This asymmetry is an explicit hypothesis about the reader, not a theorem about self-generation. If all cue weights are equal, the same learner concentrates on the initially more common B strategy instead. A teacher providing correct A-style solutions performs at least as well in this numerical example. Relabeling a saved own context as teacher-provided gives exactly the same prediction distribution.

More generally, the own-context winner at large N is the strategy maximizing w_j*p_j, where p=softmax(h0), when the maximum is unique. In the two-strategy example the mean logit increments are .731 for B and 1.076 for A. With equal cue weights they are .731 and .269, so B wins. Improvement therefore requires the fixed generation and recognition tendencies to align with a useful strategy; the update cannot discover that usefulness from correctness feedback it never receives.

A three-strategy extension adds C, a salient but invalid strategy. We use h0=(0,-1,-3), w=(1,4,45), and success=(.25,.9,0). Rare C examples can eventually dominate the context. Exact multinomial averaging produces early gains followed by deterioration. Changing only C's validity from 0 to 1 changes the scoring curve while leaving all contexts and predictions fixed. This is a diagnostic illustration, not a simulation of the actual instruction intervention, which also changed Qwen's generation distribution.

All toy parameters are illustrative and unfitted. The simulation records every parameter and exact curve under `results/task_search/strategy_context_toy`; its two PNGs appear in the HTML report. No experimental outcomes were used for inclusion, filtering, or adaptive parameter fitting. The toy explains a possible content effect and supplies controls; it has not identified Qwen's internal circuit.

## A normalized probabilistic interpretation of the reader

The cue weights can also be written as ordinary likelihood ratios. For positive w_j, define u_j=1/(exp(w_j)-1), c=1/(1+sum_j u_j), a_j=c*u_j, and

```
L(cue=j | latent strategy=k) = a_j + c * 1[j=k].
```

Every column sums to one. Bayes' rule using this likelihood gives exactly h <- h + w_j*one_hot(j), since the other terms are common to all candidate strategies. Thus the update can represent a reader inferring an external shared strategy from diagnostic cues. It still does not infer correctness.

The generative/recognition mismatch is explicit: our own generator emits style j with probability softmax(h0)_j, whereas this reader predicts cue probabilities L*softmax(h0). We verified the normalized likelihood and a critical null control: if one cue is instead drawn from the reader's own predictive distribution, the mean updated posterior is exactly its original prior. For any number of cues drawn from the coherent joint posterior predictive, the same identity follows from total expectation. It leaves this toy's mean linear strategy readout unchanged. Posterior averaging alone is not a no-learning theorem: external observations can improve Bayesian decisions even though their average posterior also equals the prior. The no-new-information claim below instead uses the conditional independence of internal rehearsal from the external task. The source-unaware reader in the successful toy is doing inference under a mismatched model of where its context came from. This is a possible computational asymmetry, not a violation of Bayesian information bounds. The numerical checks are saved as `likelihood_check.json`.

## Information and ordinary self-training

With pretrained weights W, initial data D, unknown task parameter theta, and internally sampled rehearsal R, the construction has conditional independence theta -> (D,W) -> R. Thus

```
I(theta ; D,R | W) = I(theta ; D | W).
```

An optimal Bayesian decision rule cannot gain information from these samples. A limited decoder can nevertheless use additional computation and a more accessible context representation. Treating internally generated answers as independent external evidence can also make it confidently wrong.

This context-state mechanism is different from ordinary on-policy SFT. For differentiable normalized q_theta, with sampling and fitting evaluated at the same theta and x,

```
E_(y~q_theta(.|x)) [grad_theta log q_theta(y|x)] = 0.
```

This is a first-order population identity, not a claim that finite noisy training can never change risk. A weight-learning explanation needs an explicit asymmetry, such as a useful generative map and an imperfect inverse recognizer. The existing linear Gaussian toy in `LINEAR_ELICITATION_TOY.md` has exactly this property: generate z and x=Az+noise, then train B by B <- B-eta*(Bx-z)*x^T. Its mean gradient is B*(AA^T+sigma^2 I)-A^T. The useful information was already encoded in A. This resembles the recognition update in the [wake–sleep algorithm](https://www.cs.utoronto.ca/~hinton/absps/ws.htm). It does not by itself establish a source advantage when a teacher supplies the identical joint distribution of pairs.

Greedy self-targets, or sampling at one temperature and fitting a policy at another, need not have zero expected gradient. Those interventions can move the weights without judging answers, but the identity then no longer supplies a prediction about their direction or usefulness. Our current experiment updates context only, so it does not test any of these SFT mechanisms.

A latent-concept account of ICL has precedent in [Xie et al.](https://arxiv.org/abs/2111.02080). Our reader is deliberately approximate and source-unaware; its updates to self-generated context are not coherent Bayesian evidence about an external task.

## Scaling predictions and falsification

With finitely many strategies and a strict gap in expected context drift, concentration generally gives an exponential approach to a limiting strategy, which may be good or bad. A steeper power law does not follow from this toy.

A heterogeneous collection of task modes can yield a power law if their effective rates have the required density near zero. For example, if the weighted rate density behaves as c*r^(alpha-1), then integrating exp(-N*r) gives asymptotically c*Gamma(alpha)*N^(-alpha). Different exponents require different small-rate behavior; multiplying every rate by a constant changes the prefactor, not alpha. Neither condition has been established in these experiments.

Useful tests are: use an independently constructed teacher with the same simple strategy; manipulate strategy cues while holding correctness fixed; shuffle labels or answers only as a separately declared diagnostic; replay identical content through both sources; and measure whether growing rehearsal concentrates valid or invalid answer patterns. More samples should help only when the resulting context drift favors a useful strategy. The new constructive-task search can test that claim without the factorization identity answer.
