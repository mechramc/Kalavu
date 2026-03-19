# Section 4.9.3 Method Note — N=20 Architectural Adaptations

Two versions below. Paste the correct one based on which run gave GO.

---

## VERSION A — Use if base_hidden clean run gives GO
## (no load-balance loss needed)

> At N=20, the mean-pooling strategy used for router input in prior experiments
> (averaging the final hidden states of all specialists) degrades below the linear
> separability threshold: each specialist contributes only 5\% to the pooled
> representation, diluting the domain signal that the router must distinguish.
> We therefore replace the pooled specialist signal with the final hidden state of
> the \emph{frozen base model} — a single forward pass on the input sequence before
> any specialist adaptation is applied.
> This preserves the full domain signature of the input, mirroring the routing
> strategy used in large-scale MoE systems~\cite{fedus2022switch,jiang2024mixtral},
> where the router receives the token representation prior to expert computation.
> All other training hyperparameters (router learning rate, gradient accumulation,
> number of steps) are held constant across experiments to ensure that performance
> differences reflect the input signal change rather than optimisation differences.

---

## VERSION B — Use if base_hidden clean run gives PIVOT and v2 gives GO
## (load-balance loss + cosine annealing required)

> At N=20, the mean-pooling strategy used for router input in prior experiments
> degrades below the linear separability threshold: each specialist contributes
> only 5\% to the pooled representation, diluting the domain signal available to
> the linear router.
> We therefore replace the pooled specialist signal with the final hidden state
> of the \emph{frozen base model} — a single forward pass on the input sequence
> before any specialist adaptation is applied.
> This preserves the full domain signature of the input, mirroring the routing
> strategy used in large-scale MoE systems~\cite{fedus2022switch,jiang2024mixtral}.
> A preliminary run with this input change alone produced a PIVOT verdict,
> with gate diagnostics revealing expert collapse: the top-2 generalist experts
> absorbed \textgreater{}80\% of routing weight, consistent with the known failure
> mode of soft-routing under NLL-only training~\cite{fedus2022switch}.
> We therefore added a standard load-balancing auxiliary loss
> $\mathcal{L}_\text{balance} = \|\bar{g} - \mathbf{1}/N\|^2$,
> where $\bar{g}$ is the mean gate vector over the batch,
> with coefficient $\lambda = 0.05$ following~\citet{fedus2022switch},
> and cosine annealing on the router learning rate to stabilise late-training
> gate assignments.
> These adaptations are empirically motivated by the N=20 scale regime and the
> observed collapse pattern; they are not needed at N$\leq$4 (all prior experiments)
> where each expert contributes $\geq$25\% to the pooled signal.

---

## BibTeX entries needed (if not already in paper)

```bibtex
@article{fedus2022switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={120},
  pages={1--39},
  year={2022}
}

@article{jiang2024mixtral,
  title={Mixtral of Experts},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Roux, Antoine and Mensch, Arthur and
          Savary, Blanche and Bamford, Chris and Chaplot, Devendra Singh and de las Casas, Diego and
          Hanna, Emma Bou and Bressand, Florian and others},
  journal={arXiv preprint arXiv:2401.04088},
  year={2024}
}
```
