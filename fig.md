Based on the analysis of Version 1, the missing figure is the **"Target Drift Analysis" plot** (originally Figure 5 in Version 1). Restoring this figure is critical because it visually proves your central hypothesis: that standard methods suffer from "pivot instability" while SODA fixes it.

Here is a detailed description of exactly how this figure should be constructed, defined, and interpreted to satisfy the reviewers.

### Figure Title: Quantitative Stability Analysis (Target Drift)

**Type of Chart:** Line Plot
**Placement:** Section 4.4 (Quantitative Stability Analysis)

#### 1. The Axes
*   **X-Axis:** **Training Iterations** (e.g., 0 to 5,000 steps).
    *   *Label:* "Training Iterations"
*   **Y-Axis:** **Target Drift** (Magnitude of change in the alignment target).
    *   *Label:* "Target Drift: $W_2(P_{target}^{(t)}, P_{target}^{(t+1)})$"
    *   *Scale:* Linear (e.g., 0.0 to 0.5).

#### 2. The Plotted Series
You need two distinct curves on this plot to show the contrast:

*   **Curve A (The Baseline - Red/Dashed):** **"Standard Batch-Derived Pivot"**
    *   *Mechanism:* At every step $t$, the target distribution $P_{target}$ is constructed using only the features currently in the mini-batch (size $B=32$).
    *   *Visual Behavior:* This line should be **high** and **highly oscillatory** (noisy). It essentially looks like a thick band of noise that never settles down.
    *   *Meaning:* This visualizes the "Impossibility Theorem." Because the batch size is small ($B \ll d'$), the distribution changes radically every time you sample a new batch. The target is never the same twice.

*   **Curve B (SODA - Blue/Solid):** **"Dynamic Feature Reservoir (Ours)"**
    *   *Mechanism:* The target $P_{target}$ is the distribution defined by the Reservoir ($K=512$).
    *   *Visual Behavior:* This line should start high (during initialization) but **rapidly decay** (exponentially) to a low value and become **smooth/flat**.
    *   *Meaning:* This proves that as the reservoir fills up and evolves, the geometric target stabilizes. The "momentum" of the queue filters out the sampling noise.

#### 3. Mathematical Definition (For the Caption)
To satisfy the "theoretical rigor" reviewer, you must define exactly what is being plotted in the caption:
*   **Target Drift ($\mathcal{S}$):** The Wasserstein-2 distance between the alignment target distribution used at iteration $t$ and the target used at iteration $t+1$.
    $$ \mathcal{S}^{(t)} = W_2\left( P_{pivot}^{(t)}, P_{pivot}^{(t+1)} \right) $$

#### 4. The Narrative (How to describe it in the text)
When you insert this figure, you should add the following text to Section 4.4:

> "To empirically validate the impact of the DFR on optimization dynamics, **Figure 5** visualizes the 'Target Drift'—defined as the Wasserstein distance between the alignment pivot at consecutive iterations ($W_2(P_t, P_{t+1})$).
>
> As shown by the **red curve**, standard batch-derived pivots exhibit persistent, high-amplitude variance throughout training. This confirms our theoretical analysis that $B \ll d'$ leads to statistically ambiguous targets that fluctuate wildly based on sampling noise.
>
> In contrast, SODA (**blue curve**) demonstrates a rapid convergence to a stable geometric anchor. The low-variance regime achieved after initialization confirms that the DFR ($K \gg B$) successfully decouples the alignment target from mini-batch stochasticity, providing the stable optimization landscape required for effective Optimal Transport."

### Why This Figure Fixes the Reviews
1.  **Addresses Reviewer 1 (Comment 2):** They explicitly asked for *"visual analyses to support the claim of reduced instability."* This figure is that exact visual analysis.
2.  **Addresses Reviewer 2 (Weakness 2):** They complained about *"qualitative reasoning about instability without evidence."* This figure converts the qualitative argument into quantitative, empirical evidence.
3.  **Addresses Reviewer 3 (Comment 13):** They asked for *"More visualization results... to visually show the working principle."* This shows the working principle (stability) in action.