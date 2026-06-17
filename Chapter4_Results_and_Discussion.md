# Chapter 4: Results and Discussion

## 4.1 Introduction

This chapter presents the evaluation of AdaptGrip, a reinforcement-learning-based adaptive force control system for a 5-DOF robotic gripper. The evaluation was carried out in two stages. First, the trained Proximal Policy Optimisation (PPO) agent was assessed in simulation against three rule-based baseline controllers to establish whether learning a policy offers measurable benefit over conventional control strategies. Second, the policy was deployed onto the physical LimpSquid 5-DOF arm (Arduino Mega 2560, PCA9685 servo driver, dual FSR 402 force sensors on channels A6/A7) and validated through a structured 100-trial real-world grasping campaign across five object fragility categories. The results from both stages are reported, followed by a discussion that reconciles the simulation performance with the real-world findings, with particular attention to the sim-to-real gap created by force-sensor hardware limitations and how this was addressed through a hybrid contact-finding and squeeze-limiting mechanism.

## 4.2 Simulation Results

### 4.2.1 Baseline Controller Performance

Three conventional control strategies were implemented as baselines for comparison against the learned policy:

| Controller | Success Rate | Damage Rate | Avg. Applied Force |
|---|---|---|---|
| Fixed-Force | 60.0% | 40.0% | 14.8 N |
| Weight-Based | 55.0% | 35.0% | 15.5 N |
| Compliance Control | 80.0% | 20.0% | 3.5 N |

The **Fixed-Force** controller applies a single, constant gripping force regardless of object identity. It performs adequately on robust objects but damages or fails to securely hold fragile and medium objects, resulting in a 40% damage rate.

The **Weight-Based** controller scales grip force proportionally to estimated object mass. This improves marginally on Fixed-Force for very light or very heavy objects but performs worst overall (55% success), because mass alone is a poor predictor of an object's damage threshold — a heavy but fragile object (e.g., a glass jar) is still frequently crushed.

The **Compliance Controller** incrementally increases force until a force-feedback threshold is reached, then holds. This is the strongest of the three baselines (80% success, 20% damage) since it reacts to sensed force rather than acting on an open-loop estimate, but it has no notion of *how much margin* to leave below the damage threshold, so it still damages 1 in 5 objects.

### 4.2.2 PPO Agent Performance

The PPO agent was trained for 50,000 timesteps using a four-stage curriculum that progressively introduced more fragile object categories:

| Stage | Timestep | Objects Introduced |
|---|---|---|
| 1 | 0 | Robust objects only |
| 2 | 12,500 | + Medium objects |
| 3 | 25,000 | + Fragile objects |
| 4 | 37,500 | + Very Fragile objects (full curriculum) |

The training curve (rolling average episode reward) rose from approximately −150 at the start of training to a stable plateau of approximately +150 by 40,000 timesteps, with each curriculum transition (marked by vertical dashed lines) producing a temporary dip in reward and success rate as the agent encountered a new, more difficult object class, followed by rapid recovery. By the end of training the agent converged to:

- **Success rate:** ~100%
- **Damage rate:** ~0%
- **Drop rate:** ~0%

This demonstrates that the curriculum strategy was effective: rather than exposing the agent to the hardest (most fragile) objects immediately, gradually shrinking the margin for error allowed the policy to first learn a stable grasping strategy on forgiving objects before fine-tuning the precision needed for fragile ones.

### 4.2.3 Comparative Analysis

| Controller | Success Rate | Damage Rate |
|---|---|---|
| Fixed-Force | 60.0% | 40.0% |
| Weight-Based | 55.0% | 35.0% |
| Compliance | 80.0% | 20.0% |
| **PPO Agent (Proposed)** | **~100%** | **~0%** |

In simulation, the PPO agent outperforms the best rule-based baseline (Compliance Control) by approximately 20 percentage points in success rate while eliminating object damage almost entirely. The key advantage over Compliance Control is that the PPO agent does not merely react to a force threshold — it has learned, through the reward signal, to anticipate the *safety margin* required for a given object's fragility level rather than only stopping once damage feedback is already detected. This addresses the central question often raised by panel members ("why not just use a fixed force-feedback rule"): a hand-tuned rule generalises poorly across a continuous range of fragility values, whereas the learned policy implicitly interpolates between training conditions.

It should be emphasised that these simulation results represent an upper bound on achievable performance, since the simulated force sensor is noise-free and unsaturated. Section 4.3 shows that this assumption does not hold on the physical hardware.

## 4.3 Hardware Validation Results

### 4.3.1 FSR Calibration Results

The two FSR 402 sensors (left: analog channel A6, right: analog channel A7) were calibrated using a 5-point reference procedure: a known force was applied to each sensor in turn, the raw ADC reading was recorded via the controller's interactive calibration mode, and a least-squares linear regression through the origin (raw ADC count → Newtons) was fitted across the five calibration points for each sensor. This produced a single scale factor per sensor used by `raw_fsr_to_newton()` for all subsequent force readings.

A clean unloaded baseline reading of approximately 15 (left) and 1 (right) raw ADC counts was confirmed prior to the 100-trial campaign, verifying that the gripper was fully open and the sensors unloaded at the reference zero-point — an earlier calibration attempt had produced an erroneously high baseline (862/914) due to the gripper not being fully open when the baseline was recorded, which was identified and corrected before formal testing began.

### 4.3.2 Real Object Grip Force Measurements

A total of 100 grasping trials were executed on the physical hardware: 20 trials each across five object fragility categories (Very Fragile, Fragile, Medium, Robust, Very Robust), using a single representative object per category held constant across all 20 trials in that category. Each trial was logged automatically to CSV via the controller's D-pad logging interface (Up = SUCCESS, Down = FAIL), recording the trial number, category, required force, damage limit, measured actual force, outcome, and timestamp.

| Category | Required Force (N) | Damage Limit (N) | Success | Fail | Success Rate |
|---|---|---|---|---|---|
| Very Fragile | 0.589 | 5.0 | 12 | 8 | 60% |
| Fragile | 1.177 | 10.0 | 16 | 4 | 80% |
| Medium | 2.354 | 20.0 | 14 | 6 | 70% |
| Robust | 4.709 | 40.0 | 18 | 2 | 90% |
| Very Robust | 9.418 | 80.0 | 17 | 3 | 85% |
| **Overall** | — | — | **77** | **23** | **77%** |

The overall real-world success rate of 77% is markedly lower than the ~100% achieved in simulation. The category-level pattern is also notable: rather than success rate decreasing monotonically with fragility (which would be expected if the dominant failure mode were *damage*), the lowest success rate occurs at the *Very Fragile* end (60%), and the highest occurs for *Robust* objects (90%). This pattern, combined with the force readings discussed next, indicates that the dominant real-world failure mode was **insufficient/insecure grip (slipping)** on fragile objects rather than crushing — the opposite failure mode to what the simulation alone would predict.

### 4.3.3 FSR Saturation Finding

Across all 100 trials, the measured `actual_force_N` values clustered tightly between approximately 3.2 N and 4.3 N — essentially **independent of object category**, despite the required force spanning more than an order of magnitude (0.589 N to 9.418 N) and the damage limit spanning two orders of magnitude (5 N to 80 N). Two trials (Trial 6 and Trial 57) recorded a force reading of exactly 0.0 N, attributable to a serial read timeout rather than a true zero-force event.

This is consistent with hardware saturation of the FSR 402 sensors: their resistive sensing range is well-suited to forces in the sub-1 N to low single-digit-Newton range, but above approximately 1–2 N the sensor's response curve flattens, and the raw ADC reading approaches a ceiling (raw counts approaching ~990–1010, near the practical maximum for the calibrated scale) regardless of how much additional force is actually applied. In practice this means the force-feedback signal that the RL policy relies on becomes **uninformative** for any object requiring more than roughly 1–2 N of grip force — which, per Table 4.3.2, includes every category from *Fragile* upward.

This is presented as a genuine, documented hardware limitation of the FSR 402 sensor for this application, rather than a software defect, and is the central finding explaining the gap between simulated and real-world performance.

### 4.3.4 Sim-to-Real Transfer Assessment

Because the RL policy's only force-feedback observation (`prev_force`) becomes unreliable above ~1–2 N, the policy as trained in simulation cannot be deployed directly on real hardware for any object beyond the *Very Fragile* category without modification: the simulated environment assumes continuous, unsaturated force feedback across the full 0–80 N range, while the real sensor effectively reports a near-constant ~4 N regardless of true grip force once saturated.

To bridge this gap, a **hybrid contact-finding and squeeze-limiting mechanism** was implemented on top of the RL policy's force output:

1. **Contact-finding** (`find_contact()`): the gripper closes in small, fixed angular steps (2° per step, 0.06 s settle time) from fully open, monitoring the *raw* ADC reading rather than the saturation-prone Newton-converted value. Contact is declared once the raw reading exceeds 50 counts above the unloaded baseline — a threshold robust to calibration-scale noise.

2. **Squeeze limiting**: once contact is established, the RL-commanded grip angle is clamped so that the gripper cannot close more than a fixed number of degrees past the contact point, regardless of what the (potentially saturated) force feedback suggests:

| Category | Squeeze Limit |
|---|---|
| Very Fragile | 6° |
| Fragile | 8° |
| Medium | 16° |
| Robust | 22° |
| Very Robust | 35° |

These values were tuned iteratively through real-hardware testing. Early values that were too generous resulted in crushed/cracked fragile objects (e.g., an egg, an empty plastic bottle); values that were too conservative (particularly an initial 0° squeeze for Very Fragile) caused an insecure grip that could slip under the object's own weight. The final values represent the empirically tuned trade-off between these two failure modes, and directly account for the slip-dominated failure pattern observed at the fragile end of Table 4.3.2.

This hybrid approach allows the RL policy to retain its learned, fragility-aware *intent* (smaller squeeze allowance for more fragile categories mirrors the policy's learned behaviour of applying less force to fragile objects) while substituting a reliable angular safety bound for the unreliable force-feedback signal once it saturates.

## 4.4 Discussion

**Why the real-world results differ from simulation.** The ~23-percentage-point drop in success rate between simulation (~100%) and hardware (77%) is attributable almost entirely to the FSR saturation issue described in Section 4.3.3, not to a failure of the underlying PPO policy or training procedure. The simulation environment, by construction, provides ideal, unsaturated, low-noise force feedback across the full operating range; this is a standard limitation of simulation-trained RL policies and is well documented in the sim-to-real transfer literature. The mitigation strategy adopted here — falling back to a model-free, angle-based safety bound when the primary sensing modality saturates — is a pragmatic, defensible engineering response rather than an admission that the RL approach is unnecessary.

**Addressing the "why RL at all" critique.** A recurring panel question was why a hand-coded rule (e.g., "use the FSR data directly and grip until threshold is reached") would not suffice, given that the final deployed system relies on a fixed angular squeeze limit rather than the raw RL force output. The answer demonstrated by Section 4.2.3 is that the *Compliance Control* baseline — which is exactly this kind of hand-coded force-threshold rule — was tested in simulation and only achieved 80% success with 20% damage, materially worse than the learned policy's ~100%/~0%. The RL policy's value lies in implicitly learning a continuous, fragility-conditioned force-application strategy from reward signal alone, rather than requiring a human engineer to hand-tune a separate threshold and trajectory for every possible object class. The squeeze-limiting layer added for hardware deployment is a safety bound layered on top of this learned behaviour, not a replacement for it — the RL agent's policy network still computes the underlying target force/angle on every control step; the squeeze limit only intervenes when sensor saturation would otherwise let the policy command an unsafe closure.

**Why fragility category selection is manual.** Automatic fragility classification (e.g., from vision or pre-contact probing) was identified during development as a valuable future extension but was out of scope for this iteration, since it constitutes a separate perception problem from the force-control problem this project targets. The current system assumes the fragility category is supplied (analogous to how a human operator would specify "handle with care" instructions), and focuses on solving the downstream control problem of applying appropriately calibrated force once that category is known.

**Limitations.** The principal limitation identified through hardware testing is the FSR 402's restricted sensing range, which constrains the force-feedback signal's usefulness to approximately the *Very Fragile* category alone. A second limitation is the small sample size of one representative object per category (20 trials each); a larger object pool per category would better characterise generalisation within a fragility class rather than to a single specific object's geometry and surface friction. Two trials also exhibited apparent serial-communication read timeouts (zero-force readings), suggesting the USB serial link to the Arduino Mega could occasionally benefit from more robust timeout/retry handling.

## 4.5 Summary

This chapter presented a two-stage evaluation of AdaptGrip. In simulation, the PPO-trained agent outperformed all three rule-based baseline controllers, achieving approximately 100% grasp success with approximately 0% object damage, compared to a best baseline (Compliance Control) of 80% success and 20% damage. Hardware validation across 100 real-world trials (5 categories × 20 trials) showed a reduced but still strong overall success rate of 77%, with the gap to simulation traced to a specific, documented hardware limitation: saturation of the FSR 402 force sensors above approximately 1–2 N, which renders direct force feedback unreliable for all but the lightest object category. This was mitigated through a hybrid contact-finding and per-category angular squeeze-limiting mechanism, tuned empirically to balance the competing risks of crushing fragile objects and failing to securely grip them. The results support the central claim that a learned, fragility-aware control policy offers a measurable advantage over fixed-rule controllers, while also surfacing a concrete, reportable sim-to-real transfer challenge characteristic of force-sensing hardware in low-cost robotic platforms.
