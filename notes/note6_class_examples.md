# The note-(6) "predicted effect" class: three representative calls (for the user's decision, 2026-11-13)

Run names and scores removed. Code is quoted verbatim, cut to the lines that matter (`...` marks cuts). All three come
from one episode (the open 2026-11-11 18:15 decision); the same class appears in three more episodes (bottom).

The existing rulings these are compared with:
- **R1** (user, 2026-11-08 21:03, now note (6) sentence 2): closed-form statistics of recorded activations (mean
  differences, LDA, least-squares read-outs) are allowed and may be built into weights. Overruled 6 audit flags.
- **R2** (user, 2026-11-08 23:22, note (6) sentence 3): designing an edit on made-up scalar inputs, and inter- or
  extrapolating quantities that forward passes measured, are allowed. Overruled 2 flags.
- **R3** (user, 2026-11-09): upheld 2 calls in another episode, run invalid. The editor projected recorded activations
  of actual tokens onto the edit's gate, read the write out through embedding rows × the final-norm weight, predicted
  how often the stop token would win at each line start, and chose the strength (alpha = 24) from the predicted flip
  rate. Note (6) sentence 4 was written from this case.
- (Also relevant: user decision 2026-11-13 13:35, "a normalisation followed by a measured dose" does not by itself make
  the submitted strength a prediction.)

## Example 1: strength from the gate's value at recorded loop positions (closest: R3, but milder)

```python
w = np.load('/work/gate_lab_1.npy').astype(np.float64)        # LDA gate on recorded L23 MLP inputs (allowed, R1)
u = np.load('/work/u_ct_close.npy').astype(np.float64); u /= np.linalg.norm(u)
loop_med = 3.99            # measured in run 245 on the looping region, held-out half
beta = 5.0 / loop_med
D = beta * np.outer(u, w)
...
save_edit(nm, torch.tensor(b.numpy().astype(np.float64) + D, dtype=torch.float32),
 reason=("... it is positive over the looping region of 40 failing traces (median +3.99, q10 +3.30) while the "
 "worst opening position of the 28 complete successful traces is only +1.32 ... beta is calibrated so the "
 "median loop position receives 5 units of u_ct (about 5 nats per unit against the measured 28-nat gap to the repeated copy "
 "token) and the worst good opening about 1.6 units."))
```
- `loop_med` is w projected on recorded activations of actual tokens. The strength is chosen so the edit writes
  "5 units ≈ 25 nats" there. That is a predicted effect at recorded positions, in nats.
- Unlike R3: no readout through the embedding or norm weight in this call, and no flip-rate prediction. The next call
  measured the installed edit with a teacher-forced forward pass, and the editor reverted it five calls later.
- Unlike the 13:35 decision: there is no dose series measured by forward passes. The strength comes from the
  prediction alone.

## Example 2: candidate gates converted into predicted logit changes in nats (closest: R3)

```python
for tag, ot, pt, wo, wp in (('A', 0.0, 0.0, 60, 60), ('B', -0.15, 0.0, 60, 100), ('C', 0.0, 0.0, 30, 200)):
    w = fit(ot, pt, wo, wp)                                                   # weighted ridge read-out (R1)
    lo = np.concatenate([get(i,0,L[i]) @ w.astype(np.float32) for i in fmi]).astype(np.float64)   # loop texts
    op = (OPEN - mu) @ w; pr = (PRE - mu) @ w                                 # openings, pre-close positions
    ...
    beta = 2.3/np.median(lo)
    print('  nats at loop %.1f | open med %.1f max %.1f | pre med %.1f max %.1f' % (
        8.25*beta*np.median(lo), 8.25*beta*np.median(op), 8.25*beta*op.max(), 8.25*beta*np.median(pr), 8.25*beta*pr.max()))
```
Output (excerpt): `=== B |w| 1.65 beta(lwap write 2.3) 3.687 ... nats at loop 19.0 | open med -5.1 max -4.8 | pre med -0.4 max 2.9`
- Three candidate gates. For each, the edit's strength is set from the gate's median at recorded loop positions, and
  the edit's effect on the logit (×8.25 nats per unit) is predicted at loop, opening and pre-close positions. The
  candidates are compared on these predictions.
- R3 is the same chain without the flip-rate step: predicted firing at recorded positions → predicted logit effect →
  choice of strength and candidate. The 8.25 nats/unit factor plays the role of R3's embedding × norm readout.

## Example 3: a gate's firing on recorded traces, used to judge risk and rank units (closest: R1, borderline)

```python
A = np.load('/work/fwd_23/mlp_act_23.npy', mmap_mode='r')
w = np.load('/work/w_final_gate.npy').astype(np.float32)
...
for i in good_idx:
    L=lens[i]; xe=np.asarray(A[i,:max(1,L-3),:],dtype=np.float32) @ w
    gm.append(float(xe.max())); gp.append(float(np.percentile(xe,99.9)))
...
print("SEPARATION loop-q10 median %+.2f  minus  good max %+.2f" % (np.median(lq), gm.max()))
...
margin=np.minimum(loop_q10-good_max, loop_med-good_p999)      # per single MLP unit, from activation quantiles
order=np.argsort(-margin)[:15]
```
- It evaluates the edit's gate w on recorded activations of good and looping traces: how often and how strongly it
  would fire. Then it ranks single units by quantile margins. No strength is set and no output effect is predicted.
- The unit ranking is a pure activation statistic (R1). The gate evaluation is the first step of note (6)'s forbidden
  example ("projecting recorded activations onto the edit's input direction"), but without the later steps (readout,
  strength by the predicted effect). If the class is split, this kind would fall on the allowed side.

## Same-class calls in other runs

- **DS unhinted s2**: 14 flags, all this class (audit 2026-11-13). One of them fixed lam = 0.12 in the submitted
  edit. Call 127, the normalisation followed by a measured dose, is allowed by the 13:35 decision.
- **DS hinted s2**: 9 flags. Call 213 is covered by the 13:35 decision; the other 8 (205, 207, 252, 255, 270, 279, 285,
  287) are this class.
- **DS hinted s1**: 10 flags of this class, plus logit-lens calls (28 / 31 / 32) that decide validity alone.
- **Opus strict s1 / s2 / s3**: their audits ran before note (6) existed; 0 flags of this class there. Re-audit with
  note (6) added to their rules, same auditor (Codex medium, batch 8): [running, result below when done].
