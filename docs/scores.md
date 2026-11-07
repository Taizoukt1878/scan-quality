# Scan Quality Scores

This document defines the scores used to **accept or reject a document photo at capture time**. It
targets phone and kiosk captures for insurance and banking: cheques, accident reports, claim forms,
invoices and supporting documents. A rejected photo is never sent downstream; the user is asked to
retake it. Accepted photos go to a dedicated OCR and extraction pipeline.

Each section gives the recommended formula for this use case, how to read it, a worked example, and
the alternatives we considered.

> **Status.** The code in [`src/scan_quality/scoring/`](../src/scan_quality/scoring/) still runs the
> original five checks: sharpness, lighting, contrast, skew and OCR, combined with a plain mean.
> Each section below says what runs today and what we recommend. All anchor values are reasoned
> starting points and **must be calibrated on real captures** (see [Calibration](#calibration)).

---

## Design principles for this use case

1. **Score the document, not the frame.** A phone photo includes the table, the user's hand and the
   car seat. Every score is computed on the detected document after it has been rectified to a
   known physical size. Without this, a cheque that fills a third of the frame would be judged at
   the wrong resolution, and the table's texture would count as "sharpness".
2. **Penalize only defects the downstream pipeline can't fix.** Rotation and perspective can be
   corrected later and are *not* penalized. Blur, glare, cut-off edges, low resolution and ink that
   is too faint are lost for good, so these are what we gate on.
3. **Use the weakest link, not the average.** One glare spot over the amount on a cheque ruins the
   capture, however good the other scores are. A mean hides that kind of defect.
4. **Every failure maps to an instruction.** Each check catches one physical problem, so a
   rejection can tell the user what to do: "hold still", "avoid the reflection", "move closer".
5. **Keep it fast enough for the live camera preview.** All checks are pixel statistics with no OCR
   and no external binary, so they can run on preview frames and guide the user before they press
   the shutter.

### Why OCR confidence was removed

- The final OCR runs downstream and is better than Tesseract. Tesseract's confidence would be a
  weaker estimate of a result we get anyway.
- It was about 10× slower than the other four checks combined (638 ms out of 682 ms per request).
  That rules out live feedback.
- It needs the Tesseract binary and language packs on the device.
- Its confidence depends on the language and vocabulary as much as on the image. A clean French
  quote scored with `eng` looks worse than a blurred English one.

---

## Shared conventions

Each check measures a **raw value** in physical units and maps it onto a **0–100 score** (higher is
better) with a clamped ramp from [`base.py`](../src/scan_quality/scoring/base.py):

$$
\text{linear}(x) = \operatorname{clamp}_{[0,100]}\left(100 \cdot \frac{x - \text{zero\_at}}{\text{hundred\_at} - \text{zero\_at}}\right)
\qquad
\text{log}(x) = \text{linear}\ \text{applied to}\ \log_{10} x
$$

When `zero_at > hundred_at` the ramp runs backwards (lower raw value is better). Values beyond the
anchors are clamped.

| Score | Label | Capture decision |
|-------|-------|------------------|
| ≥ 85 | `excellent` | Accept |
| ≥ 70 | `good` | Accept |
| ≥ 50 | `fair` | Accept (default threshold; tune per check) |
| < 50 | `poor` | Reject and ask for a retake |

---

## Step 0: Document detection and rectification

This step isn't a score, but every score below depends on it.

1. Find the document's four corners, either with contours and polygon approximation or with a
   lightweight edge-detection model.
2. Compute the homography $H$ that maps the quad to a rectangle of the document's **known physical
   size** at a canonical resolution of **8 px/mm** (about 200 DPI):

| Document type | Physical size | Rectified size |
|---|---|---|
| French cheque | 175 × 80 mm | 1400 × 640 px |
| A4 page (claim form, constat, invoice) | 210 × 297 mm | 1680 × 2376 px |

3. Warp the image with $H$. All checks run on this rectified image $D$, in color.

If no quad is found, the capture is rejected with the message *"Place the document on a darker,
plain surface"*. We don't fall back to the full frame, because scores computed on the frame aren't
reliable.

---

## 1. Sharpness

**Today in code:** the variance of the Laplacian over the whole frame, mapped with a log ramp
(15 → 0, 600 → 100).

### Recommended formula

The global variance has two problems with phone captures:

- **Focus falloff.** The phone focuses on one part of the page, and the far edge, where the cheque's
  amount or signature often sits, is blurred. A global variance averages that away.
- **Content density.** A cheque is mostly blank and a claim form is dense. The same blur gives very
  different variances on the two.

The recommended formula therefore measures **contrast-normalized sharpness per tile** and scores
the **worst part** of the document:

1. Split $D$ (grayscale) into 64 × 64 px tiles (8 × 8 mm).
2. Keep only **content tiles**: tiles where at least 3 % of pixels are ink, using the adaptive
   threshold mask from the skew code. Blank paper tiles contain only noise and are ignored.
3. For each content tile $t$, compute the ratio of Laplacian variance to intensity variance. The
   ratio is independent of how dark the ink is, so it measures **how much of the tile's contrast
   sits in fine detail**:

$$
r_t = \frac{\operatorname{Var}\left(\nabla^2 D_t\right)}{\operatorname{Var}\left(D_t\right) + \varepsilon}
$$

4. The raw value is the **10th percentile** across content tiles. This represents the blurriest
   tenth of the text:

$$
R = p_{10}\left(\{ r_t \}\right)
$$

5. Score it with a log ramp, using illustrative anchors **0.1 → 0** and **2.0 → 100**.

With fewer than 10 content tiles, for example a nearly blank document, use the median of all tiles
instead.

### Interpretation

- $R$ reads as *"how sharp is the blurriest tenth of the text?"*. A capture passes only if all of
  the text can be read, not just most of it.
- Motion blur from hand shake lowers $r_t$ in every tile. Focus falloff lowers it in some tiles
  only, and the percentile catches that.
- The 10th percentile, rather than the minimum, stops a single odd tile from rejecting a good
  capture. An odd tile can be a stamp, a smudge or a fold.

### Example

A user photographs a cheque at an angle. The phone focuses on the left side, where the payee line
is. The right side, with the amount box and the signature, is soft. There are 48 content tiles:

| Statistic | Value | Score |
|---|---|---|
| Median $r_t$ (what a global measure sees) | 1.6 | $100 \cdot \log_{10}(16)/\log_{10}(20) = 92.55$, `excellent` |
| $p_{10}$ of $r_t$ (recommended) | 0.35 | $100 \cdot \log_{10}(3.5)/\log_{10}(20) = 41.82$, **`poor`** |

A global score would accept this cheque even though the amount can't be read. The tiled score
rejects it with the message *"Hold the phone parallel to the document"*.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Global Laplacian variance (today) | Misses focus falloff; depends on content density |
| Tenengrad / Brenner per tile | Good alternatives; they depend on contrast unless normalized the same way |
| Edge width (Marziliano) | Gives physical units (px of blur), but is harder to make robust on handwriting |
| FFT high-frequency ratio | Costs more per tile for a similar signal |

---

## 2. Lighting: shadows

**Today in code:** the standard deviation of mean brightness across an 8 × 8 grid (45 → 0,
4 → 100). This responds to layout as well as light: a clean page scores about 75.

### Recommended formula

Measure the light falling on the **paper only**, after removing the ink:

1. Estimate the background $B$: apply a morphological closing to grayscale $D$ with a 15 × 15
   kernel (about 2 mm, wider than any pen stroke), which removes the ink. Then smooth with a
   51 × 51 median blur.
2. The raw value is the **shading ratio** between the darkest and brightest parts of the paper:

$$
S = \frac{p_{5}(B)}{p_{95}(B)}
$$

3. Score it with a linear ramp, using illustrative anchors **0.40 → 0** and **0.80 → 100**.

### Interpretation

- $S$ is the share of the brightest region's light that the darkest region receives. With
  $S = 0.5$, the shadowed part of the page is half as bright as the lit part.
- A ratio doesn't depend on exposure: a dim room with even light isn't penalized.
- The anchors are **lenient** on purpose. A soft gradient is easy to correct downstream by dividing
  by the background (flat-field correction). The real harm of a deep shadow is lost ink contrast,
  and the [Contrast](#4-contrast) check measures that directly in the worst region. This check
  exists mainly to give the user a clear message.
- **Caveat:** cheques with large colored security backgrounds can read as shading. Calibrate on
  your real cheque stock.

### Example

A user photographs an accident report on a car seat. Their arm casts a hard shadow across the lower
half. The background measures 230 in the lit part and 110 in the shadow:

$$
S = \frac{110}{230} = 0.478 \qquad \text{score} = 100 \cdot \frac{0.478 - 0.40}{0.40} = 19.6 \;\rightarrow\; \texttt{poor}
$$

The capture is rejected with *"Move to even light; avoid casting a shadow"*. A soft window gradient
($S = 0.75$) would score 87.5 and pass.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Cell-mean standard deviation (today) | Responds to text density, not just to light |
| Background standard deviation | Depends on exposure; harder to read than a ratio |
| Fitted brightness plane | Only catches linear gradients, not a hard shadow edge |

---

## 3. Glare (new, split from lighting)

**Today in code:** not measured. The lighting check averages a glare spot into one grid cell.

### Recommended formula

Glare is the most common capture failure for **glossy and laminated documents** and for photos
taken under ceiling lights. Unlike a shadow, it **destroys the information underneath**, so it gets
its own strict check:

1. On the color image $D$, mark a pixel as specular when **all three channels are near
   saturation**: $\min(R, G, B) \ge 250$.
2. Apply a morphological opening to drop isolated pixels, and keep connected blobs of at least
   20 px.
3. The raw value is the **area of the largest blob** as a share of the document's area:

$$
G = \frac{\max_k \lvert \text{blob}_k \rvert}{\lvert D \rvert}
$$

4. Score it with a reversed linear ramp, using illustrative anchors **1.0 % → 0** and
   **0.05 % → 100**.

We use the largest blob rather than the total because one solid blob hides a word, while scattered
specks hide nothing.

### Interpretation

- At 8 px/mm, 1 % of a cheque (1400 × 640 px) is about 9 000 px, or roughly 12 × 12 mm. That is
  enough to hide the amount in figures.
- Strongly overexposed paper also saturates and counts as glare. That is intended, because
  overexposure wipes out faint ink the same way.
- **Zone rule (cheques):** a glare blob that overlaps a required zone (the amount in figures, the
  amount in words, the MICR/CMC7 line or the signature) rejects the capture whatever the score.

### Example

A user photographs a laminated insurance card under an office light. A reflection of
100 × 60 px appears on a document of 1400 × 640 px:

$$
G = \frac{6\,000}{896\,000} = 0.670\,\% \qquad \text{score} = 100 \cdot \frac{1.0 - 0.670}{1.0 - 0.05} = 34.8 \;\rightarrow\; \texttt{poor}
$$

The capture is rejected with *"Tilt the phone slightly to remove the reflection"*.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Total saturated area | Penalizes harmless scattered specks as much as one blob hiding a word |
| HSV rule (high V, low S) | Similar result; the min-channel rule is simpler and doesn't need a color-space conversion |
| Glare detection model | More robust on holograms, but needs training data; worth revisiting later |

---

## 4. Contrast

**Today in code:** RMS contrast over the whole frame (12 → 0, 65 → 100). This penalizes sparse
documents such as cheques, which are mostly blank paper.

### Recommended formula

Measure **how well ink separates from paper**, independently of how much ink there is:

1. Convert $D$ to a **min-channel** image $M = \min(R, G, B)$. Blue ballpoint ink, which is common
   on cheques and handwritten claim forms, is faint in ordinary grayscale but dark in the red
   channel. Red stamps are dark in green and blue. The minimum keeps all inks dark and white paper
   bright.
2. Flat-field correct it, $F = 255 \cdot M / B_M$, where $B_M$ is the background of $M$ estimated
   as in the shadows check. This removes the shadows so they aren't counted twice.
3. Split pixels into ink $i$ and paper $p$ with the adaptive threshold mask.
4. On a 4 × 4 grid over the document, in each cell with at least 200 ink pixels, compute the
   **separability** $d'$:

$$
d' = \frac{\mu_p - \mu_i}{\sqrt{\tfrac{1}{2}\left(\sigma_p^2 + \sigma_i^2\right)}}
$$

5. The raw value is the **minimum $d'$ across cells**, i.e. the region where ink is hardest to
   tell from paper. Score it with a linear ramp, using illustrative anchors **2 → 0** and
   **6 → 100**.

### Interpretation

- $d'$ asks *"how many noise widths apart are ink and paper?"*. Around 2, a threshold misclassifies
  many pixels and strokes break up. Above about 6, binarization is clean.
- It uses only the two groups' statistics, not their sizes. A cheque with a few lines of ink scores
  the same as a dense page with the same ink quality, which fixes the sparse-document problem.
- Taking the minimum across cells means the check catches a deep shadow, a faded corner or a
  sensor-noise region (low light), wherever it is.

### Example

The handwritten amount on a cheque, in blue ballpoint. The paper averages 230 (σ = 8):

| Channel | Ink μ (σ = 20) | $d'$ | Score |
|---|---|---|---|
| Ordinary grayscale | 170 | $60 / 15.23 = 3.94$ | 48.5, `poor` |
| Min-channel (recommended) | 90 | $140 / 15.23 = 9.19$ | 100, `excellent` |

The ink is perfectly legible. Ordinary grayscale would have rejected a good cheque because of the
pen color, not the photo. The same cheque under the current RMS formula measures low as well, just
because it is mostly blank paper.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| RMS contrast (today) | Depends on ink coverage; penalizes cheques |
| Ink–paper mean difference $\mu_p - \mu_i$ | Ignores noise: a dark but grainy low-light photo would pass |
| Michelson / Weber on percentiles | Also depend on coverage through the percentiles |

---

## 5. Framing and resolution (replaces skew)

**Today in code:** the text-line angle from a Hough transform on baseline points (5° → 0,
0.5° → 100).

### Why skew is no longer scored

Rotation is **fully recoverable**. Step 0 rectifies the document, and the downstream OCR deskews
whatever is left. Penalizing it would reject captures that extract perfectly. The angle is still
reported for information and for 90°/180° orientation handling downstream.

What a phone capture *can* get wrong for good is **framing**: part of the document is outside the
photo, or the document is too far away to have enough pixels.

### Recommended formula

1. **In frame (gate).** All four detected corners must be at least 1 % of the frame size away from
   the frame edge. If they aren't, the score is **0**, with the message *"Fit the whole document in
   the frame"*.
2. **Effective resolution.** For each side $k$ of the detected quad in the *original* photo, divide
   its length in pixels by its physical length in mm. The raw value is the **worst side**:

$$
\rho = \min_k \frac{\text{length}_k\ (\text{px})}{\text{length}_k\ (\text{mm})}
$$

3. Score $\rho$ with a linear ramp: **4 px/mm (≈ 100 DPI) → 0** and **8 px/mm (≈ 200 DPI) → 100**.

$$
\text{score} =
\begin{cases}
0 & \text{if a corner is out of frame} \\
\text{linear}(\rho;\ 4,\ 8) & \text{otherwise}
\end{cases}
$$

### Interpretation

- $\rho$ is the resolution of the worst-captured part of the document. It is an upper bound on the
  detail available there, whatever the downstream processing.
- Perspective isn't penalized separately. It matters only through what it costs: fewer pixels on
  the far side, which $\rho$ catches by taking the minimum side, and focus falloff, which the
  sharpness check catches.
- 8 px/mm is enough for printed text and handwriting. Raise the 100 anchor if downstream extraction
  needs to read small print, such as policy terms.

### Example

A user photographs a cheque from too far away with a 4000 × 3000 phone camera. The far (top) edge
of the cheque spans 1 100 px over 175 mm:

$$
\rho = \frac{1\,100}{175} = 6.29\ \text{px/mm} \qquad \text{score} = 100 \cdot \frac{6.29 - 4}{4} = 57.1 \;\rightarrow\; \texttt{fair}
$$

The capture is accepted, but the app can show *"Move closer"*. If the user had cropped off the
signature corner, the capture would score 0 and be rejected, however sharp it was.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Skew angle (today) | Recoverable downstream; penalizing it rejects good captures |
| Separate perspective ratio | Duplicates what $\rho$ and sharpness already measure |
| Total megapixels of the photo | Says nothing about how much of the photo is the document |

---

## Accept or reject decision (replaces the overall mean)

**Today in code:** an unweighted mean of the five scores, labeled with the same thresholds.

### Recommended formula

Use a **weakest-link gate** with one threshold per check:

$$
\text{accept} \iff s_i \ge \tau_i \quad \text{for every check } i
\qquad\qquad
\text{overall} = \min_i\, s_i
$$

- $\tau_i$ defaults to 50 and is tuned per check (see [Calibration](#calibration)).
- The **lowest-scoring check** decides the retake message shown to the user.
- **Retry budget:** after 3 rejected attempts, accept the best capture so far and flag it for manual
  review. A user filing an accident claim must never get stuck in a loop.

### Interpretation

The mean answers *"how good is this photo on average?"*. For a gate, the question is *"is anything
in this photo bad enough to break extraction?"*, and only the minimum answers that.

### Example

A claim form photographed on a desk with a ceiling light above:

| Check | Score |
|---|---|
| Sharpness | 88.00 |
| Shadows | 82.00 |
| Glare | **34.78** |
| Contrast | 90.00 |
| Framing and resolution | 85.00 |
| **Mean (today)** | **75.96, `good`: accepted** |
| **Minimum (recommended)** | **34.78, `poor`: rejected, "Tilt to remove the reflection"** |

Under the mean, this form goes downstream with a field hidden under the reflection. Under the gate,
the user retakes it in two seconds.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Unweighted mean (today) | Averages away a single fatal defect |
| Weighted mean | Same problem, with weights that are harder to justify |
| Geometric or power mean | Softer than the minimum, but the user still can't tell which defect caused a rejection |
| Learned classifier | A good next step once we have labeled captures; the gate stays the explainable fallback |

---

## Calibration

The anchors above are illustrative. Before release:

1. Collect real captures from the target channels (phone app, kiosk) and document types (cheques,
   constats, claim forms, invoices).
2. Label each one by its **downstream outcome**: extraction succeeded, or needed correction or
   manual review.
3. For each check, plot the score against the outcome. Pick $\tau_i$ so the **false-accept rate**
   stays under the business target (e.g. ≤ 2 %), then check how many retakes that causes.
4. False accepts are the costly error: a manual review, a delayed claim, a returned cheque. False
   rejects cost a few seconds of retaking. Set thresholds with that asymmetry in mind.

---

## Summary

| Check | Catches | Recoverable downstream? | Raw value | Retake message |
|---|---|---|---|---|
| Sharpness | Hand shake, focus falloff | No | $p_{10}$ of tile $r_t$ | "Hold still / hold parallel" |
| Shadows | Shadows from the user's hand or phone | Partly | $p_5(B)/p_{95}(B)$ | "Move to even light" |
| Glare | Reflections, overexposure | No | Largest saturated blob, % of area | "Tilt to remove the reflection" |
| Contrast | Faint ink, low-light noise | No | Min cell $d'$ (min-channel) | "Find more light" |
| Framing and resolution | Cut-off edges, document too far away | No | In-frame gate, worst-side px/mm | "Fit the whole document / move closer" |
| ~~OCR confidence~~ | Removed: handled by the downstream OCR | — | — | — |
| ~~Skew~~ | Removed: fixed by rectification and downstream deskew | Yes | Reported only | — |

**Open question:** the checks run today in Python/OpenCV on a server. Running them on the phone,
including on live preview frames, means porting them to the mobile OpenCV SDK or to native code,
and measuring the time budget on low-end devices.
