# Scan Quality Scores

This document defines the scores used to **accept or reject a document photo at capture time**. It
targets phone and kiosk captures for insurance and banking: cheques, accident reports, claim forms,
invoices and supporting documents. A rejected photo is never sent downstream; the user is asked to
retake it. Accepted photos go to a dedicated OCR and extraction pipeline.

Each section gives the formula, how to read it, a worked example, and the alternatives we
considered.

> **Status.** Implemented in [`src/scan_quality/scoring/`](../src/scan_quality/scoring/) (v0.2.0).
> The anchors were set against **synthetic test captures** (see the measurements in each section).
> They have not yet been checked against real phone photos and **must be calibrated before
> release** (see [Calibration](#calibration)).

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
5. **Keep it fast and self-contained.** All checks are pixel statistics with no OCR and no external
   binary, so they can eventually run on the device.

### Removed checks

**OCR confidence** was removed:
- The final OCR runs downstream and is better than Tesseract.
- It was about 10× slower than the other checks combined (638 ms out of 682 ms per request).
- It needed the Tesseract binary and language packs.
- Its confidence depended on the language as much as on the image.

**Skew** was removed: rotation is fixed by rectification and by the downstream deskew. The angle is
still reported (see [Framing](#5-framing-and-resolution)).

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
| 50 | `unknown` | Nothing to measure (sharpness on a page with no text); neutral |

Several checks share two building blocks:
- **Ink mask:** an adaptive threshold (block 31 px, C = 10), followed by a 2 × 2 opening that
  removes isolated noise pixels.
- **Background estimate $B$:** a 15 × 15 closing that removes the ink, followed by a 51 × 51 median
  blur.

---

## Step 0: Document detection and rectification

Module: [`document.py`](../src/scan_quality/scoring/document.py). This step isn't a score, but every
score below depends on it.

1. **Find the paper.** Measure the surface brightness as the 25th percentile of a 2 % band around
   the frame border. Measure the paper brightness as the 99th percentile of the whole frame. Every
   pixel brighter than this threshold counts as paper:

$$
\text{threshold} = \text{surface} + 0.2 \cdot (\text{paper} - \text{surface})
$$

   The threshold sits low on purpose. A page half in deep shadow is still much brighter than the
   table under it, and a threshold half-way up (where Otsu tends to place it) would cut the
   shadowed half off. If paper and surface differ by less than 30 levels, the page can't be told
   apart from what it lies on.
2. **Find the corners.** Take the largest paper region, which must cover at least 15 % of the
   frame, and simplify its convex outline to a polygon. Only a clean quadrilateral counts.
3. **Flatten.** Warp the quad onto a rectangle of the document's **physical size** at
   **8 px/mm** (about 200 DPI). If the photo has more detail than that, shrink it first with area
   averaging, because warping point-samples and would otherwise create fake edges. Then trim
   **1 mm** off every edge, where the paper still blends with the surface.

| Document type (`doc_type`) | Physical size | Rectified size |
|---|---|---|
| `cheque` (French format) | 175 × 80 mm | 1384 × 624 px |
| `a4` (claim form, constat, invoice) | 210 × 297 mm | 1664 × 2360 px |

All checks run on this rectified image $D$, in color. A **PDF page** is a render, not a photo, so
the whole page is used as the document and detection is skipped.

If no document is found, the capture is rejected and only the framing check is reported, with the
message *"Fit the whole document in the frame, on a darker plain surface."* There is no fallback to
the full frame: the other checks would be measuring the table.

---

## 1. Sharpness

Module: [`sharpness.py`](../src/scan_quality/scoring/sharpness.py). Replaces the global Laplacian
variance (15 → 0, 600 → 100).

### Formula

A single global measurement fails on phone captures in two ways:
- **Focus falloff.** The phone focuses on one part of the page, and the far edge, where the cheque's
  amount or signature often sits, is blurred. A global variance averages that away.
- **Content density.** A cheque is mostly blank and a claim form is dense. The same blur gives very
  different global variances on the two.

So sharpness is measured **per tile, normalized by contrast**, and scored on the **worst part of
the text**:

1. Split grayscale $D$ into 64 × 64 px tiles (8 × 8 mm). Keep only **text tiles**, where at least
   3 % of pixels are ink.
2. Smooth lightly with a Gaussian ($\sigma$ = 1 px) to get $\tilde D$. For each text tile, compute:

$$
r_t = \frac{\operatorname{Var}\left(\nabla^2 \tilde D_t\right)}{\operatorname{Var}\left(\tilde D_t\right) + \varepsilon}
$$

3. The raw value is the **10th percentile** over text tiles. With 1 to 9 text tiles, use the
   median instead:

$$
R = p_{10}\left(\{ r_t \}\right)
$$

4. Score it with a log ramp: **0.005 → 0** and **0.10 → 100**.

With no text tile at all, the score is a neutral **50** with label `unknown`. A page with no
readable ink is rejected by the contrast check instead, with the right message.

### Interpretation

- $r_t$ is the share of a tile's variation that sits in fine detail. Dividing by the intensity
  variance makes it independent of how dark the ink is, so pale ink isn't mistaken for blur.
- **Why the 1 px pre-smoothing:** the bare 3 × 3 Laplacian amplifies sensor grain about 20×. On a
  grainy or washed-out photo it would measure the noise, not the text; without the smoothing, a
  heavily blurred page scored as sharp once noise was added. One pixel of smoothing removes the
  noise, and any blur worth rejecting (σ ≥ 1.5 px, i.e. 0.2 mm) stays measurable.
- $R$ reads as *"how sharp is the blurriest tenth of the text?"*. The 10th percentile, rather than
  the minimum, stops a single stamp or fold from rejecting a good capture.

### Measured on synthetic pages (sensor noise σ = 3)

| Page | $R$ | Score |
|---|---|---|
| Clean | 0.128 | 100 |
| Slight blur (σ ≈ 0.8 px, 0.1 mm) | 0.085 | 94.6 |
| Moderate blur (σ ≈ 1.7 px, 0.2 mm) | 0.023 | 50.7 |
| Heavy blur (σ ≈ 3.8 px, 0.5 mm) | 0.002 | 0 |
| Right half blurred only | 0.018 | 42.4, rejected |
| Same noise at σ = 8, or ink washed out to 25 % | ≈ 0.13 | 100 (unchanged) |

### Example

A user photographs a cheque at an angle. The phone focuses on the payee line on the left, and the
amount box on the right is soft:

| Statistic | Value | Score |
|---|---|---|
| Median $r_t$ (what a global measure sees) | 0.09 | $100 \cdot \log_{10}(18)/\log_{10}(20) = 96.5$, `excellent` |
| $p_{10}$ of $r_t$ (used) | 0.012 | $100 \cdot \log_{10}(2.4)/\log_{10}(20) = 29.2$, **`poor`** |

A global score would accept a cheque whose amount can't be read. The tiled score rejects it with
*"Hold the phone still and parallel to the document."*

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Global Laplacian variance (before) | Misses focus falloff; depends on content density |
| Tile ratio without pre-smoothing | Measured noise instead of blur on grainy photos |
| Tenengrad / Brenner per tile | Good alternatives; they depend on contrast unless normalized the same way |
| Edge width (Marziliano) | Gives physical units, but is harder to make robust on handwriting |

---

## 2. Shadows

Module: [`shadows.py`](../src/scan_quality/scoring/shadows.py). Replaces the 8 × 8 cell-brightness
spread, which also responded to layout: a clean page scored about 75.

### Formula

1. Estimate the paper background $B$ of grayscale $D$, with the ink removed (see Shared
   conventions).
2. The raw value is the **shading ratio**:

$$
S = \frac{p_{5}(B)}{p_{95}(B)}
$$

3. Score it with a linear ramp: **0.40 → 0** and **0.80 → 100**.

### Interpretation

- $S$ is the share of the brightest region's light that the darkest region receives.
- A ratio doesn't depend on exposure: a dim room with even light scores 100.
- The anchors are **lenient** on purpose. A soft gradient can be divided out downstream. A deep
  shadow costs ink contrast, and the contrast check measures that directly. This check mainly
  tells the user *why*.
- The median keeps a hard shadow edge sharp. A blurred edge would leave a band of mis-corrected
  paper, and the contrast check would read it as faint ink.
- **Caveat:** large colored security backgrounds on cheques, and black logo blocks, read as
  shading. Calibrate on your real document stock.

### Measured on synthetic pages

| Page | $S$ | Score |
|---|---|---|
| Even light | 0.996 | 100 |
| Lower half at 75 % light | 0.751 | 87.8 |
| Side light losing 55 % | 0.496 | 24.0, rejected |
| Lower half at 45 % light | 0.458 | 14.5, rejected |

### Example

An accident report photographed on a car seat, with the user's arm shadowing the lower half. The
paper measures 230 in the light and 110 in the shadow:

$$
S = \frac{110}{230} = 0.478 \qquad \text{score} = 100 \cdot \frac{0.478 - 0.40}{0.40} = 19.6 \;\rightarrow\; \texttt{poor}
$$

The capture is rejected with *"Move to even light and avoid casting a shadow on the document."*

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Cell-mean standard deviation (before) | Responds to text density, not just to light |
| Background standard deviation | Depends on exposure; harder to read than a ratio |
| Fitted brightness plane | Only catches linear gradients, not a hard shadow edge |

---

## 3. Glare

Module: [`glare.py`](../src/scan_quality/scoring/glare.py). New: the old lighting check averaged a
glare spot into one grid cell.

### Formula

Glare destroys the information underneath it and can't be corrected afterwards, so this check is
strict:

1. A pixel is **saturated** when $\min(R, G, B) \ge 250$. Apply a 3 × 3 opening and keep blobs of
   at least 20 px.
2. The raw value is the **largest blob** as a share of the document's area:

$$
G = \frac{\max_k \lvert \text{blob}_k \rvert}{\lvert D \rvert}
$$

3. Score it with a reversed linear ramp: **1.0 % → 0** and **0.05 % → 100**.

### Interpretation

- We use the largest blob, not the total: one solid spot hides a word, while scattered specks hide
  nothing.
- At 8 px/mm, 1 % of a cheque is about 12 × 12 mm, enough to hide the amount in figures. 0.05 % is
  about 2.5 × 2.5 mm, smaller than one handwritten digit.
- Strongly overexposed paper also saturates and counts as glare. That is intended, but see the
  risk under [Calibration](#calibration).
- **Not implemented yet: zone rule for cheques.** This rule would reject any glare over the amount,
  the CMC7 line or the signature, regardless of size. It needs the zone coordinates of the cheque
  layouts we accept.

### Example

A laminated card photographed under an office light, with a 100 × 60 px reflection on a
1400 × 640 px document:

$$
G = \frac{6\,000}{896\,000} = 0.670\,\% \qquad \text{score} = 100 \cdot \frac{1.0 - 0.670}{1.0 - 0.05} = 34.8 \;\rightarrow\; \texttt{poor}
$$

The capture is rejected with *"Tilt the phone slightly to remove the reflection."*

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Total saturated area | Penalizes harmless specks as much as one blob hiding a word |
| HSV rule (high V, low S) | Similar result; the min-channel rule is simpler |
| Glare detection model | More robust on holograms, but needs training data |

---

## 4. Contrast

Module: [`contrast.py`](../src/scan_quality/scoring/contrast.py). Replaces RMS contrast, which
penalized cheques for being mostly blank paper.

### Formula

1. **Min-channel image** $M = \min(R, G, B)$. Paper is bright in every channel, but every colored
   ink is dark in at least one. A pale red stamp keeps its real contrast instead of fading into
   the paper as it does in luminance.
2. **Flat-field correction** $F = 255 \cdot M / B_M$, where $B_M$ is the background of $M$. This
   removes shadows so they aren't counted twice. The noise they amplify still counts.
3. Split $F$ into ink $i$ and paper $p$ with the ink mask, on a **4 × 4 grid**. A cell is measured
   when it has at least 200 ink pixels and 200 paper pixels. In each cell, compute two values from
   the paper noise $\sigma_p = \max(1.4826 \cdot \mathrm{MAD}(p),\ 2)$:

$$
\text{separation} = \frac{\operatorname{med}(p) - \operatorname{med}(i)}{\sigma_p}
\qquad
\text{depth} = \frac{\operatorname{med}(p) - \operatorname{med}(i)}{\operatorname{med}(p)}
$$

4. Take the **worst cell** for each value, score each with a linear ramp (separation **2 → 0**,
   **6 → 100**; depth **0.08 → 0**, **0.25 → 100**), and keep the **lower** of the two scores.
   The raw value is the worst separation; the worst depth is in `details`.

No measurable ink scores **0**.

### Interpretation

- **Separation** asks *"how many noise widths apart are ink and paper?"*. At 2σ, a threshold
  misclassifies about 16 % of pixels and strokes break up. Past 6σ, binarization is clean.
- **Depth** is Weber contrast: how much darker the ink is than the paper. It is needed because a
  very faint ink on a clean photo can be statistically separable and still be lost to the phone's
  JPEG compression or the downstream binarizer. On the synthetic pages, rectification averaged
  away enough grain that an 11-level gap passed on separation alone.
- **Why the paper's noise and not a pooled ink-and-paper deviation (d′):** the ink class is
  dominated by anti-aliased stroke edges. Their spread grows with contrast itself, so a pooled
  ratio barely moved when the page was washed out.
- Medians and MAD make the statistics robust to the odd pixel the mask gets wrong. The worst cell
  catches a faded corner or a noisy shadowed strip wherever it is.
- Only ink and paper levels are used, not how much ink there is, so a cheque with a few lines
  scores the same as a dense page.

### Example

A pale red "VU" stamp, in RGB (235, 190, 190), on paper at 230 with noise σ = 8:

| Channel | Ink level | Separation | Depth | Score |
|---|---|---|---|---|
| Luminance | 203.5 | 26.5 / 8 = 3.31 → 32.8 | 0.115 → 20.7 | **20.7, rejected** |
| Min-channel (used) | 190 | 40 / 8 = 5.0 → 75.0 | 0.174 → 55.2 | **55.2, accepted** |

The stamp can be read. In luminance, the capture would have been rejected because of the ink's
color, not because of the photo.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| RMS contrast (before) | Depends on ink coverage; penalizes cheques |
| Pooled d′ | Hardly reacts to washing out (see above) |
| Separation alone | Accepted near-invisible ink on low-noise photos |
| Depth alone | Accepts a dark but very grainy low-light photo |

---

## 5. Framing and resolution

Module: [`framing.py`](../src/scan_quality/scoring/framing.py). Replaces the skew angle.

### Formula

1. **Found.** No document detected → score **0**.
2. **In frame.** Any corner closer than 1 % of the frame size to the frame edge → score **0**, with
   the message *"Fit the whole document inside the frame."* This rule doesn't apply to PDF renders.
3. **Resolution.** For each side of the detected quad in the original photo, divide its length in
   pixels by its physical length in mm. Keep the **worst side**:

$$
\rho = \min_k \frac{\text{length}_k\ (\text{px})}{\text{length}_k\ (\text{mm})}
\qquad
\text{score} = \text{linear}(\rho;\ 4,\ 8)
$$

   **4 px/mm (≈ 100 DPI) → 0** and **8 px/mm (≈ 200 DPI) → 100**. Below the threshold, the message
   is *"Move closer to the document."*

The rotation of the top edge is reported in `details.rotation_deg` but isn't scored. A 90° or 180°
turned page isn't detected and is left to the downstream pipeline.

### Interpretation

- $\rho$ is the resolution of the worst-captured part of the document, an upper bound on the detail
  available there.
- Perspective is penalized only through what it costs: fewer pixels on the far side (the minimum
  side) and focus falloff (sharpness).
- Raise the 100 anchor if extraction must read small print, such as policy terms.

### Example

A cheque photographed from too far away. The far edge spans 1 100 px over 175 mm:

$$
\rho = \frac{1\,100}{175} = 6.29\ \text{px/mm} \qquad \text{score} = 100 \cdot \frac{6.29 - 4}{4} = 57.1 \;\rightarrow\; \texttt{fair}
$$

The capture is accepted. If the user had cropped off the signature corner, it would score 0 and be
rejected, however sharp it was.

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Skew angle (before) | Recoverable downstream; penalizing it rejects good captures |
| Separate perspective ratio | Duplicates what $\rho$ and sharpness already measure |
| Total megapixels | Says nothing about how much of the photo is the document |

---

## Accept or reject decision

Module: [`pipeline.py`](../src/scan_quality/scoring/pipeline.py). Replaces the unweighted mean.

### Formula

$$
\text{accept} \iff s_i \ge \tau_i \quad \text{for every check } i
\qquad\qquad
\text{overall} = \min_i\, s_i
$$

- $\tau_i$ defaults to 50 for every check (`ACCEPT_THRESHOLDS`).
- The response lists `failed_checks` worst first. The worst one's message becomes `retake_message`.
- **Retry budget (app side):** after 3 rejected attempts, the app accepts the best capture so far
  and flags it for manual review. The API is stateless, so the app counts the attempts and keeps
  the best capture.

### Interpretation

The mean answers *"how good is this photo on average?"*. A gate must answer *"is anything bad enough
to break extraction?"*, and only the minimum does that.

### Example

A claim form photographed under a ceiling light:

| Check | Score |
|---|---|
| Framing and resolution | 85.00 |
| Sharpness | 88.00 |
| Shadows | 82.00 |
| Glare | **34.78** |
| Contrast | 90.00 |
| Mean (before) | 75.96, `good`: accepted |
| **Minimum (now)** | **34.78, `poor`: rejected, "Tilt the phone slightly to remove the reflection."** |

### Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Unweighted mean (before) | Averages away a single fatal defect |
| Weighted mean | Same problem, with weights that are harder to justify |
| Geometric or power mean | Softer than the minimum, and doesn't say which defect caused a rejection |
| Learned classifier | A good next step once we have labeled captures |

---

## Calibration

All anchors were set against synthetic captures from [`tests/fixtures.py`](../tests/fixtures.py):
rendered text with controlled blur, shadows, glare, washing out and sensor noise. Real phones add
demosaicing, sharpening, denoising and JPEG compression, which will move the raw values. Before
release:

1. Collect real captures from the target channels and document types (cheques, constats, claim
   forms, invoices).
2. Label each one by its **downstream outcome**: extracted fine, or needed correction or manual
   review.
3. For each check, read the `raw_value`s and pick $\tau_i$ and the anchors so the **false-accept
   rate** stays under the business target (e.g. ≤ 2 %). Then check how many retakes that causes.
4. False accepts are the costly error: a manual review, a delayed claim, a returned cheque. False
   rejects cost a few seconds of retaking.

Risks to check first on real photos:
- **Glare vs. auto-exposure.** A white page on a dark table can make the phone overexpose until the
  paper clips at 255. The glare check would then see one huge blob. If that is common, require a
  blob to be brighter than the paper around it.
- **Detection on busy surfaces.** Wood grain, patterned seats and a hand holding the page can merge
  with the paper region. A lightweight edge or corner model may be needed.
- **Sharpness anchors.** Phone sharpening raises $r_t$. The 0.10 ceiling may need to move up.

---

## Summary

| Check | Catches | Raw value | Retake message |
|---|---|---|---|
| Framing and resolution | Missing, cut-off or too-distant document | Worst-side px/mm | "Fit the whole document…" / "Move closer…" |
| Sharpness | Hand shake, focus falloff | $p_{10}$ of tile $r_t$ | "Hold the phone still and parallel…" |
| Shadows | Shadows from the user's hand or phone | $p_5(B)/p_{95}(B)$ | "Move to even light…" |
| Glare | Reflections, overexposure | Largest saturated blob, % of area | "Tilt the phone slightly…" |
| Contrast | Faint ink, low-light noise | Worst-cell separation (+ depth) | "Find more light…" |

**Performance:** about 0.5 s per A4 photo (2 000 × 2 850 px) on a laptop CPU; a PDF page takes about
0.35 s. Running on the phone, including on live preview frames, means porting to the mobile OpenCV
SDK or native code and measuring on low-end devices.
