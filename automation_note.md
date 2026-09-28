# Automating Egocentric Video Labeling at Scale

## Pipeline
1. **Shot / activity segmentation** — split long recordings into candidate
   action segments using scene-change or motion-based cut detection
   (e.g. PySceneDetect) before any labeling happens, so humans and models
   both work on pre-chunked clips instead of raw hour-long footage.
2. **Object detection & tracking** — an open-vocabulary detector
   (YOLO-World, Grounding DINO) run per-frame or per-segment to identify
   candidate objects in view, tracked across frames (ByteTrack/DeepSORT)
   to reduce duplicate labels for the same object.
3. **Action / verb recognition** — either (a) a video-specific action
   recognition model fine-tuned on a small labeled seed set (e.g. a
   lightweight I3D/X3D head), or (b) a general video-language model
   (VideoLLaMA2, InternVideo2) or a frame-sampled prompt to a multimodal
   LLM (GPT-4V/Gemini) asked to describe "what action is happening,"
   which trades per-call cost for zero fine-tuning.
4. **Merge into label format** — auto-generated (timestamp, action,
   object) triples matching the same schema as the hand labels, so
   downstream training/eval code doesn't care which labels came from
   which source.

## Grounded in this task's clips
The two hand-labeled clips illustrate exactly where this pipeline would
succeed and struggle:
- **Latte-pouring clip**: a generic action-recognition vocabulary would
  likely collapse "pours to create latte art" and "pours heavily through
  surface" into a single "pours liquid" label — the *manner* of pouring
  (steady vs. heavy, creating a pattern vs. filling) is the meaningful
  distinction here, and that's a fine-grained motion cue current
  open-vocabulary detectors don't capture well. This is a case for
  routing to human review rather than trusting the model's default
  granularity.
- **Pizza-making clip**: object detection handles "dough," "sauce,"
  "cheese" and "pepperoni" reasonably well since they're visually
  distinct, but distinguishing *which* topping is being spread (cheese
  vs. pepperoni) needs fine-grained classification, not just detection.
  The "checking the base" action (repeatedly lifting a corner of the
  pizza with a peel) is also a good example of a domain-specific action
  with no obvious entry in a generic action vocabulary — it would need
  either a custom fine-tuned class or a human-reviewed catch-all label.

## Where humans stay in the loop
- **Verification, not generation**: reviewers accept/correct/reject
  machine-proposed labels rather than writing from scratch — much faster
  per clip.
- **Ambiguous or novel actions**: anything the model is low-confidence on,
  or actions outside the seed vocabulary, get routed to a human queue.
- **Vocabulary drift**: periodically spot-check a random sample even from
  "high confidence" auto-labels, since confidence scores can be
  miscalibrated on out-of-distribution footage.
- **Edge cases**: occlusion, multiple simultaneous actions, and unusual
  camera motion are exactly where auto-labelers fail most, so these are
  flagged for manual pass by simple heuristics (e.g. rapid camera motion,
  low detection confidence).

## Time / cost estimate
- Manual hand-labeling (from this task): roughly **10 minutes per
  short clip** for careful (start, end, action, object) annotation.
- With auto-label-then-verify: reviewers mostly click accept/reject
  rather than writing labels, which typically runs **5-10x faster** per
  clip than labeling from scratch — call it **~1 to 1.7 minutes**
  per clip once the pipeline is warmed up.
- At scale (e.g. 10,000 clips), that's the difference between roughly
  **1,667 labeler-hours** (fully manual, 10 min/clip) and
  **~208 labeler-hours** (verify-only, ~1.25 min/clip), before
  accounting for model inference cost, which is typically a small
  fraction of labor cost at this scale. That's roughly an **87-88%
  reduction** in labeling labor.
- The real gate isn't compute — it's building a good enough seed model
  to make verification faster than labeling from scratch, which usually
  needs a few hundred to a few thousand hand-labeled examples first.
