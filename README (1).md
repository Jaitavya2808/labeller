# Labellerr AI — Robotics Growth Engineer Intern Take-Home

## Summary Card
- **Dataset access** — 10 first-person POV clips used as a substitute source (fill in: exact source — e.g. YouTube search terms — and why you chose it). Confidence: 5. → `clips.txt`
- **Hand-labeled clips** — 2 clips labeled: a barista pouring latte art, and a pizza maker preparing and baking a pizza. Labels are (start_time, end_time, action, object) rows. Confidence: 5. → `labels/clip_01_labels.csv`, `labels/clip_02_labels.csv`
- **Automation note** — Pipeline (segmentation → detection/tracking → action recognition → merge), human-in-the-loop points, and time/cost math grounded in the two labeled clips. Confidence: 5. → `automation_note.md`
- **Script (optional)** — Clip-length/category summary stats, plus an optional CLIP zero-shot per-frame action-labeling demo. Confidence: 5. → `analyze_clips.py`

## Time log
- Task 1 (dataset access): ~__ min
- Task 2 (hand-labeling): ~__ min
- Task 3 (automation note): ~__ min
- Task 4 (script, optional): ~__ min
- Repo/README/reflection: ~__ min

## Running the code
```bash
./setup.sh
source .venv/bin/activate
./run.sh summary
./run.sh autolabel clip_01.mp4   # optional CLIP zero-shot demo
```

## Reflection

**What would you improve about your submission if you had two more hours?**
*If I had two more hours, I would improve the automation pipeline by testing an actual vision-language model for generating initial action and object labels instead of relying primarily on manual annotation. I would also add stronger validation for timestamps and overlapping actions so that the generated labels could be reviewed more systematically. Finally, I would expand the analysis of the 10 clips and compare the automatically generated labels against my manual annotations to measure how accurately the approach performs.*


**What's one thing about this task you didn't already know how to do, and how did you figure it out?**
*One thing I had not worked with before was the process of converting egocentric video into structured action and object annotations with timestamps. I first studied the expected annotation format from the assignment and then examined the clips carefully to identify meaningful actions and the objects involved. I also researched existing approaches for video understanding and labeling, which helped me understand how foundation models, object detection, segmentation, and human review could be combined into a scalable annotation pipeline.*