<h2 align="center">Neural Preset: Faithful 4K Color Style Transfer in Real Time</h2>

<div align="center"><i>Neural Preset for Color Style Transfer (CVPR 2023)</i></div>

---

## Metrics
We provide code/model of the metrics we used in our paper. The following are the commands to calculate the metrics. Please refer to the code for details.

### Style Similiary Metric
```
cd src/metric/style_similiary
calc_style_similiary.py --result-folder ../test_data/result --style-folder ../test_data/style
```

### Content Similiary Metric
```
cd src/metric/content_similiary
calc_content_similiary.py --result-folder ../test_data/result --content-folder ../test_data/content
```
