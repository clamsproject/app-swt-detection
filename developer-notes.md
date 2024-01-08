
### Developer notes

To test the code without running a Flask server you can call the classify script directly:

```bash
python -m modeling.classify \
	--config modeling/config/classifier.yml \
	--input modeling/data/cpb-aacip-690722078b2-0500-0530.mp4 \
	--debug
```
