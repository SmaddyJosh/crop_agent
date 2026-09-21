import argparse
from pathlib import Path
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def load_image(path, img_size):
	img = tf.keras.utils.load_img(path, target_size=img_size)
	arr = tf.keras.utils.img_to_array(img)
	arr = arr
	return np.expand_dims(arr, axis=0)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--model', default='crop_simple.keras', help='path to saved keras model')
	p.add_argument('--image', required=True, help='path to image to predict')
	p.add_argument('--classes', help='comma-separated class names (in index order)')
	p.add_argument('--classes-file', help='path to a file with one class name per line')
	p.add_argument('--img-size', type=int, default=180)
	args = p.parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		raise SystemExit(f"Model not found: {model_path}. Train and save a model first.")

	model = tf.keras.models.load_model(str(model_path))
	img = load_image(args.image, (args.img_size, args.img_size))

	preds = model.predict(img)
	idx = int(np.argmax(preds[0]))
	prob = float(np.max(preds[0]))

	# resolve class names: CLI args > model metadata > fallback to indices
	class_names = None
	if args.classes:
		class_names = [c.strip() for c in args.classes.split(',') if c.strip()]
	elif args.classes_file:
		cf = Path(args.classes_file)
		if cf.exists():
			class_names = [l.strip() for l in cf.read_text().splitlines() if l.strip()]
	else:
		# If model has metadata (unlikely), try reading
		try:
			class_names = model.class_names
		except Exception:
			class_names = None

	label = class_names[idx] if class_names and idx < len(class_names) else str(idx)

	print(f"Predicted: {label} (index={idx}) with confidence={prob:.4f}")


if __name__ == '__main__':
	main()

