import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

subprocess.run(["streamlit", "run", "app.py"])
