import shutil
import os

cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'sentence_transformers')
shutil.rmtree(cache_path, ignore_errors=True)

print(f"✅ Cleared cache at: {cache_path}")