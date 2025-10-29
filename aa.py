import shutil
import os

# Path to the persist directory
law_persist_directory = "path/to/law_vector_store"

# Check if the directory exists, then delete it
if os.path.exists(law_persist_directory):
    shutil.rmtree(law_persist_directory)
    print(f"Deleted the directory: {law_persist_directory}")
else:
    print(f"Directory does not exist: {law_persist_directory}")
