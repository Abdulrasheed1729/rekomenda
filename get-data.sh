ZIP_FILE_PATH="ml-32m.zip"
ML_SMALL="ml-10k.zip"

# Download ml-10k if not present
if [ ! -f "$ML_SMALL" ]; then
  echo "Downloading ml-10k dataset..."
  wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O ml-10k.zip
  unzip ml-10k.zip
else
  echo "ml-10k.zip already exists."
fi

# Download ml-32m if not present
if [ ! -f "$ZIP_FILE_PATH" ]; then
  echo "Downloading ml-32m dataset..."
  wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
  unzip ml-32m.zip
else
  echo "ml-32m.zip already exists."
fi
