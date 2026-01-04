ZIP_FILE_PATH="ml-32m"
ML_SMALL="ml-1m"

# Download ml-1m if not present
if [ ! -f "$ML_SMALL" ]; then
  echo "Downloading ml-1m dataset..."
  wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O ml-1m.zip
  unzip ml-1m.zip
else
  echo "ml-1m.zip already exists."
fi

# Download ml-32m if not present
if [ ! -f "$ZIP_FILE_PATH" ]; then
  echo "Downloading ml-32m dataset..."
  wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
  unzip ml-32m.zip
else
  echo "ml-32m.zip already exists."
fi
