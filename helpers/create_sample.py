import os
import shutil
import argparse
import random
import csv

def create_sample(dir, size=200):
  # create sample dir
  if dir[-1] == '/' or dir[-1] == '\\':
    dir = dir[:-1] # remove last slash
  data_dir = os.path.dirname(dir)
  dataset_name = os.path.basename(dir)
  sample_dir = os.path.join(*[data_dir, dataset_name + '_sample'])
  if os.path.exists(sample_dir):
    shutil.rmtree(sample_dir)
  os.mkdir(sample_dir)

  # process sub dirs which are labels
  dir_list = get_subdirectories(dir)
  for label in dir_list:
    if label.startswith('.') or label == 'models':
      continue

    # make sample_subdir to save sample random files
    sample_subdir = os.path.join(*[sample_dir, label])
    if not os.path.exists(sample_subdir):
      os.mkdir(sample_subdir)

    # get sample random files
    ori_subdir = os.path.join(*[dir, label])
    files = os.listdir(ori_subdir)
    sample_size = size if len(files) >= size else len(files)
    sample_files = random.sample(files, sample_size)

    for sample_file in sample_files:
      copy_from = os.path.join(*[ori_subdir, sample_file])
      copy_to = os.path.join(*[sample_subdir, sample_file])
      try:
        shutil.copy(copy_from, copy_to)
      except Exception as e:
        print(e)


# https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_subdirectories(data_directory):
  return [name for name in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory, name))]


parser = argparse.ArgumentParser(
  description="Copy portion of the dataset to create sample dataset, so we can experiment faster")
parser.add_argument("--data", type=str, default=os.getcwd(), help="Path to data, default=cwd")
parser.add_argument("--size", type=int, default=200, help="Sample size per label, default=200")
args = parser.parse_args()

create_sample(args.data, args.size)
