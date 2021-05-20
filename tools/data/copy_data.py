from os import makedirs
from os.path import exists, join, isfile
from shutil import copyfile, copytree
from argparse import ArgumentParser

from tqdm import tqdm


def load_sources(file_path, root_dir):
    sources = []
    with open(file_path) as input_stream:
        for line in input_stream:
            parts = line.strip().split(' ')
            if len(parts) != 7:
                continue

            rel_path = parts[0]
            abs_path = join(root_dir, rel_path)
            assert exists(abs_path)

            sources.append((abs_path, rel_path))

    return sources


def copy_data(sources, out_dir):
    for src_path, rel_path in tqdm(sources, leave=False):
        dst_path = join(out_dir, rel_path)

        if isfile(src_path):
            copyfile(src_path, dst_path)
        else:
            copytree(src_path, dst_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--annot', '-a', type=str, required=True)
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.annot)
    assert exists(args.input_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    sources = load_sources(args.annot, args.input_dir)
    print(f'Loaded {len(sources)} records')

    copy_data(sources, args.output_dir)
    print('Done')


if __name__ == '__main__':
    main()
