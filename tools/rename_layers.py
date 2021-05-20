from os.path import exists
from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.input)

    model_checkpoint = torch.load(args.input,  map_location='cpu')
    assert 'state_dict' in model_checkpoint

    state_dict = model_checkpoint['state_dict']

    replacements = []
    for old_key in list(state_dict.keys()):
        if 'fc_pre_cls' in old_key:
            new_key = old_key.replace('fc_pre_cls', 'fc_pre_angular')
            state_dict[new_key] = state_dict.pop(old_key)
            replacements.append((old_key, new_key))

    torch.save(model_checkpoint, args.output)

    if len(replacements) > 0:
        print(f'Replacements ({len(replacements)}):')
        for old_key, new_key in replacements:
            print(f'   {old_key} --> {new_key}')


if __name__ == '__main__':
    main()
