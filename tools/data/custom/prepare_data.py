import json
from os import walk, makedirs
from os.path import exists, join, basename
from shutil import copyfile
from collections import defaultdict
from argparse import ArgumentParser

from lxml import etree
from tqdm import tqdm


NUM_CLASSES = 12
VIDEO_FILE_EXTENSIONS = 'webm'
XML_FRAGMENT_TEMPLATE = '<annotations>\n{}</annotations>'


def load_annot(input_dir):
    out_data = dict()
    for root, dirs, files in walk(input_dir):
        if len(files) == 0:
            continue

        local_data = defaultdict(dict)
        for f in files:
            file_name_parts = f.split('.')
            file_name = '.'.join(file_name_parts[:-1])
            file_extension = file_name_parts[-1]
            record = local_data[file_name]

            if file_extension == 'xml':
                try:
                    annot_str = repair_annot(join(root, f))
                    annot = parse_annot(annot_str)
                except:
                    annot = None

                record['annot'] = annot
                if record['annot'] is None:
                    print(f'   * invalid: {basename(root)}/{f}')
            elif file_extension in VIDEO_FILE_EXTENSIONS:
                name_components = file_name.split('-')
                assert len(name_components) == 5, f'Incorrect naming: {file_name}'

                record['label'] = name_components[4]
                record['user_id'] = name_components[3]
                record['video_name'] = f
                record['video_path'] = join(root, f)

        filtered_data = {k: v for k, v in local_data.items() if 'video_path' in v and 'annot' in v}
        out_data.update(filtered_data)

    return out_data


def repair_annot(file_path):
    content = ''
    enable_collecting = False
    with open(file_path, encoding='unicode_escape') as input_stream:
        for line in input_stream:
            if '<track id=\"0\" label=\"person\" source=\"manual\">' in line:
                enable_collecting = True
            elif '</track>' in line:
                content += line
                break

            if enable_collecting:
                content += line

    return XML_FRAGMENT_TEMPLATE.format(content)


def parse_annot(xml_fragment):
    root = etree.XML(xml_fragment.encode('utf-8'))

    tracks = []
    for element in root:
        if element.tag != 'track':
            continue

        all_frame_ids, valid_frame_ids = [], []
        for bbox in element:
            frame_id = int(bbox.attrib['frame'])
            all_frame_ids.append(frame_id)

            actions = []
            for action in bbox:
                if action.tag == 'attribute' and action.attrib['name'] == 'sign_action':
                    actions.append(action.text)
            assert len(actions) == 1,\
                f'Expected single action per frame but got {len(actions)} actions'

            action = actions[0]
            valid_frame = action == 'yes'
            if valid_frame:
                valid_frame_ids.append(frame_id)

        if len(valid_frame_ids) > 0:
            tracks.append(dict(
                video_start=min(all_frame_ids),
                video_end=max(all_frame_ids) + 1,
                clip_start=min(valid_frame_ids),
                clip_end=max(valid_frame_ids) + 1,
            ))

    if len(tracks) == 0:
        return None
    else:
        assert len(tracks) == 1, f'Expected single track per video but got {len(tracks)} tracks'
        return tracks[0]


def dump_annot(annot, out_path):
    with open(out_path, 'w') as output_stream:
        json.dump(annot, output_stream)


def copy_videos(annot, out_dir):
    for record in tqdm(annot, desc='Copying videos', leave=False):
        input_file_path = record['video_path']
        output_file_path = join(out_dir, record['video_name'])

        if not exists(output_file_path):
            copyfile(input_file_path, output_file_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.input_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    data = load_annot(args.input_dir)
    user_ids = set([record['user_id'] for record in data.values()])
    print(f'Loaded {len(data)} records ({len(user_ids)} unique users).')

    out_annot_path = join(args.output_dir, 'annot.json')
    dump_annot(data, out_annot_path)
    print(f'Annotation has been dumped to {out_annot_path}')

    out_videos_dir = join(args.output_dir, 'videos')
    if not exists(out_videos_dir):
        makedirs(out_videos_dir)
    copy_videos(data.values(), out_videos_dir)
    print(f'Videos have been copied to {out_videos_dir}')


if __name__ == '__main__':
    main()
