import json
from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser
from collections import defaultdict
from random import shuffle


CLASS_MAP = {
    'digit_0': 0,
    'digit_1': 1,
    'digit_1_hand_to_the_camera': 1,
    'digit_2': 2,
    'digit_2_hand_to_the_camera': 2,
    'digit_3': 3,
    'digit_3_hand_to_the_camera': 3,
    'digit_3_middle_fingers': 3,
    'digit_3_middle_fingers_hand_to_the_camera': 3,
    'digit_3_with_big_finger': 3,
    'digit_3_with_big_finger_hand_to_the_camera': 3,
    'digit_4': 4,
    'digit_4_hand_to_the_camera': 4,
    'digit_5': 5,
    'digit_5_hand_to_the_camera': 5,
    'thumb_up': 6,
    'thumb_down': 7,
    'sliding_two_fingers_up': 8,
    'sliding_two_fingers_down': 9,
    'sliding_two_fingers_left': 10,
    'sliding_two_fingers_right': 11,
}


def load_videos_info(file_path):
    with open(file_path) as input_stream:
        data = json.load(input_stream)

    out_data = {k.lower(): v for k, v in data.items()}

    return out_data


def update_samples_info(records, file_path, class_map):
    out_records = []
    with open(file_path) as input_stream:
        for line in input_stream:
            line_parts = line.strip().split(' ')
            if len(line_parts) != 7:
                continue

            name, _, _, _, _, _, fps = line_parts
            name = name.lower()
            fps = max(5.0, min(30.0, float(fps)))

            assert name in records, f'Cannot find \"{name}\" in records'
            record = records[name]

            video_annot = record['annot']
            if video_annot is None:
                continue

            assert video_annot['clip_start'] >= video_annot['video_start']
            assert video_annot['clip_end'] <= video_annot['video_end']

            record['video_start'] = video_annot['video_start']
            record['video_end'] = video_annot['video_end']
            record['clip_start'] = video_annot['clip_start']
            record['clip_end'] = video_annot['clip_end']
            record['fps'] = fps
            del record['annot']

            label = record['label']
            assert label in class_map, f'Cannot find {label} in class_map'
            record['label'] = class_map[label]

            record['name'] = name
            out_records.append(record)

    return out_records


def split_train_val(records, test_ratio):
    data_by_id = defaultdict(list)
    for record in records:
        data_by_id[record['user_id']].append(record)

    num_all_ids = len(data_by_id)
    num_test_ids = max(1, int(test_ratio * float(num_all_ids)))
    assert 0 < num_test_ids < num_all_ids

    all_ids = list(data_by_id.keys())
    shuffle(all_ids)

    test_ids = set(all_ids[:num_test_ids])

    train_records, test_records = [], []
    for user_id, user_records in data_by_id.items():
        if user_id in test_ids:
            test_records.extend(user_records)
        else:
            train_records.extend(user_records)

    return train_records, test_records


def dump_annot(records, out_path):
    with open(out_path, 'w') as output_stream:
        for record in records:
            name = record['name']
            label = record['label']
            fps = record['fps']
            video_start = record['video_start']
            video_end = record['video_end']
            clip_start = record['clip_start']
            clip_end = record['clip_end']

            output_stream.write(f'{name} {label} {clip_start} {clip_end} {video_start} {video_end} {fps}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--videos_info', '-iv', type=str, required=True)
    parser.add_argument('--samples_info', '-is', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--test_ratio', '-t', type=float, required=False, default=0.2)
    args = parser.parse_args()

    assert exists(args.videos_info)
    assert exists(args.samples_info)
    if not exists(args.output_dir):
        makedirs(args.output_dir)

    records = load_videos_info(args.videos_info)
    print(f'Loaded {len(records)} video records')

    records = update_samples_info(records, args.samples_info, CLASS_MAP)
    print(f'Merged {len(records)} final records')

    train_records, test_records = split_train_val(records, test_ratio=args.test_ratio)
    print(f'Split on {len(train_records)} train and {len(test_records)} test records')

    train_out_path = join(args.output_dir, 'train.txt')
    dump_annot(train_records, train_out_path)
    print(f'Train annotation is dumped to: {train_out_path}')

    test_out_path = join(args.output_dir, 'test.txt')
    dump_annot(test_records, test_out_path)
    print(f'Test annotation is dumped to: {test_out_path}')


if __name__ == '__main__':
    main()