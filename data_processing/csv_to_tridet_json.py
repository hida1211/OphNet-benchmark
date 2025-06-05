import argparse
import json
import pandas as pd


def convert(csv_path, json_path, default_fps=30, fps_csv=None):
    df = pd.read_csv(csv_path)
    fps_lookup = None
    if fps_csv:
        fps_df = pd.read_csv(fps_csv)
        fps_lookup = dict(zip(fps_df["file"], fps_df["fps"]))
    database = {}
    labels = sorted(df['phase_id'].unique())
    for vid, group in df.groupby('video_id'):
        subset = group['split'].iloc[0]
        duration = float(group['end'].max())
        annotations = []
        for _, row in group.iterrows():
            annotations.append({
                'segment': [float(row['start']), float(row['end'])],
                'label': str(row['phase_id']),
                'label_id': int(row['phase_id'])
            })
        database[vid] = {
            'subset': subset,
            'fps': fps_lookup.get(vid, default_fps) if fps_lookup else default_fps,
            'duration': duration,
            'annotations': annotations
        }
    out = {'database': database}
    with open(json_path, 'w') as f:
        json.dump(out, f)
    print(f"Saved {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV annotations to TriDet JSON format')
    parser.add_argument('--csv', required=True, help='annotation csv file')
    parser.add_argument('--out', required=True, help='output json file')
    parser.add_argument('--fps', type=float, default=30, help='default fps value to store in json')
    parser.add_argument('--fps-csv', help='optional CSV mapping file,fps')
    args = parser.parse_args()
    convert(args.csv, args.out, args.fps, args.fps_csv)


if __name__ == '__main__':
    main()
