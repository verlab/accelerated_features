"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Camera pose metrics adapted from LoFTR https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py
    The main difference is the use of poselib instead of OpenCV's vanilla RANSAC for E_mat, which is more stable and MUCH and faster.

"""

import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
import json
import multiprocessing as mp

# Disable scientific notation
np.set_printoptions(suppress=True)

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, type='poselib'):
    if len(kpts0) < 5:
        return None
    if type == 'poselib':
        import poselib
        (pose,details) = poselib.estimate_relative_pose(
            kpts0.tolist(), 
            kpts1.tolist(),
            intrinsics_to_camera(K0),
            intrinsics_to_camera(K1),
            ransac_opt={
                'max_iterations': 10000, # default 100000
                'success_prob': conf, # default 0.99999
                'max_epipolar_error': thresh, # default 1.0
            },
            bundle_opt={  # all defaults
                },
            )
        ret = (pose.R, pose.t, details['inliers'])

    elif type == 'opencv':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        norm_thresh = thresh / f_mean

        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
            method=cv2.RANSAC)

        assert E is not None

        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
    else:
        raise NotImplementedError

    return ret

def estimate_pose_parallel(args):
    return estimate_pose(*args)

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def pose_accuracy(errors, thresholds):
    return [np.mean(errors < t) * 100 for t in thresholds]

def get_relative_transform(pose0, pose1):
    R0 = pose0[..., :3, :3] # Bx3x3
    t0 = pose0[..., :3, [3]] # Bx3x1

    R1 = pose1[..., :3, :3] # Bx3x3
    t1 = pose1[..., :3, [3]] # Bx3x1
    
    R_0to1 = R1.transpose(-1, -2) @ R0 # Bx3x3
    t_0to1 = R1.transpose(-1, -2) @ (t0 - t1) # Bx3x1
    T_0to1 = np.concatenate([R_0to1, t_0to1], axis=-1) # Bx3x4

    return T_0to1

class Scannet1500:
    default_config = {
        'scannet_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ScanNet/scannet_test_1500')),
        'gt_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ScanNet/test.npz')),
        'pose_estimator': 'poselib', # poselib, opencv
        'cache_images': True,
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'pose_thresholds': [5, 10, 20],
        'max_pairs': -1,
        'output': './output/scannet/',
        'n_workers': 8,
    }

    def __init__(self, config={}) -> None:
        self.config = {**self.default_config, **config}
        if not os.path.exists(self.config['scannet_path']):
            raise RuntimeError(
                f"Dataset {self.config['scannet_path']} does not exist! \n \
                  > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")
        
        self.pairs = self.read_gt()

        os.makedirs(self.config['output'], exist_ok=True)

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        self.image_cache = {}
        if self.config['cache_images']:
            self.load_images()

    def load_images(self):
        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self.image_cache:
                self.image_cache[pair['image0']] = cv2.imread(pair['image0'])
            if pair['image1'] not in self.image_cache:
                self.image_cache[pair['image1']] = cv2.imread(pair['image1'])

    def read_image(self, path):
        if self.config['cache_images']:
            return self.image_cache[path]
        else:
            return cv2.imread(path)

    def read_gt(self):
        pairs = []
        gt_poses = np.load(self.config['gt_path'])
        names = gt_poses['name']
        
        for i in range(len(names)):
                scene_id = names[i, 0]
                scene_idx = names[i, 1]
                scene = f'scene{scene_id:04d}_{scene_idx:02d}'
            
                image0 = str(int(names[i, 2]))
                image1 = str(int(names[i, 3]))
                
                K0 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene,  'intrinsic/intrinsic_color.txt')
                )
                K1 = K0

                pose_0 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'pose', image0 + '.txt')
                )
                pose_1 = np.loadtxt(
                    os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'pose', image1 + '.txt')
                )
                T_0to1 = get_relative_transform(pose_0, pose_1)
                
                pairs.append({
                    'image0': os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'color', image0 + '.jpg'),
                    'image1': os.path.join(self.config['scannet_path'], 'scannet_test_1500', scene, 'color', image1 + '.jpg'),
                    'K0': K0,
                    'K1': K1,
                    'T_0to1': T_0to1,
                })

        return pairs

    def extract_and_save_matches(self, matcher_fn, name='', force=False):
        all_matches = []
        if name == '':
            name = matcher_fn.__name__

        fname = os.path.join(self.config['output'], f'{name}_matches.npz')
        if not force and os.path.exists(fname):
            return np.load(fname, allow_pickle=True)['all_matches']

        for pair in tqdm(self.pairs, desc='Extracting matches'):
            image0 = self.read_image(pair['image0'])
            image1 = self.read_image(pair['image1'])
            mkpts0, mkpts1 = matcher_fn(image0, image1)

            all_matches.append({
                'image0': pair['image0'],
                'image1': pair['image1'],
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
            })
        
        np.savez(fname, all_matches=all_matches)
        
        return all_matches

    def run_benchmark(self, matcher_fn, name='', force=False):
        if name == '':
            name = matcher_fn.__name__

        all_matches = self.extract_and_save_matches(matcher_fn, name=name, force=force)

        aucs_by_thresh = {}
        accs_by_thresh = {}
        for ransac_thresh in self.config['ransac_thresholds']:

            fname = os.path.join(self.config['output'], f'{name}_{self.config["pose_estimator"]}_{ransac_thresh}.txt')
            # check if exists and has the right number of lines
            if not force and os.path.exists(fname) and len(open(fname, 'r').readlines()) == len(self.pairs):
                errors = []
                with open(fname, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.replace('\n', '')
                        err_t, err_R = line.split(' ')
                        errors.append([float(err_t), float(err_R)])
            # redo the benchmark
            else:
                errors = []
                pairs = self.pairs
                errors_file = open(fname, 'w')

                # do the benchmark in parallel
                if self.config['n_workers'] != 1:

                    pool = mp.Pool(self.config['n_workers'])
                    pool_args = [ (all_matches[pair_idx]['mkpts0'], all_matches[pair_idx]['mkpts1'], pair['K0'], pair['K1'], ransac_thresh) for pair_idx, pair in enumerate(pairs) ]
                    results = list(tqdm(pool.imap(estimate_pose_parallel, pool_args), total=len(pool_args), desc=f'Running benchmark for th={ransac_thresh}', leave=False))
                    pool.close()
                    

                    for pair_idx, ret in enumerate(results):
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            pair = pairs[pair_idx]
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors.append([err_t, err_R])
                # do the benchmark in serial
                else:
                    for pair_idx, pair in tqdm(enumerate(pairs), desc=f'Running benchmark for th={ransac_thresh}', leave=False, total=len(pairs)):
                        mkpts0 = all_matches[pair_idx]['mkpts0']
                        mkpts1 = all_matches[pair_idx]['mkpts1']
                        ret = estimate_pose(mkpts0, mkpts1, pair['K0'], pair['K1'], ransac_thresh)
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors_file.flush()
                        errors.append([err_t, err_R])

                errors_file.close()

            # compute AUCs
            errors = np.array(errors)
            errors = errors.max(axis=1) 
            aucs = pose_auc(errors, self.config['pose_thresholds'])
            accs = pose_accuracy(errors, self.config['pose_thresholds'])
            aucs = {k: v*100 for k, v in zip(self.config['pose_thresholds'], aucs)}
            accs = {k: v for k, v in zip(self.config['pose_thresholds'], accs)}
            aucs_by_thresh[ransac_thresh] = aucs
            accs_by_thresh[ransac_thresh] = accs

            # dump summary for this method
            summary = {
                'name': name,
                'aucs_by_thresh': aucs_by_thresh,
                'accs_by_thresh': accs_by_thresh,
            }
            json.dump(summary, open(os.path.join(self.config['output'], f'{name}_{self.config["pose_estimator"]}_summary.json'), 'w'), indent=2)

        return aucs_by_thresh

def get_xfeat():
    from modules.xfeat import XFeat
    xfeat = XFeat()
    return xfeat.match_xfeat

def get_xfeat_star():
    from modules.xfeat import XFeat
    xfeat = XFeat(top_k=10_000)
    return xfeat.match_xfeat_star

def get_alike():
    from third_party import alike_wrapper as alike
    return alike.match_alike

def print_fancy(d):
    print(json.dumps(d, indent=2))

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--scannet_path", type=str, required=True, help="Path to the Scannet 1500 dataset")
    parser.add_argument("--output", type=str, default="./output/scannet/", help="Path to the output directory")
    parser.add_argument("--max_pairs", type=int, default=-1, help="Maximum number of pairs to run the benchmark on")
    parser.add_argument("--force", action='store_true', help="Force running the benchmark again")
    parser.add_argument("--pose_estimator", type=str, default='poselib', help="Which pose estimator to use: poselib, opencv", choices=['poselib', 'opencv'])
    
    parser.add_argument("--show", action='store_true', help="Show the matches")
    parser.add_argument("--accuracy", action='store_true', help="Show the accuracy instead of AUC")
    parser.add_argument("--filter", type=str, nargs='+', help="Filter the results by the given names")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    if not args.show:
        scannet = Scannet1500({
            'scannet_path': args.scannet_path,
            'gt_path': args.scannet_path + "/test.npz",
            'cache_images': False,
            'output': args.output,
            'max_pairs': args.max_pairs,
            'pose_estimator': args.pose_estimator,
            'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            'n_workers': 8,
        })

        functions = {
            'xfeat': get_xfeat(),
            'xfeat_star': get_xfeat_star(),
            'alike': get_alike(),
        }

        # save all results to a file
        all_results = {}

        for name, fn in functions.items():
            print(name)
            result = scannet.run_benchmark(matcher_fn = fn, name=name, force=args.force)
            all_results[name] = result

        json.dump(all_results, open(os.path.join(args.output, 'summary.json'), 'w'), indent=2)

    if args.show:
        import glob
        import pandas as pd
        dataset_name = 'scannet'
        all_summary_files = glob.glob(os.path.join(args.output, "**_summary.json"), recursive=True)

        if args.filter:
            all_summary_files = [f for f in all_summary_files if any([fil in f for fil in args.filter])]

        dfs = []
        names = []
        estimators = []
        metric_key = 'aucs_by_thresh'
        if args.accuracy:
            metric_key = 'accuracies_by_thresh'    
        for summary in all_summary_files:
            summary_data = json.load(open(summary, 'r'))
            if metric_key not in summary_data:
                continue
            aucs_by_thresh = summary_data[metric_key]

            estimator = 'poselib'
            if 'opencv' in summary:
                estimator = 'opencv'

            #make sure everything is float
            for thresh in aucs_by_thresh:
                for k in aucs_by_thresh[thresh]:
                    if isinstance(aucs_by_thresh[thresh][k], str):
                        aucs_by_thresh[thresh][k] = float(aucs_by_thresh[thresh][k].replace(' ', ''))

            # find best threshold based on the 5, 10, 20 mAP and everything is float
            df = pd.DataFrame(aucs_by_thresh).T.astype(float)
            df['mean'] = df.mean(axis=1)
            # create a string column called estimator
            cols = df.columns.tolist()
            dfs.append(df)
            names.append(summary_data['name'])
            estimators.append(estimator)

        # use each col as the main col to determine the best threshold
        # for col in cols:
        col = 'mean'

        final_df = pd.DataFrame()
        # add cols
        final_df['name'] = names
        final_df['best_thresh'] = ''
        final_df['estimator'] = estimators
        final_df[cols] = -1.0

        for df, name, estimator in zip(dfs, names, estimators):
            best_thresh = df[col].idxmax()
            best_results = df.loc[best_thresh]

            # now update the best_thresh based on the estimator
            final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), 'best_thresh'] = best_thresh
            for _col in cols:
                final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), _col] = best_results[_col]

        # sort by mean
        final_df = final_df.sort_values(by=['mean'])
        # reset index
        final_df = final_df.reset_index(drop=True)

        # drop estimator column
        final_df = final_df.drop(columns=['estimator'])

        # set max float precision to 1
        final_df = final_df.round(1)

        print(f"Dataset: {dataset_name}")
        print(f"Sorting by {col}")
        print(final_df)
        print()

        final_df.to_csv(os.path.join(args.output, f"{dataset_name}_{col}.csv"), index=False)
