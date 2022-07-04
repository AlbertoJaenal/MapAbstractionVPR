import numpy as np
import os, argparse, pickle

from expectationMaximizationModel import expectationMaximization
from geometry import SE2Poses, Rotation2, combine, expSE2
from plot_map import plot_map
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topologicalfilter on trials")
    parser.add_argument(
        "-p",
        "--poses_path",
        type=str,
        help="Path to the txt file where SE2 poses are stored [x, y, theta].",
    )
    parser.add_argument(
        "-f",
        "--features_path",
        type=str,
        help="Path to the npy file where features are stored.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="",
        help="Path to the npy file where features are stored.",
    )
    parser.add_argument(
        "-c",
        "--cluster_number",
        type=int,
        help="Number of clusters of the map.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--epsilon",
        type=int,
        default=10,
        help="Epsilon for convergence.",
    )
    args = parser.parse_args()

    # Load and check format
    poses = np.loadtxt(args.poses_path)
    feats = np.load(args.features_path)
    
    assert (len(poses) == len(feats)), "Poses and feats should have the same length (%d, %d)" % (len(poses), len(feats))
    assert (poses.shape[1] == 3), "Poses are not in SE2"
    print("Loaded poses with shape", poses.shape, "and features with shape", feats.shape)
    
    # Set poses to SE2
    poses_SE2 = SE2Poses(poses[:, :2],  Rotation2.from_euler('z', poses[:, -1]))
    
    # Create and run model
    model = expectationMaximization(poses_SE2, feats, args.cluster_number)
    model.initialize('kmeans')
    model.run(args.max_iterations, converged_epsilon=args.epsilon)
    model.get_cluster_params(drop=True)
    
    # Saving
    blob_pose_centers, blob_feat_centers, blob_parameters = [], [], []
    
    for iseg, cluster in enumerate(model.clusters): 
        if cluster['prior'] <= 1e-6: continue
        blob_pose_centers.append(cluster['mu_pose'])
        blob_feat_centers.append(cluster['mu_feat'])
        blob_parameters.append(np.concatenate([cluster['cov_matrix'].flatten().squeeze(),
                                               [cluster['prior']]]))

    # Path name
    if args.output_path == "":
        if not os.path.exists("./output/"): os.makedirs("./output/")
        outputpath = f"./output/map_{args.cluster_number}"
    else:
        outputpath = args.output_path 
    
    print(f'Saving at {outputpath}')

    with open(outputpath + ".pickle", "wb") as f:
        pickle.dump({
            'pose_centers': combine(blob_pose_centers), 
            'feat_centers': np.array(blob_feat_centers, dtype=np.float32), 
            'parameters': np.array(blob_parameters, dtype=np.float32)
                      }, f)
                      
    # Plot map
    plot_map(model, 
             plot_text=True, 
             path_to_save=outputpath + ".png")