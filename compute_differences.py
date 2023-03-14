import pickle
import numpy as np
import os
import re
from params import get_params


def compute_distances(logdir, transforms, metrics):
    distances               = []
    batch_count             = -1 #number of batches for which the optimization is executed.

    for transform in transforms:
        print("====================")
        print(str(transform))
        subdir              = os.path.join(logdir, '_'.join(np.array(transform, dtype='str')))

        if not os.path.exists(subdir):
            print("Missing file: ", subdir)
            continue

        if batch_count < 0:
            # ASSUMING YOU HAVE RUN ALL CORRUPTIONS FOR EACH DATASET, i.e., batches used are the same.
            # Used for assigning unique IDs per data sample
            subfilenames    = [name for name in os.listdir(subdir) if 'batch' in name and not os.path.isdir(name)]
            subbatchids     = [int(re.findall('^batch(.+)_.+', subfilenames[i])[0]) for i in range(len(subfilenames)) if len(re.findall('^batch(.+)_.+', subfilenames[i]))>0]
            batch_count     = max(subbatchids)+1

        for batch in range(batch_count):
            #read actual in/out, optimized in/out and target images to compute distances
            transform_in    = np.transpose(np.load(os.path.join(subdir, 'batch' + str(batch) + '_transformin.npy')), (2, 1, 0))
            projected_in    = np.transpose(np.load(os.path.join(subdir, 'batch' + str(batch) + '_projectedin.npy')), (2, 1, 0))
            transform_pred  = np.load(os.path.join(subdir, 'batch' + str(batch) + '_transformpred.npy'))
            projected_pred  = np.load(os.path.join(subdir, 'batch' + str(batch) + '_projectedpred.npy'))
            transform_prob  = np.load(os.path.join(subdir, 'batch' + str(batch) + '_transformprob.npy'))
            projected_prob  = np.load(os.path.join(subdir, 'batch' + str(batch) + '_projectedprob.npy'))
            target          = np.load(os.path.join( subdir, 'batch' + str(batch) + '_target.npy'))

            distance_per_input = [[] for j in range(len(transform_in))]

            for metric in metrics.keys():
                compute_between = metrics[metric]['compute']
                for i in compute_between:
                    # compute:
                    # 0: difference between inputs (d_x),
                    # 1: difference between predicted output classes (d_y),
                    # 2: difference between transformed output probability and target (d_g),
                    # 3: difference between projected output probability and target (d^_g)
                    # -1: change in performance (d_g - d^_g)

                    if i == 0:
                        input1      = transform_in
                        input2      = projected_in
                    if i == 1:
                        input1      = transform_prob
                        input2      = projected_prob
                    if i == 2:
                        input1      = transform_prob
                        input2      = target
                    if i == 3:
                        input1      = projected_prob
                        input2      = target

                    for j in range(len(transform_in)):
                        if i == -1:
                            distance_per_input[j].append(distance_per_input[j][-2] - distance_per_input[j][-1])
                        else:
                            distance_per_input[j].append(metrics[metric]['function'](input2[j], input1[j]))

            #compute distances, CHANGE as needed.
            for subid in range(len(transform_in)):
                #print(batch, len(transform_in) * batch + subid)
                writelist   = [subdir]
                writelist.extend(list(transform))
                writelist.append(len(transform_in) * batch + subid)
                writelist.extend(distance_per_input[subid])
                distances.append(writelist)

    return distances

if __name__ == "__main__":

    params                  = get_params()
    logdir                  = 'outputs' #path to outputs_cache from input optimization (main.py)
    transforms              = params["transforms"]["values"] #transforms T

    #compute and save distances
    distances               = compute_distances(logdir, transforms, params["differences_metrics"])
    pickle.dump(distances, open(os.path.join(logdir, params["differences_fn"]), 'wb'))