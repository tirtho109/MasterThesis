import sys
import os
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
from fbpinns.decompositions import RectangularDecompositionND

def  get_subdomain_xsws(time_limit=[0,24], t_begin=0, t_end=24, num_subdomain=2, ww=1.1, w_noDataRegion=1.1):
    data_start, data_end = time_limit
    subdomains = []
    centers = []

    # no data region
    has_no_data_at_start = data_start > t_begin
    has_no_data_at_end = data_end < t_end

    if has_no_data_at_start:
        subdomains.append((t_begin, data_start))
        centers.append((t_begin + data_start)/2)
    if has_no_data_at_end:
        subdomains.append((data_end, t_end))
        centers.append((data_end + t_end)/2)
    
    # remaining_subdomains
    remaining_subdomains = num_subdomain - len(subdomains)

    if remaining_subdomains>0:
        each_subdomain_length = (data_end - data_start)/remaining_subdomains
        for i in range(remaining_subdomains):
            start = data_start + i * each_subdomain_length
            end = start + each_subdomain_length
            subdomains.append((start, end))
            centers.append((start+end)/2)

    else:
        subdomains.append((data_start, data_end))
        centers.append((data_start+data_end)/2)

    subdomain_xs = [np.array(np.sort(centers))]
    subdomain_ws = [np.array([np.diff(sub)* (ww if time_limit[0] <= cen <= time_limit[1] else w_noDataRegion) for cen, sub in sorted(zip(centers, subdomains))]).flatten()]

    return subdomain_xs, subdomain_ws


if __name__=="__main__":

    import os

    # Define the folder name
    folder_name = "Decomposition"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Specify the file path for saving the figure
    path1 = os.path.join(folder_name, "1.png")
    path2 = os.path.join(folder_name, "2.png")
    path3 = os.path.join(folder_name, "3.png")
    
    num_subdomain = 3
    ww = 2.0
    t_begin, t_end = 0, 24
    decomposition = RectangularDecompositionND

    # Case 1: time_limit = [0,24]
    time_limit = [0, 24]
    subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin, t_end, num_subdomain, ww)
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)
    f = decomposition.plot(all_params, active=active, show_norm=True, show_window=True, create_fig=True)
    f.suptitle(f"{time_limit}")
    f.savefig(path1)

    # Case 2: time_limit = [0, 10]
    time_limit = [0, 10]
    subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin, t_end, num_subdomain, ww)
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)
    active[-1] = 0
    f = decomposition.plot(all_params, active=active, show_norm=True, show_window=True, create_fig=True)
    f.suptitle(f"{time_limit}")
    f.savefig(path2)

    # Case 2: time_limit = [10, 24]
    time_limit = [10, 24]
    subdomain_xs, subdomain_ws = get_subdomain_xsws(time_limit, t_begin, t_end, num_subdomain, ww)
    ps_ = decomposition.init_params(subdomain_xs, subdomain_ws, (0,1))
    all_params = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}}
    m = all_params["static"]["decomposition"]["m"]
    active = np.ones(m)
    active[0] = 0
    f = decomposition.plot(all_params, active=active, show_norm=True, show_window=True, create_fig=True)
    f.suptitle(f"{time_limit}")
    f.savefig(path3)

