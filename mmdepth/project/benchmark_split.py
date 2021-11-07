import os

if __name__ == '__main__':
    path = "data/kitti"
    train_split = "benchmark_train_files.txt"
    val_split = "benchmark_val_files.txt"
    eigen_split = "eigen_benchmark_test_raw.txt"

    train_split_out = "benchmark_train.txt"
    val_split_out = "benchmark_val.txt"
    val_split_out = "benchmark_val_subset.txt"
    eigen_out = "eigen_benchmark_test.txt"

    with open(os.path.join(path, eigen_split)) as f:
        text_lines = f.readlines()
    
    for id, line in enumerate(text_lines):
        path_str, num, side = line.split(" ")
        folder_main, folder_sub = path_str.split("/")
        side = side[:-1]

        if side == "r":
            side_str = "image_03"
        else:
            side_str = "image_02"

        num_str = str(num).zfill(10)
        image = path_str + "/" + side_str + "/data/" + num_str + ".png"
        depth = folder_sub + '/proj_depth/groundtruth/' + side_str + "/" + num_str + ".png"
        
        line_out = image + " " + depth

        # if id % 12 == 1:
        with open(os.path.join(path, eigen_out), "a+") as f:
            f.write(line_out + "\n")
            f.close()

        print(id)

    
    
    
    
    



