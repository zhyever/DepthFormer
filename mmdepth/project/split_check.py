import os

if __name__ == '__main__':
    path = "data/kitti"
    eigen_split = "eigen_benchmark_test.txt"
    val_split_out = "benchmark_val.txt"
    final_out = "eigen_benchmark_test_final.txt"

    with open(os.path.join(path, eigen_split)) as f:
        eigen_split = f.readlines()

    with open(os.path.join(path, val_split_out)) as f:
        val_split_out = f.readlines()
    
    print(len(eigen_split))
    print(len(val_split_out))
    for id, line in enumerate(eigen_split):
        check_flag = False
        for val in val_split_out:
            if line[:-1] == val[:-1]:
                check_flag = True
                break

        if check_flag == True:
            print(id)
            with open(os.path.join(path, final_out), "a+") as f:
                f.write(line[:-1] + "\n")
                f.close()

    
    
    
    
    



