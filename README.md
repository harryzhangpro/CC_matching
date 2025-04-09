# CC_matching
A CC-based point cloud library 

### Usage (Model)

use 1.pcd & 2.pcd with standard.pcd as 

generate first matrix
```
python pd_gicp.py 0407/1.pcd 0407/standard.pcd 0407/1.txt
mv transformation_matrix.npy npys/1.npy
```

generate second matrix
```
python pd_gicp.py 0407/2.pcd 0407/standard.pcd 0407/2.txt
mv transformation_matrix.npy npys/2.npy
```

combine matrix
```
python npys/combine_matrices.py npys/1.npy npys/2.npy
```

measure result
```
python pd_lines.py --pair 0407/1.pcd 0407/2.pcd