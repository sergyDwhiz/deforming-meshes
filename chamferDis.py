import numpy as np

def chamfer_distance(A, B):
    ret = 0
    closest_points = []
    for a in range(A.shape[0]):
        min_distance = float('inf')
        closest_point = None
        for b in range(B.shape[0]):
            distance = np.linalg.norm(A[a] - B[b])
            if distance < min_distance:
                min_distance = distance
                closest_point = b
        ret += min_distance
        closest_points.append((a, A.shape[0]+closest_point))

    closest_points_vertices = np.vstack([A, B])
    closest_points_indices = np.array(closest_points)

    assert closest_points_indices.shape[0] == A.shape[0] and closest_points_indices.shape[1] == 2
    assert closest_points_vertices.shape[0] == A.shape[0] + B.shape[0] and closest_points_vertices.shape[1] == A.shape[1]

    print("chamfer distance is: ", ret)

    return ret, closest_points_vertices, closest_points_indices
