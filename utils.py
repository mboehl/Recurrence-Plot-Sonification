import numpy as np

def half_diagonals_as_columns(M, dtype=np.float32):
    """
    M = [[ 0  1  2  3  4]
        [10 11 12 13 14]
        [20 21 22 23 24]
        [30 31 32 33 34]
        [40 41 42 43 44]]

    -> [[ 0 11 22 33 44]
        [ 0  2 13 24  0]
        [ 0  0  4  0  0]
        [ 0  0  0  0  0]
        [ 0  0  0  0  0]]
    """
    n = M.shape[0]
    mod_M = np.zeros((int((n+1)/2), n), dtype=dtype)
    for I in range(n):
        for J in range(min(I + 1, n - I)):
            mod_M[J][I] = M[I - J][I + J]
    
    return mod_M    

def upper_triangle_aligned(M):
    n = M.shape[0]
    return np.array([np.append(row[i:], np.zeros(i)) for i, row in enumerate(M)])

def discretize(signal, nvalues=1):
    stretch = nvalues / np.max(np.abs(signal))
    return np.vectorize(lambda x: np.round(x * stretch) / stretch)(signal)

def dist_to_diag(i, j, N):
    """Return "supremum norm" distance of cell `(i, j)` to the diagonal of an `(N x N)`-matrix."""
    return abs(i - j)

def squish_matrix(M, factor=2):
    return np.array([np.mean(M[i:i+factor], axis=0) for i in range(int(len(M) / factor))])

def normalize(signal):
    return signal / np.max(np.abs(signal))

def matrix_to_qimage(M, cmap="binary"):
    from matplotlib import cm
    from PySide6.QtGui import QImage, QPixmap

    norm = cm.colors.Normalize(vmin=np.min(M), vmax=np.max(M))
    colormap = cm.get_cmap(cmap)
    colored_M = colormap(norm(M))[:, :, :3]  # Get RGB values, ignore alpha channel
    colored_M = (colored_M * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    height, width, _ = colored_M.shape
    qimage = QImage(colored_M.data, width, height, 3 * width, QImage.Format_RGB888)
    
    return qimage

def matrix_extract_parallelogram(M, Points):
    """Extract the part of the matrix M that lies within the parallelogram spanned by start and end."""
    start = (min(Points[:, 0]), min(Points[:, 1]))
    end = (max(Points[:, 0]), max(Points[:, 1]))
    height = end[1] - start[1]
    width = end[0] - start[0] - height
    
    extracted = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            extracted[i][j] = M[start[1] + i][start[0] + j + i]
    return extracted

def matrix_extract_square(M, Points):
    """Extract the part of the matrix M that lies within the square spanned by start and end."""
    start = (min(Points[:, 0]), min(Points[:, 1]))
    end = (max(Points[:, 0]), max(Points[:, 1]))
    side_length = min(end[0] - start[0], end[1] - start[1])
    
    extracted = np.zeros((side_length, side_length))
    for i in range(side_length):
        for j in range(side_length):
            try:
                extracted[i][j] = M[start[1] + i][start[0] + j]
            except IndexError:
                print("IndexError: Attempting to access M[{start[1] + i}][{start[0] + j}] but M has shape {M.shape}. Corresponding entry is set to 0 in the extracted square.")
    return extracted

def matrix_extract_tilted_square(M, Points):
    """Extract the part of the matrix M that lies within the tilted square spanned by start and end."""
    start = min(Points, key=lambda p: p[0])
    end = max(Points, key=lambda p: p[0])
    side_length = abs(end[0] - start[0]) // 2 + 1
    
    extracted = np.zeros((int(side_length), int(side_length)))
    for i in range(int(side_length)):
        for j in range(int(side_length)):
            try:
                extracted[j][i] = M[start[0] + i + j][start[1] + j - i]
            except IndexError:
                print(f"IndexError: Attempting to access M[{start[0] + i + j}][{start[1] + j - i}] but M has shape {M.shape}. Corresponding entry is set to 0 in the extracted tilted square.")
    return extracted
