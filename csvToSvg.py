import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# Read CSV file
def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Columns in CSV file:", df.columns.tolist())  # Print column names for debugging
        
        required_columns = ['polyline_index', 'x', 'y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {', '.join(missing_columns)}")
        
        segments = []
        for _, group in df.groupby('polyline_index'):
            segment = group[['x', 'y']].values
            segments.append(segment)
        return segments
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

# Canny edge detection
def canny_edge_detection(image_path, output_csv_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read the image file {image_path}")
    img = cv2.GaussianBlur(img, (5, 5), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(img, 50, 60)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    data = []
    polyline_index = 0
    for contour in contours:
        polyline = contour[:, 0, :]
        for point in polyline:
            x, y = point
            data.append([polyline_index, 0, x, y])
        polyline_index += 1
    
    df = pd.DataFrame(data, columns=['polyline_index', 'unused_col', 'x', 'y'])
    df.to_csv(output_csv_path, index=False)

# Sobel edge detection
def sobel_edge_detection(image_path, output_csv_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read the image file {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)
    _, binary_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    data = []
    polyline_index = 0
    for contour in contours:
        polyline = contour[:, 0, :]
        for point in polyline:
            x, y = point
            data.append([polyline_index, 0, x, y])
        polyline_index += 1
    
    df = pd.DataFrame(data, columns=['polyline_index', 'unused_col', 'x', 'y'])
    df.to_csv(output_csv_path, index=False)

# Helper functions
def points_are_close(p1, p2, tol=1e-5):
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return distance < tol

def calculate_slope(p1, p2):
    if p2[0] == p1[0]:
        return np.inf
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def calculate_angle(slope):
    return np.arctan(slope) * (180 / np.pi)

def angle_difference(angle1, angle2):
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def split_polylines_to_disjoint(polylines):
    lines = [poly.tolist() for polyline in polylines for poly in polyline]

    while True:
        modified = False
        for i in range(len(lines)):
            cur_line = lines[i]
            if len(cur_line) < 4:
                continue
            for j in range(len(cur_line) - 2):
                p1, p2, p3 = cur_line[j], cur_line[j+1], cur_line[j+2]
                angle_diff = angle_difference(calculate_angle(calculate_slope(p1, p2)), calculate_angle(calculate_slope(p2, p3)))
                if angle_diff >= 20:
                    new_line_1 = cur_line[:j+2]
                    new_line_2 = cur_line[j+1:]
                    if len(new_line_1) < 4:
                        e1 = 0.9
                    else:
                        e1, _, _ = fit_line(np.array(new_line_1))
                    if len(new_line_2) < 4:
                        e2 = 0.9
                    else:
                        e2, _, _ = fit_line(np.array(new_line_2))
                    if e1 < 5 and e2 < 5:
                        lines[i] = new_line_1
                        lines.insert(i+1, new_line_2)
                        modified = True
                        break
            if modified:
                break
        
        if not modified:
            break
    
    disjoint_polylines = []
    lines = [LineString(np.array(polyline)) for polyline in lines if len(polyline) > 1]
    disjoint_polylines = []
    pairs_checked = set()

    while True:
        modified = False
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                if (i, j) not in pairs_checked and lines[i].intersects(lines[j]) and not lines[i].touches(lines[j]):
                    pairs_checked.add((i, j))
                    intersection = lines[i].intersection(lines[j])
                    new_i = lines[i].difference(intersection)
                    new_j = lines[j].difference(intersection)
                    lines[i] = new_i
                    lines[j] = new_j
                    lines.append(intersection)
                    modified = True
                    print(f"Splitting lines {i} and {j}")
                    break
            if modified:
                break
        
        if not modified:
            break
    
    for line in lines:
        if line.geom_type == 'MultiLineString':
            for l in line.geoms:
                disjoint_polylines.append(np.array(l.coords))
        elif line.geom_type == 'LineString':
            disjoint_polylines.append(np.array(line.coords))

    return disjoint_polylines

def count_close_points(polylines, point, tolerance=1e-5):
    count = -1
    for polyline in polylines:
        if points_are_close(point, polyline[0], tolerance) or points_are_close(point, polyline[-1], tolerance):
            count += 1

    print(f"Close points to {point}: {count}")
    return count

def merge_close_points_if_unique(extended_polylines, tolerance=1e-5):
    while True:
        merged_polylines = []
        visited = set()
        merge_occurred = False

        for i, polyline_i in enumerate(extended_polylines):
            if i in visited:
                continue
            
            merged = False
            for j, polyline_j in enumerate(extended_polylines):
                if i != j and j not in visited:
                    start_i, end_i = polyline_i[0], polyline_i[-1]
                    start_j, end_j = polyline_j[0], polyline_j[-1]

                    if points_are_close(start_i, start_j, tolerance) and count_close_points(extended_polylines, start_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_j[::-1], polyline_i[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(start_i, end_j, tolerance) and count_close_points(extended_polylines, start_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_j, polyline_i[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(end_i, start_j, tolerance) and count_close_points(extended_polylines, end_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_i, polyline_j[1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
                    elif points_are_close(end_i, end_j, tolerance) and count_close_points(extended_polylines, end_i, tolerance) == 1:
                        new_polyline = np.vstack([polyline_i, polyline_j[::-1][1:]])
                        merged_polylines.append(new_polyline)
                        visited.add(i)
                        visited.add(j)
                        merged = True
                        merge_occurred = True
                        break
            if merged:
                break
        
        if not merge_occurred:
            break
    
    return merged_polylines

def fit_line(points):
    if len(points) < 2:
        raise ValueError("At least two points are required to fit a line.")
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if points.shape[1] != 2:
        raise ValueError("Points should be a Nx2 array.")
    
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    predicted_y = m * x + c
    residuals = y - predicted_y
    mse = np.mean(residuals ** 2)
    
    return mse, m, c

def visualize_polylines(polylines, title='Polylines'):
    plt.figure(figsize=(10, 10))
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], marker='o')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def process_image(image_path, canny_output_path, sobel_output_path):
    canny_edge_detection(image_path, canny_output_path)
    sobel_edge_detection(image_path, sobel_output_path)

def main():
    image_path = 'path/to/your/image.jpg'
    canny_output_path = 'path/to/your/canny_output.csv'
    sobel_output_path = 'path/to/your/sobel_output.csv'
    
    process_image(image_path, canny_output_path, sobel_output_path)
    
    print("Canny and Sobel edge detection outputs generated.")

    canny_segments = read_csv(canny_output_path)
    sobel_segments = read_csv(sobel_output_path)

    disjoint_canny_polylines = split_polylines_to_disjoint(canny_segments)
    disjoint_sobel_polylines = split_polylines_to_disjoint(sobel_segments)

    merged_canny_polylines = merge_close_points_if_unique(disjoint_canny_polylines)
    merged_sobel_polylines = merge_close_points_if_unique(disjoint_sobel_polylines)

    visualize_polylines(merged_canny_polylines, 'Merged Canny Polylines')
    visualize_polylines(merged_sobel_polylines, 'Merged Sobel Polylines')

if __name__ == '__main__':
    main()
