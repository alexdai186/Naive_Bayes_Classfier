import numpy as np
import random
from PIL import Image

SINGLE_VALIDATION = False
WRITE_TO_FILE = True

NOISE_FILTER_THRESH = 100
LINE_SEG_PIXEL_DEV = 3
MIN_VALID_PATH_LEN = 3
LONG_TAIL = 0.90

"""
This is your object classifier. You should implement the train and
classify methods for this assignment.
"""
class ObjectClassifier():
    labels = ['Tree', 'Sydney', 'Steve', 'Cube']
    
    """
    Everytime a snapshot is taken, this method is called and
    the result is displayed on top of the four-image panel.
    """
    def classify(self, edge_pixels, orientations):
        self.log("------ Classifying Objects ------")
        segs = self.get_segments_from_edges(edge_pixels)
        objs = self.get_objects(segs, false, None)

        return ", ".join(objs)
    
    """
    This is your training method. Feel free to change the
    definition to take a directory name or whatever else you
    like. The load_image (below) function may be helpful in
    reading in each image from your datasets.
    """
    def train(self):
        self.log("------ Training the Program ------")
        path = "./snapshots/"
        images = {}
        i = 0
        num_correct = 0

        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if i < 9000:
                    self.log("Loading image " + f)
                    images[f] = load_image(path + f)
                    i = i+1
        f = None
        if WRITE_TO_FILE:
            f = open("results.txt", "w")

        for image in images:
            self.log("------ Parsing Image " + image + " ------")
            image_list = images[image]
            edge_arr = image_list[0]
            segs = self.get_segments_from_edges(edge_arr)
            objs = self.get_objects(segs, true, image)

    """
    Logging methods
    """
    def log(self, text):
        self.log_text.append(text)

    """
    Identify Objects
    """
    def get_objects(self, segs, training, image):
        x_ranges = self.find_obj_x_ranges(segs)
        objs = []
        obj_num = 1
        for r in x_ranges:
            self.log("------ Identifying Object: %s of %s ------" % (obj_num, len(x_ranges)))
            obj_num += 1
            range_segs = self.find_segs_in_range(r, segs)
            total_seg_len = self.len_of_all_segs(range_segs)
            if len(range_segs) <= 1:
                self.log("Only contains one path, object disregarded.")
            elif total_seg_len < 50:
                self.log("Segments not long enough, object disregarded.")
            else:
                if training:
                    best_guess = self.single_obj_classifier(image, range_segs)
                    if WRITE_TO_FILE:
                        f.write("Image: %s | Classification: %s\n" % (image, best_guess))
                    if SINGLE_VALIDATION:
                        if best_guess.lower() in image.lower():
                            num_correct += 1
                else:
                    best_guess = self.single_obj_classifier("Live Image ", range_segs)
                    objs.append(best_guess)
            self.log("\n")
        if training and SINGLE_VALIDATION:
            self.log("------ Results ------")
            res_str = "Correctly IDed %s/%s objs." % (num_correct, len(images))
            if WRITE_TO_FILE:
                f.write(res_str)
                f.close()
            self.log(res_str)
        return objs

    """
    Finding Line Segments
    """
    def get_segments_from_edges(self, edge_arr):
        edge_arr = edge_arr[270:]
        edge_arr = edge_arr[1:-1]

        for y in range(0, len(edge_arr)):
            row = edge_arr[y]
            row[0] = 0
            row[-1] = 0
        edge_arr = self.noise_reduct_edge_arr(edge_arr)
        paths = self.find_all_paths(edge_arr, [])
        path_str = ""

        for path in paths:
            path_str += self.path_str_from_arr(path)

        sorted_paths = self.remove_tail(sorted(paths, key=len, reverse=True), LONG_TAIL)
        all_segs = []
        all_segs_str = ""
        for path in sorted_paths:
            path_segs = self.find_segs_from_path(path, [])
            all_segs_str += self.path_segs_str(path_segs)
            for path_seg in path_segs:
                all_segs.append(path_seg)

        f = open('segments_string.txt', 'w')
        f.write(all_segs_str)
        f.close()
        return all_segs

    def noise_reduct_edge_arr(self, arr):
        edge_arr = []
        for y in range(0, len(arr)):
            y_arr = array[y]
            for x in range(0, len(y_arr)):
                pixel_val = array[y][x]
                if pixel_val > NOISE_FILTER_THRESH:
                    edge_arr.append([y,x])
        return edge_arr

    def find_all_paths(self, edge_arr, path_arr):
        if len(edge_arr) > 0:
            edge_arr, path = self.path_from_edge_arr(edge_arr, edge_arr[0], [])
            path_arr.append(path)
            return self.find_all_paths(edge_arr, path_arr)
        return (edge_arr, path)

    def path_str_from_arr(self, path_arr):
        path_str = ""
        for edge in path_arr:
            y = 600-edge[0]
            x = edge[1]
            path_str += "(%s,%s)" % (x,y)
        return path_str

    def remove_tail(self, sorted_paths, n):
        length = 0
        filtered_paths = []
        for path in sorted_paths:
            if len(path) < MIN_VALID_PATH_LEN:
                sorted_paths.remove(path)
            else:
                length += len(path)
        filter_len = n * length
        total_path_len = 0
        for path in sorted_paths:
            if total_path_len < filter_len:
                filtered_paths.append(path)
                total_path_len += len(path)
        return filtered_paths

    def find_segs_from_path(self, path, seg_arr):
        init_pt = path[0]
        seg = []
        for i in range(0, len(path)):
            coordinate = path[i]
            if self.is_line_seg(seg, coordinate):
                seg.append(coordinate)
            else:
                seg_arr.append([init_pt, coordinate])
                return self.find_segs_from_path(path[i:], seg_arr)
        seg_arr.append([init_pt, path[-1]])
        return seg_arr

    def path_segs_str(self, path_segs):
        path_segs_list = []
        path_segs_list.append(path_segs[0][0])
        for line_seg in path_segs:
            path_segs_list.append(line_seg[1])
        return self.path_str_from_arr(path_segs_list)

    def path_from_edge_arr(self, edge_arr, coord, path):
        neighbors = self.neighbors_from_edge(coord, edge_arr)
        for edge in neighbors:
            path.append(edge)
            edge_arr.remove(edge)
        for edge in neighbors:
            edge_neighbors = self.neighbors_from_edge(edge, edge_arr)
            if len(edge_neighbors) > 0:
                return self.path_from_edge_arr(edge_arr, edge, path)
        return (edge_arr, path)

    def is_line_seg(self, seg, coord):
        if len(seg) == 0:
            return True
        init_pt = seg[0]
        for point in seg:
            distance = self.dist(init_pt[0], init_pt[1], coord[0], coord[1], point[0], point[1])
            if distance > LINE_SEG_PIXEL_DEV:
                return False
        return True

    def neighbors_from_edge(self, coord, edge_arr):
        y = coord[0]
        x = coord[1]
        neighbors = self.get_neighbors(x, y)
        valid_neighbors = []
        for neighbor in neighbors:
            if neighbor in edge_arr:
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def get_neighbors(self, x, y):
        neighbors = []
        neighbors.append([y-1, x-1])
        neighbors.append([y, x-1])
        neighbors.append([y+1, x-1])
        neighbors.append([y-1, x])
        neighbors.append([y, x])
        neighbors.append([y+1, x])
        neighbors.append([y-1, x+1])
        neighbors.append([y, x+1])
        neighbors.append([y+1, x+1])
        return neighbors

    def dist(self, x1, y1, x2, y2, x3, y3):
        px = x2 - x1
        py = y2 - y1
        z = px * px + py * py
        u = ((x3 - x1) * px + (y3 - y1) * py) / float(z)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3
        dist = math.sqrt(dx * dx + dy * dy)

        return dist

    def find_obj_x_ranges(self, segs):
        x_values = []
        for seg in segs:
            a = seg[0][1]
            b = seg[1][1]
            for i in range(min(a, b), max(a, b) + 1):
                x_values.append(i)
        x_values = list(set(x_values))
        ranges = self.find_ranges(x_values, [])
        return ranges

    def find_ranges(self, x_values, range_arr):
        if len(x_values) > 0:
            init_pt = 0
            for i in range(0, 800):
                if i in x_values:
                    init_pt = i
                    break
            end_pt = init_pt
            max_gap = 2 * MIN_VALID_PATH_LEN + 1
            gap = 0
            while end_pt in x_values or gap < max_gap:
                if end_pt in x_values:
                    x_values.remove(end_pt)
                    gap = 0
                else:
                    gap += 1
                end_pt += 1
            range_arr.append([init_pt, end_pt-1])
            return self.find_ranges(x_values, range_arr)
        return range_arr

    def find_segs_in_range(self, range, segs):
        min_x = range[0]
        max_x = range[1]
        range_segs = []
        for seg in segs:
            a = seg[0][1]
            b = seg[1][1]
            min_seg_x = min(a,b)
            max_seg_x = max(a,b)
            if min_seg_x >= min_x and max_seg_x <= max_x:
                range_segs.append(seg)
        return range_segs

    def len_of_all_segs(self, range_segs):
        total_len = 0
        for seg in range_segs:
            total_len += self.length(seg)
        return total_len

    def length(self, seg):
        x1 = seg[0][0]
        y1 = seg[0][1]
        x2 = seg[1][0]
        y2 = seg[1][1]
        value = math.pow(x2-x1, 2) + math.pow(y2-y1, 2)
        return math.sqrt(value)

"""
Loads an image from file and calculates the edge pixel orientations.
Returns a tuple of (edge pixels, pixel orientations).
"""
def load_image(filename):
    im = Image.open(filename)
    np_edges = np.array(im)
    upper_left = push(np_edges, 1, 1)
    upper_center = push(np_edges, 1, 0)
    upper_right = push(np_edges, 1, -1)
    mid_left = push(np_edges, 0, 1)
    mid_right = push(np_edges, 0, -1)
    lower_left = push(np_edges, -1, 1)
    lower_center = push(np_edges, -1, 0)
    lower_right = push(np_edges, -1, -1)
    vfunc = np.vectorize(find_orientation)
    orientations = vfunc(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right)
    return (np_edges, orientations)

        
"""
Shifts the rows and columns of an array, putting zeros in any empty spaces
and truncating any values that overflow
"""
def push(np_array, rows, columns):
    result = np.zeros((np_array.shape[0],np_array.shape[1]))
    if rows > 0:
        if columns > 0:
            result[rows:,columns:] = np_array[:-rows,:-columns]
        elif columns < 0:
            result[rows:,:columns] = np_array[:-rows,-columns:]
        else:
            result[rows:,:] = np_array[:-rows,:]
    elif rows < 0:
        if columns > 0:
            result[:rows,columns:] = np_array[-rows:,:-columns]
        elif columns < 0:
            result[:rows,:columns] = np_array[-rows:,-columns:]
        else:
            result[:rows,:] = np_array[-rows:,:]
    else:
        if columns > 0:
            result[:,columns:] = np_array[:,:-columns]
        elif columns < 0:
            result[:,:columns] = np_array[:,-columns:]
        else:
            result[:,:] = np_array[:,:]
    return result

# The orientations that an edge pixel may have.
np_orientation = np.array([0,315,45,270,90,225,180,135])

"""
Finds the (approximate) orientation of an edge pixel.
"""
def find_orientation(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right):
    a = np.array([upper_center, upper_left, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right])
    return np_orientation[a.argmax()]