from shapely.geometry import Polygon, Point
from shapely.affinity import scale
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from matplotlib.patches import Rectangle
import config
import time
import matplotlib.ticker as mticker


class LoadBoundingBox:

    def __init__(self):
        self.bounding_boxes = []
        self.num_boxes = 0
        self.load_bboxes()
        self.weed, self.crop = self.separate_class()

    def load_bboxes(self):
        self.bounding_boxes = config.yolo_v_11()
        self.num_boxes = len(self.bounding_boxes)
        return self.bounding_boxes

    def separate_class(self):
        weed = []
        crop = []
        for box in self.bounding_boxes:
            if box[0] in config.WEED_LIST:
                weed.append(box)
            elif box[0] in config.CROP_LIST:
                crop.append(box)
        return weed, crop


class UnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, i):
        while i != self.par[i]:
            self.par[i] = self.par[self.par[i]]
            i = self.par[i]
        return i

    def union(self, i, j):
        root1, root2 = self.find(i), self.find(j)
        if root1 == root2:
            return False  # already in the same group!

        if self.rank[root2] > self.rank[root1]:
            self.par[root1] = root2
            self.rank[root2] += self.rank[root1]
        else:
            self.par[root2] = root1
            self.rank[root1] += self.rank[root2]
        return True


class Poly:
    def __init__(self, vertices):
        self.polygon = Polygon(vertices)

    def add_padding(self, factor):
        if factor <= 0:
            raise ValueError(
                "Padding factor must be greater than 0")
        centroid = self.polygon.centroid
        self.polygon = scale(self.polygon, xfact=factor,
                             yfact=factor, origin=centroid)

    def area(self):
        return self.polygon.area

    def perimeter(self):
        return self.polygon.length

    def get_vertices(self):
        # list with all vertices of the polygon
        return list(self.polygon.exterior.coords)


class ManagePolygons:

    def __init__(self):
        self.polygons = []
        self.num_polygons = 0

    def add_polygon(self, polygon):
        self.polygons.append(polygon)
        self.num_polygons += 1

    def remove_polygon(self, index):
        if 0 <= index < len(self.polygons):
            del self.polygons[index]
            self.num_polygons -= 1
        else:
            raise IndexError("Index not valid!")

    def count_polygons(self):
        return self.num_polygons

    def get_all_vertices(self):
        #  return all vetices for all polygons
        return [poly.get_vertices() for poly in self.polygons]

    def get_area(self, index):
        if 0 <= index < len(self.polygons):
            return self.polygons[index].area()
        else:
            raise IndexError("Index not valid!")

    def find_intersecting_polygons(self):
        intersecting = []
        for i in range(self.num_polygons):
            for j in range(i+1, self.num_polygons):
                if self.polygons[i].polygon.intersects(self.polygons[j].polygon):
                    intersecting.append((i, j))
        return intersecting

    def find_connected_components(self):
        intersection = self.find_intersecting_polygons()
        n = self.num_polygons

        uf = UnionFind(n)

        # Merge intersecting polygons
        for i, j in intersection:
            uf.union(i, j)

        # Group polygons by connected component
        groups = {}
        for i in range(n):
            radice = uf.find(i)
            groups.setdefault(radice, []).append(i)

        return groups

    def find_intersecting_groups(self):
        n = len(self.polygons)
        uf = UnionFind(n)

        # Check intersections between polygons
        for i in range(n):
            for j in range(i + 1, n):
                if self.polygons[i].polygon.intersects(self.polygons[j].polygon):
                    uf.union(i, j)

        # Group polygons by connected component
        groups = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(self.polygons[i])

        # Merge polygons of each group and return only the external points
        merged_polygons = [
            unary_union([poly.polygon for poly in group]
                        ).exterior.coords  # type: ignore
            if unary_union([poly.polygon for poly in group]).geom_type == "Polygon"
            else None
            for group in groups.values()
        ]

        return merged_polygons

    def get_external_coordinates(self, groups):
        external_coordinates = []
        for i, group in enumerate(groups):
            external_coordinates.append(list(group))
        return external_coordinates

    def print_external_coordinates_groups(self, groups):
        for i, group in enumerate(groups):
            print(f"Group {i}: {list(group)}")

    def plot_all_polygons(self):
        fig, ax = plt.subplots()
        for poly in self.polygons:
            x, y = poly.polygon.exterior.xy
            ax.plot(x, y, 'r')
        plt.axis('equal')
        plt.grid()


class BresenhamPath:
    def __init__(self, poligoni):
        """
        Inizializza l'oggetto con una lista di poligoni.
        Ogni poligono è una lista di vertici (tuple) ordinati.
        """
        self.poligoni = poligoni

    def bresenham_line(self, p1, p2):
        """
        Implementazione dell'algoritmo di Bresenham per ottenere
        tutti i punti (coordinate intere) lungo la retta da p1 a p2.
        """
        x0, y0 = p1
        x1, y1 = p2
        punti = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            punti.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return punti

    def point_in_polygon(self, x, y, poly):
        """
        Testa se il punto (x, y) è interno al poligono 'poly'.
        Utilizza l'algoritmo del ray-casting (pari/dispari).
        I punti che cadono esattamente sui bordi verranno considerati "dentro".
        """
        num_vertici = len(poly)
        inside = False
        j = num_vertici - 1  # indice dell'ultimo vertice
        for i in range(num_vertici):
            xi, yi = poly[i]
            xj, yj = poly[j]
            # Se il punto è esattamente su una linea, lo consideriamo dentro.
            # Questo controllo può essere personalizzato se necessario.
            if ((yi == y and xi == x) or (yj == y and xj == x)):
                return True

            # Controllo del crossing del raggio orizzontale
            if ((yi > y) != (yj > y)):
                try:
                    intersezione = (xj - xi) * (y - yi) / (yj - yi) + xi
                except ZeroDivisionError:
                    intersezione = xi
                if x == intersezione:  # sul bordo
                    return True
                if x < intersezione:
                    inside = not inside
            j = i
        return inside

    def get_edge_points(self):
        """
        Per ogni poligono, calcola e restituisce l'insieme di punti (coordinate intere)
        che si trovano sulle linee che collegano i vertici.
        """
        edge_points = set()
        for poly in self.poligoni:
            # Assumiamo che il primo e l'ultimo punto siano uguali per chiudere il poligono;
            # altrimenti si può aggiungere manualmente la chiusura.
            for i in range(len(poly) - 1):
                # Converte i vertici in interi (se necessario, si può usare round())
                p1 = (int(poly[i][0]), int(poly[i][1]))
                p2 = (int(poly[i+1][0]), int(poly[i+1][1]))
                line_points = self.bresenham_line(p1, p2)
                edge_points.update(line_points)
        return edge_points

    def get_interior_points(self):

        interior_points = set()
        for poly in self.poligoni:
            # Creazione dell'oggetto Polygon di Shapely
            shapely_poly = Polygon(poly)
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    pt = Point(x, y)

                    if shapely_poly.contains(pt) or shapely_poly.touches(pt):
                        interior_points.add((x, y))
        return interior_points

    def get_all_points(self):
        """
        Restituisce l'insieme di tutti i punti (coordinate intere) che sono:
        - Toccati dalle linee (bordi) del poligono
        - O al loro interno
        """
        # Uniamo i punti bordo e interni di tutti i poligoni
        return self.get_edge_points().union(self.get_interior_points())


class Grid:
    def __init__(self, points, grid_sizeX, grid_sizeY):
        if len(grid_sizeX) < 1:
            raise ValueError("at least one value is required in grid_sizeX")

        self.points = points
        self.grid_sizeX = grid_sizeX
        self.grid_sizeY = grid_sizeY
        self.grid = self.organize_points()

    def find_block_x(self, x):

        cum_width = 0
        for i, width in enumerate(self.grid_sizeX):
            cum_width += width
            if x < cum_width:
                return i
        # if x is greater than the last block, return the last index
        return len(self.grid_sizeX) - 1

    def organize_points(self):
        block_groups = []
        current_group = []
        prev_y = None

        for x, y in self.points:
            block_x = self.find_block_x(x)
            block_y = y // self.grid_sizeY

            if prev_y is not None and y != prev_y:
                block_groups.append(current_group)
                current_group = []

            current_group.append((block_x, block_y))
            prev_y = y

        if current_group:
            block_groups.append(current_group)

        return block_groups


class NozzleActivator:
    def __init__(self, points):
        self.points = points

    def group_for_y(self):
        grouped_by_y = defaultdict(list)

        for sublist in self.points:
            for x, y in sublist:
                grouped_by_y[y].append((x, y))
        # return a list of lists where every element is a list of points with the same y value
        grouped_by_y = list(grouped_by_y.values())
        return grouped_by_y

    def binary_sequences(self):
        result = []

        for sublist in self.group_for_y():
            y_value = sublist[0][1]
            binary_sequence = ['0'] * 32

            for x, _ in sublist:
                if 0 <= x < 32:
                    binary_sequence[x] = '1'

            binary_string = ''.join(binary_sequence)
            result.append([y_value, binary_string])

        # return a list of lists  WHERE every element is (y, row value!) and the binary string
        return result


class DelayCalulator:
    def __init__(self, sequence, current_row):
        self.sequence = sequence  # sequence is List[List[row, binary string]]


if __name__ == "__main__":

    os.system('cls')
    poly0 = Poly([(1, 1), (6, 1), (6, 3), (1, 3)])
    poly1 = Poly([(2, 2), (5, 2), (5, 5), (2, 5)])
    poly2 = Poly([(10, 1), (11, 1), (11, 4), (10, 4)])
    poly3 = Poly([(9, 2), (12, 2), (12, 5), (9, 5)])
    poly4 = Poly([(3, 4), (8, 4), (8, 6), (3, 6)])
    poly5 = Poly([(10, 3), (13, 3), (13, 4), (10, 4)])
    poly6 = Poly([(1, 7), (2, 7), (2, 8), (1, 8)])
    poly7 = Poly([(7, 7), (13, 7), (13, 8), (7, 8)])
    poly8 = Poly([(10, 3), (11, 3), (11, 6), (10, 6)])
    poly9 = Poly([(0, 6.5), (3, 6.5), (3, 9), (0, 9)])

    manager = ManagePolygons()
    manager.add_polygon(poly0)
    manager.add_polygon(poly1)
    manager.add_polygon(poly2)
    manager.add_polygon(poly3)
    manager.add_polygon(poly4)
    manager.add_polygon(poly5)
    manager.add_polygon(poly6)
    manager.add_polygon(poly7)
    manager.add_polygon(poly8)
    manager.add_polygon(poly9)

    groups = manager.find_connected_components()

    merged_polygons = manager.find_intersecting_groups()
    external_coordinates = manager.get_external_coordinates(merged_polygons)

    path = BresenhamPath(external_coordinates)

    points = path.get_all_points()

    # sorted from bottom to top!
    points = sorted(points, key=lambda p: (p[1], p[0]))

    # [[first row from bottom (x,y) where x and y are the value of the grid], [second row from bottom], ...]
    # every vector inside the list  is a row of the big grid
    blockGroup = Grid(points, [1, 4, 2, 7, 5], 0.01)
    print(blockGroup.grid)

    activator = NozzleActivator(blockGroup.grid)
    print(activator.binary_sequences())
